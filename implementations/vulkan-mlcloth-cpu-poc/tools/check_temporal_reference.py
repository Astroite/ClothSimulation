"""Blender: evaluate every reference frame, convergence and independent geometry.

blender -b --factory-startup --python-exit-code 1 -P tools/check_temporal_reference.py --
  --coarse DIR480 --fine DIR960 --output gate.json [--solver-report solver.json]
Thresholds are fixed engineering gates, not evidence of exact physical ground truth.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_ray_tri
sys.path.insert(0,str(Path(__file__).resolve().parent))
from temporal_reference import load_reference,digest
from demo_contact_metrics import solid_metrics

def geometry(x,rest,tri,pins,body,body_tri,caps):
    rest_area=np.linalg.norm(np.cross(rest[tri[:,1]]-rest[tri[:,0]],rest[tri[:,2]]-rest[tri[:,0]]),axis=-1)
    area=np.linalg.norm(np.cross(x[tri[:,1]]-x[tri[:,0]],x[tri[:,2]]-x[tri[:,0]]),axis=-1)/np.maximum(rest_area,1e-20)
    edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=-1),axis=0)
    lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=-1)
    strain=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=-1)/lengths
    result=dict(area_ratio_min=float(area.min()),degenerate_fraction=float((area<.05).mean()),stretch_ratio_p95=float(np.quantile(strain,.95)),stretch_ratio_max=float(strain.max()))
    if len(body_tri):
        tree=BVHTree.FromPolygons(body.tolist(),body_tri.tolist(),all_triangles=True)
        distances=[tree.find_nearest(Vector(p))[3] for p in x]
        result['body']=solid_metrics(tree,x,body,distances,pins)
    else:result['body']=dict(penetration_max_m=0,penetrating_vertices=0,ambiguous_vertices=0)
    depths=np.zeros(len(x))
    for row in caps:
        a,b,r=row[:3],row[3:6],row[6];axis=b-a;t=np.clip((x-a)@axis/max(float(axis@axis),1e-16),0,1)
        depths=np.maximum(depths,r-np.linalg.norm(x-a-t[:,None]*axis,axis=-1))
    result['capsules']=dict(penetration_max_m=float(depths.max()),penetrating_vertices=int((depths>1e-6).sum()),
        penetrating_pins=int(((depths>1e-6)&pins.astype(bool)).sum()))
    # Non-coplanar actual segment/triangle intersections, not AABB overlap counts.
    points=[Vector(p) for p in x];tree=BVHTree.FromPolygons(points,tri.tolist(),all_triangles=True);hits=0;coplanar=0
    for a,b in tree.overlap(tree):
        if a>=b or set(tri[a])&set(tri[b]):continue
        found=False
        for source,target in ((tri[a],tri[b]),(tri[b],tri[a])):
            q=[points[int(i)] for i in target]
            normal=(q[1]-q[0]).cross(q[2]-q[0])
            if normal.length_squared<1e-20:continue
            if all(abs((points[int(i)]-q[0]).dot(normal.normalized()))<1e-7 for i in source):coplanar+=1;break
            for k in range(3):
                p0,p1=points[int(source[k])],points[int(source[(k+1)%3])];direction=p1-p0
                hit=intersect_ray_tri(q[0],q[1],q[2],direction,p0,True)
                if hit is not None and 1e-6<(hit-p0).length<direction.length-1e-6:found=True;break
            if found:break
        hits+=int(found)
    result['self_intersections']=hits;result['coplanar_candidates_unresolved']=coplanar
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--coarse',type=Path,required=True);p.add_argument('--fine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--solver-report',type=Path)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    r0,r1=load_reference(a.coarse),load_reference(a.fine);v0,v1=r0.arrays,r1.arrays
    if (r0.metadata['hz'],r1.metadata['hz'])!=(480,960):raise ValueError('expected paired 480/960 Hz references')
    for key in ('clip','speed','in_place','warmup','physics_config'):
        if r0.metadata[key]!=r1.metadata[key]:raise ValueError('reference config mismatch: '+key)
    for key in ('rest','triangles','mass','pinned','body_positions','body_normals','pin_targets','collider_positions','capsules','surface_ids'):
        if not np.array_equal(v0[key],v1[key]):raise ValueError('reference exogenous input mismatch: '+key)
    if not np.array_equal(v0['x'][0],v1['x'][0]) or not np.array_equal(v0['v'][0],v1['v'][0]):raise ValueError('different reference initial states')
    thresholds=dict(position_rmse_m=.001,position_max_m=.01,velocity_rmse_m_s=.1,penetration_max_m=.003,degenerate_fraction=.005,
        self_intersections=0,ambiguous_body_vertices=0)
    dp=v0['x']-v1['x'];dv=v0['v']-v1['v'];convergence=dict(position_rmse_m=float(np.sqrt(np.mean(np.sum(dp*dp,axis=-1)))),
        position_max_m=float(np.linalg.norm(dp,axis=-1).max()),velocity_rmse_m_s=float(np.sqrt(np.mean(np.sum(dv*dv,axis=-1)))))
    per_frame=[]
    for label,values in [('480',v0),('960',v1)]:
        for frame,x in enumerate(values['x']):
            i=min(frame,len(values['capsules'])-1);k=0 if frame<len(values['capsules']) else -1
            metrics=geometry(x,values['rest'],values['triangles'],values['pinned'],values['collider_positions'][i,k],values['collider_triangles'],values['capsules'][i,k])
            per_frame.append(dict(hz=int(label),frame=frame,time=frame/30,**metrics));print(f'Geometry {label} Hz frame {frame}',flush=True)
    failures=[]
    for key,value in convergence.items():
        if value>thresholds[key]:failures.append(f'{key}: {value:.8g} > {thresholds[key]}')
    for row in per_frame:
        pen=max(row['body']['penetration_max_m'],row['capsules']['penetration_max_m'])
        if pen>thresholds['penetration_max_m']:failures.append(f"{row['hz']}Hz frame {row['frame']}: penetration {pen:.8g}m")
        if row['degenerate_fraction']>thresholds['degenerate_fraction']:failures.append(f"{row['hz']}Hz frame {row['frame']}: degenerate fraction {row['degenerate_fraction']:.8g}")
        if row['self_intersections'] or row['coplanar_candidates_unresolved'] or row['body']['ambiguous_vertices']:
            failures.append(f"{row['hz']}Hz frame {row['frame']}: intersecting or unresolved geometry")
    solver=json.loads(a.solver_report.read_text()) if a.solver_report and a.solver_report.exists() else {}
    forward=bool(solver.get('reference_forward_passed'));grad=bool(solver.get('contact_gradients_passed'));quality=not failures
    result=dict(version=1,reference_quality_passed=quality,reference_forward_passed=forward,contact_gradients_passed=grad,
        passed=quality and forward and grad,training_allowed=quality and forward and grad,thresholds=thresholds,convergence=convergence,
        failures=failures,frames=per_frame,solver_report=solver,scope='Short paired reference diagnostic; passing is necessary but not sufficient for full dataset qualification',
        references=[dict(path=str(path),metadata_sha256=digest(path/'metadata.json'),frames_sha256=digest(path/'frames.bin')) for path in (a.coarse,a.fine)])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('frames','solver_report')}))
if __name__=='__main__':main()
