"""Bake exact surface attachments for physical cloth pins, preserving the ML output contract."""
from pathlib import Path
import json,struct,sys
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def surface_attachments(body,cloth,pinned,clearance=.005):
    points=body['reference_skin'];tri=body['triangles'];tree=BVHTree.FromPolygons(points.tolist(),tri.tolist(),all_triangles=True)
    vertices=[];coordinates=[];offsets=[];corrected=cloth.copy();changes=[]
    for i,point in enumerate(cloth):
        q,normal,face,distance=tree.find_nearest(Vector(point));ids=tri[face];a,b,c=points[ids]
        uv=np.linalg.lstsq(np.column_stack((b-a,c-a)),np.array(q)-a,rcond=None)[0];bc=np.maximum([1-uv.sum(),*uv],0);bc/=bc.sum()
        tangent=b-a;tangent/=np.linalg.norm(tangent);n=np.cross(b-a,c-a);n/=np.linalg.norm(n);bitangent=np.cross(n,tangent)
        origin=bc@points[ids];delta=point-origin;offset=np.array([delta@tangent,delta@bitangent,max(delta@n,clearance)])
        vertices.append(ids);coordinates.append(bc);offsets.append(offset)
        if pinned[i]:corrected[i]=origin+tangent*offset[0]+bitangent*offset[1]+n*offset[2];changes.append(np.linalg.norm(corrected[i]-point))
    return corrected,dict(attachment_vertices=np.array(vertices,dtype=np.uint32),attachment_bary=np.array(coordinates),attachment_offset=np.array(offsets)),dict(
        version=1,clearance_m=clearance,pins=int(np.count_nonzero(pinned)),max_rest_pin_adjustment_m=float(max(changes)),
        definition='barycentric independently skinned body vertices plus transported tangent-space offset')

def main():
    sys.path.insert(0,str(Path(__file__).resolve().parent));from bake_demo_blender import pack
    out=Path(__file__).resolve().parents[1]/'.work/demo'
    def read(name):
        raw=(out/name).read_bytes();n=struct.unpack_from('<Q',raw,8)[0];meta=json.loads(raw[16:16+n]);payload=raw[16+n:]
        return meta,{k:np.frombuffer(payload,dtype=e['dtype'],offset=e['offset'],count=e['bytes']//4).reshape(e['shape']).copy() for k,e in meta['arrays'].items()}
    _,body=read('character.dmp');meta,cloth=read('cloth.dmp');original=cloth['positions'].copy()
    cloth['positions'],attachments,report=surface_attachments(body,original,cloth['pinned']);cloth.update(attachments)
    # A hard attachment is invalid if another body region encloses its target.
    # Release such armpit/contact pins using independent poses; leave physics to resolve them.
    original_pinned=cloth['pinned'].copy();released=set()
    for clip in ['idle','walk','run','sprint','jump','turn','complex']:
        _,ref=read(clip+'.reference.dmp')
        for reference in ref['skin']:
            tree=BVHTree.FromPolygons(reference.tolist(),body['triangles'].tolist(),all_triangles=True)
            for i in np.where(original_pinned>0)[0]:
                a,b,c=reference[cloth['attachment_vertices'][i]];n=np.cross(b-a,c-a);n/=np.linalg.norm(n);t=b-a;t/=np.linalg.norm(t);bt=np.cross(n,t)
                bc=cloth['attachment_bary'][i];offset=cloth['attachment_offset'][i];target=a*bc[0]+b*bc[1]+c*bc[2]+t*offset[0]+bt*offset[1]+n*offset[2]
                q,normal,face,distance=tree.find_nearest(Vector(target))
                if (Vector(target)-q).dot(normal)<-.001:released.add(int(i))
    cloth['source_pinned']=original_pinned
    for i in released:cloth['pinned'][i]=0
    report['released_conflicting_pins']=sorted(released);report['pins']=int(np.count_nonzero(cloth['pinned']))
    pack(out/'cloth-attached.dmp',dict(model_sha256=meta['model_sha256'],pin_binding=report),**cloth)
    manifest=json.loads((out/'demo-full-collision.json').read_text(encoding='utf-8'));manifest['cloth']='cloth-attached.dmp'
    (out/'demo-attached.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');(out/'cloth-attachment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
