"""Higher-detail STM candidate for the torso/arms/hips, with leg capsules retained."""
from pathlib import Path
import json, struct, sys, argparse
import bpy, bmesh, numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent));from bake_demo_blender import pack
OUT=Path(__file__).resolve().parents[1]/'.work/demo'
parser=argparse.ArgumentParser();parser.add_argument('--source',default='collision-full.dmp');parser.add_argument('--output',default='collision-regional')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
raw=(OUT/args.source).read_bytes();n=struct.unpack_from('<Q',raw,8)[0];meta=json.loads(raw[16:16+n]);payload=raw[16+n:]
source={k:np.frombuffer(payload,dtype=e['dtype'],offset=e['offset'],count=e['bytes']//4).reshape(e['shape']).copy() for k,e in meta['arrays'].items()}
mesh=bpy.data.meshes.new('Regional STM');mesh.from_pydata(source['positions'],[],source['triangles']);mesh.update()
bm=bmesh.new();bm.from_mesh(mesh)
bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),dist=1e-6,plane_co=(0,.60,0),plane_no=(0,1,0),clear_inner=True)
boundary=[edge for edge in bm.edges if edge.is_boundary]
if boundary:bmesh.ops.holes_fill(bm,edges=boundary,sides=0)
bmesh.ops.triangulate(bm,faces=list(bm.faces));bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
points=np.array([v.co[:] for v in mesh.vertices]);triangles=np.array([p.vertices[:] for p in mesh.polygons],dtype=np.uint32)
tree=BVHTree.FromPolygons(source['positions'].tolist(),source['triangles'].tolist(),all_triangles=True)
ids=[];weights=[]
for point in points:
    q,_,face,_=tree.find_nearest(Vector(point));vertices=source['triangles'][face];a,b,c=source['positions'][vertices]
    uv=np.linalg.lstsq(np.column_stack((b-a,c-a)),np.array(q)-a,rcond=None)[0];bc=np.maximum([1-uv.sum(),*uv],0);bc/=bc.sum();influences={}
    for vertex,w in zip(vertices,bc):
        for bone,value in zip(source['bone_ids'][vertex],source['weights'][vertex]):influences[int(bone)]=influences.get(int(bone),0)+float(w*value)
    selected=sorted(influences.items(),key=lambda x:-x[1])[:32];total=sum(w for _,w in selected);selected += [(0,0)]*(32-len(selected))
    ids.append([bone for bone,_ in selected]);weights.append([w/total for _,w in selected])
edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1);_,counts=np.unique(edges,axis=0,return_counts=True)
if not (counts==2).all() or len(points)+20>8192 or len(triangles)>16384:raise RuntimeError('Regional STM fails topology/capacity checks')
report=dict(vertices=len(points),triangles=len(triangles),closed=True,source=args.source,leg_cut_height_m=.60,status='candidate; animated contact quality validation pending')
pack(OUT/(args.output+'.dmp'),dict(influences=32,construction=report),positions=points,triangles=triangles,normals=np.array([v.normal[:] for v in mesh.vertices]),
     uv=np.zeros((len(points),2)),material=np.zeros(len(triangles),dtype=np.uint32),bone_ids=np.array(ids,dtype=np.uint32),weights=np.array(weights))
manifest=json.loads((OUT/'demo.json').read_text(encoding='utf-8'));manifest['collision']=args.output+'.dmp'
(OUT/('demo-regional.json' if args.output=='collision-regional' else 'demo-'+args.output+'.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8');(OUT/(args.output+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
