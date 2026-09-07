"""Rebuild a closed full-body collision surface; preserve source body and old STM proxy.

blender --factory-startup -b --python-exit-code 1 -P tools/build_demo_collision.py
"""
from pathlib import Path
import json,struct,sys,argparse
import bpy,bmesh,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bake_demo_blender import pack
OUT=Path(__file__).resolve().parents[1]/'.work/demo'
parser=argparse.ArgumentParser();parser.add_argument('--vertices',type=int,default=2800);parser.add_argument('--voxel',type=float,default=.008)
parser.add_argument('--output',default='collision-full');args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
if not 500<=args.vertices<=8000 or not .002<=args.voxel<=.02:raise ValueError('Invalid collision resolution')
def read(name):
    raw=(OUT/name).read_bytes();n=struct.unpack_from('<Q',raw,8)[0];meta=json.loads(raw[16:16+n]);payload=raw[16+n:]
    return meta,{k:np.frombuffer(payload,dtype=e['dtype'],offset=e['offset'],count=e['bytes']//4).reshape(e['shape']).copy() for k,e in meta['arrays'].items()}
meta,body=read('character.dmp')
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
mesh=bpy.data.meshes.new('Complete body collision');mesh.from_pydata(body['positions'],[],body['triangles']);mesh.update()
obj=bpy.data.objects.new('C10032_FullBodyCollision',mesh);bpy.context.collection.objects.link(obj);bpy.context.view_layer.objects.active=obj;obj.select_set(True)
bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
remesh=obj.modifiers.new('Close body surface','REMESH');remesh.mode='VOXEL';remesh.voxel_size=args.voxel;remesh.use_smooth_shade=True
bpy.ops.object.modifier_apply(modifier=remesh.name)
remeshed_vertices=len(obj.data.vertices)
decimate=obj.modifiers.new('Collision resolution','DECIMATE');decimate.ratio=min(1,args.vertices/max(1,remeshed_vertices));decimate.use_collapse_triangulate=True
bpy.ops.object.modifier_apply(modifier=decimate.name)
bm=bmesh.new();bm.from_mesh(obj.data);bmesh.ops.triangulate(bm,faces=list(bm.faces));bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(obj.data);bm.free()
tri=np.array([p.vertices[:] for p in obj.data.polygons],dtype=np.uint32)
old=np.array([v.co[:] for v in obj.data.vertices],dtype=np.float32)
tree=BVHTree.FromPolygons(body['positions'].tolist(),body['triangles'].tolist(),all_triangles=True)
projected=[]
for vertex in old:
    q,n,face,distance=tree.find_nearest(Vector(vertex));projected.append(tuple(q+n*.001))
projected=np.array(projected,dtype=np.float32);fraction=np.ones(len(old),dtype=np.float32);new=projected.copy()
old_normals=np.cross(old[tri[:,1]]-old[tri[:,0]],old[tri[:,2]]-old[tri[:,0]])
for _ in range(24):
    normals=np.cross(new[tri[:,1]]-new[tri[:,0]],new[tri[:,2]]-new[tri[:,0]])
    invalid=np.einsum('ij,ij->i',normals,old_normals)<=0
    if not invalid.any():break
    affected=np.unique(tri[invalid]);fraction[affected]*=.5;new=old+(projected-old)*fraction[:,None]
ids=[];weights=[]
for vertex in new:
    q,n,face,distance=tree.find_nearest(Vector(vertex));indices=body['triangles'][face];a,b,c=body['positions'][indices]
    uv=np.linalg.lstsq(np.column_stack((b-a,c-a)),np.array(q)-a,rcond=None)[0];bary=np.maximum([1-uv.sum(),*uv],0);bary/=bary.sum()
    influences={}
    for v,w in zip(indices,bary):
        for bone,weight in zip(body['bone_ids'][v],body['weights'][v]):influences[int(bone)]=influences.get(int(bone),0)+float(w*weight)
    selected=sorted(influences.items(),key=lambda pair:-pair[1])[:32];total=sum(w for _,w in selected);selected += [(0,0)]*(32-len(selected))
    ids.append([i for i,w in selected]);weights.append([w/total for i,w in selected])
normals=np.zeros_like(new);face_normals=np.cross(new[tri[:,1]]-new[tri[:,0]],new[tri[:,2]]-new[tri[:,0]])
for face,normal in zip(tri,face_normals):
    for v in face:normals[v]+=normal
normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1);unique,counts=np.unique(edges,axis=0,return_counts=True)
closed=bool((counts==2).all());flips=int((np.einsum('ij,ij->i',face_normals,old_normals)<=0).sum())
if not closed or flips or len(new)>8192 or len(tri)>16384:raise RuntimeError('Invalid collision topology or GPU capacity exceeded')
proxy_tree=BVHTree.FromPolygons(new.tolist(),tri.tolist(),all_triangles=True)
distances=[proxy_tree.find_nearest(Vector(v))[3] for v in body['positions']]
report=dict(vertices=len(new),triangles=len(tri),closed=closed,flipped_faces=flips,remeshed_vertices=remeshed_vertices,voxel_m=args.voxel,
            limited_vertices=int((fraction<1).sum()),body_surface_distance_p95_m=float(np.quantile(distances,.95)),body_surface_distance_max_m=max(distances),
            bounds_min=new.min(axis=0).tolist(),bounds_max=new.max(axis=0).tolist(),status='full-body candidate; animated and visible-contact validation pending')
pack(OUT/(args.output+'.dmp'),dict(influences=32,closed=True,source='character.dmp',construction=report),positions=new,normals=normals,
     uv=np.zeros((len(new),2)),triangles=tri,material=np.zeros(len(tri),dtype=np.uint32),bone_ids=np.array(ids,dtype=np.uint32),weights=np.array(weights),inverse_bind=body['inverse_bind'])
for vertex,position in zip(obj.data.vertices,new):vertex.co=position
obj.data.update();bpy.ops.wm.save_as_mainfile(filepath=str(OUT/('C10032_Collision.blend' if args.output=='collision-full' else args.output+'.blend')))
(OUT/(args.output+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
manifest=json.loads((OUT/'demo.json').read_text(encoding='utf-8'));manifest['collision']=args.output+'.dmp';(OUT/('demo-full-collision.json' if args.output=='collision-full' else 'demo-'+args.output+'.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
