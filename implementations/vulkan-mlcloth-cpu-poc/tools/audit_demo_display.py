"""Read-only rest-space audit; never publishes a display mapping."""
import bpy,sys,json,struct
from pathlib import Path
import numpy as np
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from bake_demo_blender import reference,calibration,WORLD
OUT=ROOT/'.work/demo'
def read(name):
    raw=(OUT/name).read_bytes();n=struct.unpack_from('<Q',raw,8)[0];meta=json.loads(raw[16:16+n]);data=raw[16+n:]
    return {k:np.frombuffer(data,dtype=e['dtype'],offset=e['offset'],count=e['bytes']//4).reshape(e['shape']).copy() for k,e in meta['arrays'].items()}
_,rest,_=reference();f,_,_=calibration(bpy.data.objects['Root'],rest)
cloth=read('cloth.dmp');body=read('character.dmp')
tree=BVHTree.FromPolygons(cloth['positions'].tolist(),cloth['triangles'].tolist(),all_triangles=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
source='F:/ArtWorks/Cloth/CH_10032_V3/SM_C10032_Cloth_01.fbx'
bpy.ops.import_scene.fbx(filepath=source,use_anim=False)
obj=bpy.data.objects['SM_C10032_Cloth_01'];matrix=WORLD@f@obj.matrix_world
points=np.array([(matrix@v.co)[:] for v in obj.data.vertices]);near=np.array([tree.find_nearest(matrix@v.co)[0][:] for v in obj.data.vertices])
from mathutils import Vector
ref=read('complex.reference.dmp');inverse=np.concatenate([body['inverse_bind'],np.tile([[[0,0,0,1]]],(len(body['inverse_bind']),1,1))],axis=1)
palette=np.array(WORLD)@ref['bones'][0]@inverse
body_tree=BVHTree.FromPolygons(body['positions'].tolist(),body['triangles'].tolist(),all_triangles=True)
posed=[]
for point in points:
    q,n,face,distance=body_tree.find_nearest(Vector(point));ids=body['triangles'][face];a,b,c=body['positions'][ids]
    uv=np.linalg.lstsq(np.column_stack((b-a,c-a)),np.array(q)-a,rcond=None)[0];bc=np.maximum([1-uv.sum(),*uv],0);bc/=bc.sum()
    blended=np.einsum('nkij,nk,n->ij',palette[body['bone_ids'][ids]],body['weights'][ids],bc)
    posed.append((blended@np.r_[point,1])[:3])
posed=np.array(posed);posed_distances=np.array([tree.find_nearest(Vector(p))[3] for p in posed])
np.savez_compressed(OUT/'display-rest-audit.npz',source=points,posed=posed,nearest=near,cloth=cloth['positions'],body=body['reference_skin'],body_bind=body['positions'])
report={'source':source,'armatures':[o.name for o in bpy.context.scene.objects if o.type=='ARMATURE'],
        'vertex_groups':len(obj.vertex_groups),'modifiers':[m.type for m in obj.modifiers],'bands':[]}
report['body_bind_to_reference_distance_m']=np.percentile(posed_distances,[50,95,100]).tolist()
for lo,hi in [(0,.5),(.5,1),(1,1.2),(1.2,1.4),(1.4,2)]:
    mask=(points[:,1]>=lo)&(points[:,1]<hi);distances=np.linalg.norm(points[mask]-near[mask],axis=1)
    report['bands'].append(dict(height=[lo,hi],vertices=int(mask.sum()),distance_m=np.percentile(distances,[50,95,100]).tolist() if mask.any() else []))
(OUT/'display-rest-audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2),flush=True)
