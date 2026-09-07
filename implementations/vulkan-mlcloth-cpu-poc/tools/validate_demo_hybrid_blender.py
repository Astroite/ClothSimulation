"""Evaluate the editable hybrid collision rig against independent animation references."""
from pathlib import Path
import json, struct, sys, argparse
import bpy, numpy as np
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bake_demo_blender import reference, calibration
OUT=Path(__file__).resolve().parents[1]/'.work/demo'
parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=OUT)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
OUT=args.directory.resolve()
def read(name):
    raw=(OUT/name).read_bytes();n=struct.unpack_from('<Q',raw,8)[0];meta=json.loads(raw[16:16+n]);payload=raw[16+n:]
    return meta,{k:np.frombuffer(payload,dtype=e['dtype'],offset=e['offset'],count=e['bytes']//4).reshape(e['shape']).copy() for k,e in meta['arrays'].items()}
body_meta,body=read('character.dmp');_,proxy=read('collision-hybrid.dmp');_,capsules=read('leg-capsules.dmp')
names=body_meta['bones'];name_index={name:i for i,name in enumerate(names)}
_,rest,_=reference()
bpy.ops.wm.open_mainfile(filepath=str(OUT/'C10032_HybridCollision.blend'))
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');obj=bpy.data.objects['C10032_STM_Hybrid']
capsule_objects=sorted([o for o in bpy.context.scene.objects if o.name.startswith('Leg capsule ')],key=lambda o:o.name)
if len(capsule_objects)!=len(capsules['bone']):raise RuntimeError('Editable capsule count mismatch')
f,corrections,_=calibration(rig,rest)
world=Matrix(((.01,0,0,0),(0,0,.01,0),(0,-.01,0,0),(0,0,0,1)))
inverse=np.concatenate([body['inverse_bind'],np.tile([[[0,0,0,1]]],(len(names),1,1))],axis=1)
points=np.column_stack([proxy['positions'],np.ones(len(proxy['positions']))]);report={'passed':True,'samples':[]}
for clip in ['idle','walk','run','sprint','jump','turn','complex']:
    _,ref=read(clip+'.reference.dmp')
    for sample,frame in enumerate(ref['frames'].ravel()):
        desired={name:rig.matrix_world.inverted() @ f.inverted() @ Matrix(ref['bones'][sample,name_index[name]].tolist()) @ correction.inverted() for name,correction in corrections.items()}
        for bone in rig.pose.bones:
            if bone.name not in desired:continue
            kwargs={}
            if bone.parent:kwargs=dict(parent_matrix=desired[bone.parent.name],parent_matrix_local=bone.parent.bone.matrix_local)
            bone.matrix_basis=bone.bone.convert_local_to_pose(desired[bone.name],bone.bone.matrix_local,invert=True,**kwargs)
        bpy.context.view_layer.update()
        evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
        transform=world @ f @ obj.matrix_world
        actual=np.array([(transform @ v.co)[:] for v in mesh.vertices]);evaluated.to_mesh_clear()
        palette=np.array(world) @ ref['bones'][sample] @ inverse
        expected=np.einsum('nkij,nj,nk->ni',palette[proxy['bone_ids']],points,proxy['weights'])[:,:3]
        error=float(np.linalg.norm(actual-expected,axis=1).max())
        capsule_error=0.
        for index,visual in enumerate(capsule_objects):
            evaluated=visual.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh();transform=world @ f @ visual.matrix_world
            vertices=np.array([(transform @ v.co)[:] for v in mesh.vertices]);evaluated.to_mesh_clear()
            actual_ends=np.array([vertices[6*16:7*16].mean(axis=0),vertices[7*16:8*16].mean(axis=0)])
            bone=int(capsules['bone'].ravel()[index]);matrix=np.array(world)@ref['bones'][sample,bone]
            expected_ends=np.array([(matrix@np.r_[capsules[key][index],1])[:3] for key in ['a','b']])
            capsule_error=max(capsule_error,float(np.linalg.norm(actual_ends-expected_ends,axis=1).max()))
        passed=bool(np.isfinite(actual).all() and error<=.001 and capsule_error<=.001);report['passed'] &= passed
        report['samples'].append(dict(clip=clip,frame=int(frame),stm_blender_skin_max_error_m=error,capsule_blender_endpoint_max_error_m=capsule_error,passed=passed))
(OUT/'hybrid-blender-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
if not report['passed']:raise RuntimeError('Editable hybrid collision differs from runtime skinning by more than 1 mm')
