"""Build versioned, full-skeleton demo assets in Blender 4.5.

blender -b <repaired 60fps preview.blend> -P tools/bake_demo_blender.py -- --clips all
The source blend/FBX is never saved over. Runtime coordinates are UE (x,z,-y)/100.
Bone-axis corrections are derived from reference poses, not Euler guesses.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import shutil
import os

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from bake_mlcloth_body import compose_bind_pose, read_sectioned, section_values
from demo_pose_blend import mix_pose_local,close_loop_local

OUTPUT = ROOT / '.work/demo'
SOURCE = OUTPUT / 'source'
MODEL = ROOT / '.work/runtime/model_NeuralRes4_NeuralRes4_final.enc'
SKELETON = ROOT.parent / 'vulkan-gnn-poc/.work/ch10032_library/data/skeleton.json'
WORLD = Matrix(((.01, 0, 0, 0), (0, 0, .01, 0), (0, -.01, 0, 0), (0, 0, 0, 1)))
ANCHORS = ['Root_M', 'Chest_M', 'Head_M', 'Wrist_L', 'Wrist_R', 'Ankle_L', 'Ankle_R',
           'Shoulder_L', 'Shoulder_R', 'Hip_L', 'Hip_R']


def pack(path, metadata, **arrays):
    payload = bytearray()
    table = {}
    for key, values in arrays.items():
        a = np.ascontiguousarray(values)
        if a.dtype.kind == 'f':
            a = a.astype('<f4')
            if not np.isfinite(a).all():
                raise ValueError('Non-finite array: ' + key)
        elif a.dtype.kind in 'iu':
            a = a.astype('<u4')
        else:
            raise TypeError(key)
        while len(payload) % 16:
            payload.append(0)
        table[key] = dict(offset=len(payload), shape=list(a.shape), dtype=a.dtype.str, bytes=a.nbytes)
        payload.extend(a.tobytes())
    header = json.dumps(dict(version=1, arrays=table, sha256=hashlib.sha256(payload).hexdigest(), **metadata),
                        ensure_ascii=False, separators=(',', ':')).encode('utf8')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(b'DMPACK01' + struct.pack('<Q', len(header)) + header + payload)
    temporary.replace(path)
    print('DEMO_PACK', path.name, len(payload), flush=True)


def reference():
    records = json.loads(SKELETON.read_text(encoding='utf8'))['bones']
    bind, residual = compose_bind_pose(records)
    matrices = {}
    for name, (t, xyzw) in bind.items():
        x, y, z, w = xyzw
        m = Quaternion((w, x, y, z)).to_matrix().to_4x4()
        m.translation = Vector(t)
        matrices[name] = m
    return records, matrices, residual


def calibration(rig, rest):
    # Fit a similarity reflection from anatomical positions; reject scale/shear mistakes.
    b = np.array([list((rig.matrix_world @ rig.data.bones[n].matrix_local).translation) + [1] for n in ANCHORS])
    u = np.array([list(rest[n].translation) for n in ANCHORS])
    raw = np.linalg.lstsq(b, u, rcond=None)[0].T
    left, singular, right = np.linalg.svd(raw[:, :3])
    scale = float(np.mean(singular))
    rotation = left @ right
    translation = np.mean(u - b[:, :3] @ (scale * rotation).T, axis=0)
    f = np.eye(4); f[:3, :3] = scale * rotation; f[:3, 3] = translation
    err = float(np.linalg.norm(b @ f[:3].T - u, axis=1).max())
    if err > .01 or np.ptp(singular) > .01 * scale:
        raise ValueError(f'Incompatible source skeleton: fit error {err} cm, scales {singular}')
    f = Matrix(f.tolist())
    correction = {n: (rig.matrix_world @ rig.data.bones[n].matrix_local).inverted() @ f.inverted() @ m
                  for n, m in rest.items() if n in rig.data.bones}
    return f, correction, err


def pose_ue(rig, f, correction, names, rest):
    return [f @ rig.matrix_world @ rig.pose.bones[n].matrix @ correction[n]
            if n in correction else rest[n].copy() for n in names]


def trs(matrix):
    p, q, s = matrix.decompose()
    if max(abs(v - 1) for v in s) > .002:
        raise ValueError(f'Unexpected bone scale {tuple(s)}')
    return (*p, q.x, q.y, q.z, q.w)


def mesh_data(obj, f, names):
    """Split only normal/UV/material seams; retain all source skin influences."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    lookup = {n: i for i, n in enumerate(names)}
    groups = {g.index: lookup[g.name] for g in obj.vertex_groups if g.name in lookup}
    influences = []
    width = max(sum(1 for g in v.groups if g.group in groups and g.weight > 1e-8) for v in mesh.vertices)
    width = max(1, width)
    for vertex in mesh.vertices:
        values = sorted(((groups[g.group], g.weight) for g in vertex.groups if g.group in groups and g.weight > 1e-8), key=lambda p: -p[1])
        total = sum(w for _, w in values)
        if total <= 0:
            values = [(lookup['Root_M'], 1.0)]; total = 1
        influences.append([(b, w / total) for b, w in values])
    matrix = WORLD @ f @ obj.matrix_world
    normal_matrix = matrix.to_3x3().inverted().transposed()
    positions, normals, uv, bone_ids, weights, indices, materials = [], [], [], [], [], [], []
    source_vertex = []; cache = {}
    for triangle in mesh.loop_triangles:
        face = []
        for loopid, vertexid in zip(triangle.loops, triangle.vertices):
            normal = mesh.corner_normals[loopid].vector
            tex = mesh.uv_layers.active.data[loopid].uv if mesh.uv_layers.active else Vector((0, 0))
            key = (vertexid, tuple(round(v, 6) for v in normal), tuple(tex), triangle.material_index)
            if key not in cache:
                cache[key] = len(positions)
                positions.append(tuple(matrix @ mesh.vertices[vertexid].co))
                normals.append(tuple((normal_matrix @ normal).normalized()))
                uv.append(tuple(tex)); source_vertex.append(vertexid)
                values = influences[vertexid]
                bone_ids.append([b for b, w in values] + [0] * (width-len(values)))
                weights.append([w for b, w in values] + [0] * (width-len(values)))
            face.append(cache[key])
        if matrix.to_3x3().determinant() < 0:
            face[1], face[2] = face[2], face[1]
        indices.append(face)
        matname = mesh.materials[triangle.material_index].name if mesh.materials and mesh.materials[triangle.material_index] else ''
        materials.append(1 if any(k in matname.lower() for k in ['head','face','eye','teeth']) else 0)
    return dict(positions=np.array(positions), normals=np.array(normals), uv=np.array(uv),
                bone_ids=np.array(bone_ids,dtype=np.uint32), weights=np.array(weights),
                triangles=np.array(indices,dtype=np.uint32), material=np.array(materials,dtype=np.uint32)), influences


def skin(vertices, ids, weights, matrices):
    mats = np.array([[list(row) for row in m] for m in matrices])
    points = np.column_stack((vertices, np.ones(len(vertices))))
    out = np.zeros((len(vertices), 3))
    for j in range(ids.shape[1]):
        out += np.einsum('nij,nj->ni', mats[ids[:,j],:3], points) * weights[:,j,None]
    return out


def export_character(rig, body, rest, names, f, corrections):
    arrays, _ = mesh_data(body, f, names)
    inverse_bind = [(WORLD @ rest[n]).inverted() for n in names]
    arrays['inverse_bind'] = np.array([[list(row) for row in m][:3] for m in inverse_bind])
    # Export independently evaluated Blender surface for actual skinning validation.
    bpy.context.scene.frame_set(1)
    pose = pose_ue(rig, f, corrections, names, rest)
    skin_mats = [WORLD @ m @ inv for m, inv in zip(pose,inverse_bind)]
    posed = skin(arrays['positions'], arrays['bone_ids'], arrays['weights'], skin_mats)
    bm = body.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    # Source and evaluated mesh have the same topology; seam copies have matching nearest rest position.
    from mathutils.kdtree import KDTree
    tree = KDTree(len(body.data.vertices))
    for v in body.data.vertices: tree.insert(WORLD @ f @ body.matrix_world @ v.co,v.index)
    tree.balance()
    direct = np.array([list(WORLD @ f @ body.matrix_world @ bm.vertices[tree.find(Vector(p))[1]].co) for p in arrays['positions']])
    error = float(np.linalg.norm(posed-direct,axis=1).max())
    if error > .001:
        raise ValueError(f'Full body Blender skinning mismatch {error} m')
    arrays['reference_skin'] = direct
    pack(OUTPUT/'character.dmp', dict(bones=names, influences=arrays['weights'].shape[1],
         blender_skin_error_m=error, source=bpy.data.filepath), **arrays)

    # Cloth rest topology and vertex order remain exactly the locked model's.
    blob, sections = read_sectioned(ROOT/'.work/mesh/ch10032_cloth2607.mlmesh', b'MLMSH001')
    p, count = section_values(blob,sections,'positions','f',12)
    tri, _ = section_values(blob,sections,'triangles','I',12)
    mass, _ = section_values(blob,sections,'vertex_mass','f',4)
    pins, _ = section_values(blob,sections,'pin_mask','I',4)
    cloth_local = np.array(p).reshape(-1,3)
    root_pose = pose[names.index('Root_M')]
    cloth_world = np.array([tuple(WORLD @ root_pose @ Vector(p)) for p in cloth_local])
    # Skin-bind pinned cloth from barycentric body weights at the reference pose.
    triangles = arrays['triangles'].tolist()
    bvh = BVHTree.FromPolygons(posed.tolist(),triangles,all_triangles=True)
    ids=[]; weights=[]; bindlocal=[]; distances=[]
    for point in cloth_world:
        q, normal, ti, distance = bvh.find_nearest(Vector(point))
        a,b,c = triangles[ti]
        va,vb,vc = posed[[a,b,c]]
        edges=np.column_stack((vb-va,vc-va))
        t=np.linalg.lstsq(edges,np.array(q)-va,rcond=None)[0]
        bary=np.maximum([1-t.sum(),t[0],t[1]],0); bary/=sum(bary)
        values={}
        for v,w in zip((a,b,c),bary):
            for bone,weight in zip(arrays['bone_ids'][v],arrays['weights'][v]):
                values[int(bone)]=values.get(int(bone),0)+float(w*weight)
        ordered=sorted(values.items(),key=lambda p:-p[1])[:8]
        total=sum(w for b,w in ordered)
        ordered += [(0,0)]*(8-len(ordered))
        ids.append([b for b,w in ordered]); weights.append([w/total for b,w in ordered])
        bindlocal.append([tuple((WORLD @ pose[b]).inverted() @ Vector(point)) for b,w in ordered])
        distances.append(distance)
    pack(OUTPUT/'cloth.dmp',dict(model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
         pin_binding='barycentric full skeleton weights at T pose',max_pin_body_distance_m=float(np.max(np.array(distances)[np.array(pins)>0]))),
         positions=cloth_world,local_cm=cloth_local,triangles=np.array(tri,dtype=np.uint32).reshape(-1,3),
         mass=np.array(mass),pinned=np.array(pins,dtype=np.uint32),bone_ids=np.array(ids,dtype=np.uint32),
         weights=np.array(weights),bind_local=np.array(bindlocal))

    # Legacy upper-body STM only. The hybrid collision bake adds hips, abdomen and legs.
    proxies=[o for o in bpy.context.scene.objects if o.type=='MESH' and 'ClothCollision' in o.name]
    if not proxies: raise ValueError('Source blend has no collision proxy')
    proxy=proxies[0]
    cp,_=mesh_data(proxy,f,names)
    # Weld normal/material seams so collision topology has geometric adjacency.
    unique, inverse=np.unique(np.round(cp['positions'],7),axis=0,return_inverse=True)
    first=np.unique(inverse,return_index=True)[1]
    cp['triangles']=inverse[cp['triangles']].astype(np.uint32)
    for k in ['positions','normals','uv','bone_ids','weights']: cp[k]=cp[k][first]
    edgecount={}
    for a,b,c in cp['triangles']:
        for e in [(a,b),(b,c),(c,a)]:
            e=tuple(sorted(map(int,e))); edgecount[e]=edgecount.get(e,0)+1
    if any(v!=2 for v in edgecount.values()): raise ValueError('Collision proxy is not closed')
    pack(OUTPUT/'collision.dmp',dict(influences=cp['weights'].shape[1],closed=True,source=proxy.name),**cp)
    return error


def sample_source(rig, names, rest, fps=60, duration=None):
    f,c,error=calibration(rig,rest)
    action=rig.animation_data.action if rig.animation_data else None
    start,end=action.frame_range if action else (1,1)
    scene=bpy.context.scene
    source_fps=scene.render.fps/scene.render.fps_base
    count=max(2,round(((end-start)/source_fps if duration is None else duration)*fps)+1)
    frames=[]
    for i in range(count):
        frame=start+i/fps*source_fps
        scene.frame_set(int(frame),subframe=frame-int(frame))
        frames.append(np.array([[[v for v in row] for row in m] for m in pose_ue(rig,f,c,names,rest)]))
    return np.array(frames),error


def mix_pose(a,b,t,parents):
    return mix_pose_local(a,b,t,parents)


def close_loop(frames,root_idx,parents,blend_frames=18):
    return close_loop_local(frames,root_idx,parents,blend_frames)


def export_clip(key, frames, names, records, drivers, model_hash, warmup_frames=0):
    root=names.index('Root_M'); lookup={n:i for i,n in enumerate(names)}
    parents=[lookup.get(r['parent'],-1) for r in records]
    # Store local poses, interpolate them before FK to preserve bone lengths.
    local=np.zeros((len(frames),len(names),7),dtype=np.float32)
    for i,pose in enumerate(frames):
        for j in range(len(names)):
            m=Matrix(pose[j].tolist())
            if parents[j]>=0: m=Matrix(pose[parents[j]].tolist()).inverted() @ m
            local[i,j]=trs(m)
        if i%300==0: print('DEMO_ANIMATION',key,i,'/',len(frames),flush=True)
    # Normalize quaternion signs before interpolation.
    for i in range(1,len(local)):
        flip=np.einsum('ij,ij->i',local[i-1,:,3:],local[i,:,3:])<0
        local[i,flip,3:]*=-1
    delta=frames[-1,root,:3,3]-frames[0,root,:3,3]; delta[2]=0
    pack(OUTPUT/(key+'.dma'),dict(name=key,fps=60,bones=names,parents=parents,
         root=root,loop_start=warmup_frames,loop_end=len(frames)-1,root_delta_cm=delta.tolist(),
         model_sha256=model_hash),local=local,
         reference_bones=np.array(frames[::max(1,len(frames)//16)],dtype=np.float32),
         reference_frames=np.arange(0,len(frames),max(1,len(frames)//16),dtype=np.uint32))
    lf=[];cf=[];pos=[]
    for i in range(0,len(frames),2):
        for name in drivers:
            j=lookup[name]; q=Quaternion((float(local[i,j,6]),*map(float,local[i,j,3:6])))
            lf.extend((*tuple(q @ Vector((0,0,1))),*tuple(q @ Vector((0,1,0)))))
            m=Matrix(frames[i,j].tolist()); q=m.to_quaternion()
            cf.extend((*tuple(q @ Vector((0,0,1))),*tuple(q @ Vector((0,1,0)))))
            pos.extend(m.translation)
    payload=np.array(lf+cf+pos,dtype='<f4').tobytes(); n=len(range(0,len(frames),2))
    header=struct.pack('<8s10I32s32s32s',b'MLDRV001',1,144,n,30,1,45,0,len(lf),len(cf),len(pos),
                       bytes.fromhex(model_hash),hashlib.sha256('\n'.join(drivers).encode()).digest(),hashlib.sha256(payload).digest())
    (OUTPUT/(key+'.mldrv')).write_bytes(header+payload)
    return {'id':key,'name':{'idle':'Idle','walk':'Walk','run':'Run','sprint':'Sprint','jump':'Jump','turn':'Turn','complex':'Complex dance'}[key],
            'animation':key+'.dma','drivers':key+'.mldrv','duration':(len(frames)-1)/60,'warmup':warmup_frames/60}


def save_editable_actions(clips, names, rest):
    """Put all six derived loops on the original complete Blender rig.

    Per-bone rest-axis corrections are inverted independently of the runtime sampler.
    Constant channels have one key; moving channels retain the full 60 Hz samples.
    """
    bpy.ops.wm.open_mainfile(filepath=str(OUTPUT/'C10032_Demo.blend'))
    rig=bpy.data.objects['Root']; f,correction,_=calibration(rig,rest)
    if rig.animation_data and rig.animation_data.action:
        rig.animation_data.action.name='Demo_Complex_39s'
        rig.animation_data.action.use_fake_user=True
    lookup={name:i for i,name in enumerate(names)}
    left=np.array(rig.matrix_world.inverted() @ f.inverted())
    inverse_correction={n:np.array(m.inverted()) for n,m in correction.items()}
    body=bpy.data.objects['SK_C10032_Body_LOD1']; arrays,_=mesh_data(body,f,names)
    from mathutils.kdtree import KDTree
    tree=KDTree(len(body.data.vertices))
    for vertex in body.data.vertices:tree.insert(WORLD @ f @ body.matrix_world @ vertex.co,vertex.index)
    tree.balance();source_ids=[tree.find(Vector(p))[1] for p in arrays['positions']]
    inverse_bind=[(WORLD @ rest[n]).inverted() for n in names]
    validation={}
    for key,frames in clips.items():
        action=bpy.data.actions.new('Demo_'+key.title());action.use_fake_user=True
        action['demo_fps']=60;action['demo_duration']=(len(frames)-1)/60;action['demo_source']='versioned full-skeleton export'
        for name,bone in rig.data.bones.items():
            if name not in lookup:continue
            pose=left @ frames[:,lookup[name]] @ inverse_correction[name]
            if bone.parent:
                parent=left @ frames[:,lookup[bone.parent.name]] @ inverse_correction[bone.parent.name]
                basis=np.array(bone.matrix_local.inverted() @ bone.parent.matrix_local) @ np.linalg.inv(parent) @ pose
            else:basis=np.array(bone.matrix_local.inverted()) @ pose
            values=[]
            for matrix in basis:
                location,rotation,scale=Matrix(matrix.tolist()).decompose()
                if max(abs(v-1) for v in scale)>.002:raise ValueError('Editable action has unexpected bone scale')
                values.append((*location,*rotation))
            values=np.array(values,dtype=np.float32)
            for frame in range(1,len(values)):
                if np.dot(values[frame-1,3:],values[frame,3:])<0:values[frame,3:]*=-1
            rig.pose.bones[name].rotation_mode='QUATERNION'
            path='pose.bones["'+bpy.utils.escape_identifier(name)+'"]'
            for channel in range(7):
                curve=action.fcurves.new(path+('.location' if channel<3 else '.rotation_quaternion'),index=channel if channel<3 else channel-3,action_group=name)
                samples=values[:,channel]
                if np.ptp(samples)<1e-7:samples=samples[:1]
                points=np.column_stack((np.arange(len(samples),dtype=np.float32)+1,samples))
                curve.keyframe_points.add(len(samples));curve.keyframe_points.foreach_set('co',points.ravel())
                for point in curve.keyframe_points:point.interpolation='LINEAR'
        print('DEMO_EDITABLE_ACTION',key,len(action.fcurves),flush=True)
        rig.animation_data.action=action
        samples=sorted(set([0,len(frames)//2,len(frames)-1]));reference_skin=[];reference_bones=[]
        max_position=max_angle=max_skin=0.
        for frame in samples:
            bpy.context.scene.frame_set(frame+1)
            actual=pose_ue(rig,f,correction,names,rest);expected=[Matrix(m.tolist()) for m in frames[frame]]
            for name,a,b in zip(names,actual,expected):
                if name not in correction:continue # explicit synthetic scene node, absent from original rig
                max_position=max(max_position,(a.translation-b.translation).length*.01)
                angle=a.to_quaternion().rotation_difference(b.to_quaternion()).angle
                max_angle=max(max_angle,math.degrees(min(angle,2*math.pi-angle)))
            evaluated=body.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
            direct=np.array([tuple(WORLD @ f @ body.matrix_world @ mesh.vertices[i].co) for i in source_ids])
            posed=skin(arrays['positions'],arrays['bone_ids'],arrays['weights'],[WORLD @ p @ inv for p,inv in zip(expected,inverse_bind)])
            max_skin=max(max_skin,float(np.linalg.norm(direct-posed,axis=1).max()));evaluated.to_mesh_clear()
            reference_skin.append(direct);reference_bones.append(np.array(actual))
        validation[key]=dict(bone_position_m=max_position,bone_angle_degrees=max_angle,body_skin_m=max_skin)
        if max_position>.001 or max_angle>.1 or max_skin>.001:raise ValueError(f'Editable action validation failed: {key}: {validation[key]}')
        pack(OUTPUT/(key+'.reference.dmp'),dict(source='Blender evaluated full skeleton and body',**validation[key]),
             frames=np.array(samples,dtype=np.uint32),bones=np.array(reference_bones),skin=np.array(reference_skin),
             validated_bone_indices=np.array([i for i,n in enumerate(names) if n in correction],dtype=np.uint32))
        print('DEMO_EDITABLE_VALIDATED',key,validation[key],flush=True)
    bpy.context.scene.render.fps=60
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT/'C10032_Demo.blend'))
    (OUTPUT/'blender-animation-validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(); parser.add_argument('--clips',default='all')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUTPUT=args.output.resolve()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    records,rest,residual=reference(); names=[r['name'] for r in records]
    lookup={n:i for i,n in enumerate(names)};parents=[lookup.get(r['parent'],-1) for r in records]
    model=MODEL.read_bytes(); config=json.loads(model[4:4+struct.unpack_from('<I',model)[0]])
    drivers=config['driverNames']; digest=hashlib.sha256(model).hexdigest()
    rig=bpy.data.objects['Root']; body=bpy.data.objects['SK_C10032_Body_LOD1']
    f,c,fit=calibration(rig,rest)
    skin_error=export_character(rig,body,rest,names,f,c)
    complex_frames,_=sample_source(rig,names,rest)
    entries=[export_clip('complex',complex_frames,names,records,drivers,digest)]
    # Save editable reference scene as a separate generated artifact.
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT/'C10032_Demo.blend'),copy=True)
    if args.clips=='all':
        source_frames={}
        editable={'complex':complex_frames}
        for key in ['idle','walk','run','sprint','jump_start','jump_land','turn']:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.fbx(filepath=str(SOURCE/(key+'.fbx')),use_anim=True,automatic_bone_orientation=False)
            rigs=[o for o in bpy.context.scene.objects if o.type=='ARMATURE' and 'Root_M' in o.data.bones]
            if len(rigs)!=1: raise ValueError(f'{key}: expected one source armature')
            duration=json.loads((SOURCE/'sources.json').read_text(encoding='utf8'))[key]['duration']
            source_frames[key],_=sample_source(rigs[0],names,rest,duration=duration)
        root=names.index('Root_M')
        for key in ['idle','walk','run','sprint']:
            frames=close_loop(source_frames[key],root,parents)
            editable[key]=frames
            entries.append(export_clip(key,frames,names,records,drivers,digest))
        # Complete jump by matching landing root to takeoff, blending poses over 0.2s.
        a=source_frames['jump_start']; b=source_frames['jump_land'].copy()
        shift=a[-1,root,:3,3]-b[0,root,:3,3]; b[:,:,:3,3]+=shift
        bridge=np.array([mix_pose(a[-1],b[0],i/12,parents) for i in range(1,13)])
        frames=np.concatenate((a,bridge,b))
        # Bring height and stance back to the starting pose over a 0.5s recovery.
        target=frames[0].copy(); target[:,:2,3]+=frames[-1,root,:2,3]-frames[0,root,:2,3]
        recovery=np.array([mix_pose(frames[-1],target,i/30,parents) for i in range(1,31)])
        frames=np.concatenate((frames,recovery))
        editable['jump']=frames
        entries.append(export_clip('jump',frames,names,records,drivers,digest))
        turn=source_frames['turn']; turn=np.concatenate((turn,turn[-2::-1]))
        editable['turn']=turn
        entries.append(export_clip('turn',turn,names,records,drivers,digest))
        save_editable_actions(editable,names,rest)
    order=['idle','walk','run','sprint','jump','turn','complex']
    entries.sort(key=lambda x:order.index(x['id']))
    # Real source diffuse textures; no invented texture channels.
    textures=[]
    for label,filename in [('body','T_C10032_Body_CLR.tga'),('head','T_C10032_Head_CLR.tga')]:
        src=Path('F:/ArtWorks/Cloth/CH_Show_01')/filename
        shutil.copy2(src,OUTPUT/(label+'.tga'))
        textures.append(label+'.tga')
    manifest=dict(version=1,character='C10032',model=os.path.relpath(MODEL,OUTPUT),model_sha256=digest,
         body='character.dmp',cloth='cloth.dmp',collision='collision.dmp',textures=textures,clips=entries,
         validation=dict(reference_bind_error_cm=residual,source_fit_error_cm=fit,blender_skin_error_m=skin_error),
         default_clip='walk',default_physics_hz=240,
         animation_pipeline=dict(revision=2,pose_blend='parent_local_fk',loop_closure='moving_local_endpoint_offset'))
    (OUTPUT/'demo.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('DEMO_BAKE_COMPLETE',json.dumps(manifest['validation']),flush=True)


if __name__=='__main__': main()
