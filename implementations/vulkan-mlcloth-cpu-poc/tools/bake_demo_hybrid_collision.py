"""Bake shoulder/arm/hip STM plus an abdominal bridge and fitted multi-section leg capsules."""
from pathlib import Path
import json,struct,sys,hashlib
import bpy,bmesh,numpy as np
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bake_demo_blender import reference,calibration,mesh_data,pack
OUT=Path(__file__).resolve().parents[1]/'.work/demo'
def read(name):
    raw=(OUT/name).read_bytes();n=struct.unpack_from('<Q',raw,8)[0];meta=json.loads(raw[16:16+n]);payload=raw[16+n:]
    return meta,{k:np.frombuffer(payload,dtype=e['dtype'],offset=e['offset'],count=e['bytes']//4).reshape(e['shape']).copy() for k,e in meta['arrays'].items()}
bodymeta,body=read('character.dmp');_,full=read('collision-full.dmp');names=bodymeta['bones'];_,rest,_=reference()
bpy.ops.wm.read_factory_settings(use_empty=True)
source='F:/ArtWorks/Cloth/CH/C10032_ClothCollision_STM_ShoulderHips_800.fbx'
bpy.ops.import_scene.fbx(filepath=source,use_anim=False)
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');obj=next(o for o in bpy.context.scene.objects if o.type=='MESH')
f,corrections,error=calibration(rig,rest);stm,_=mesh_data(obj,f,names)
# Build in the exact render-world rest frame. The bridge removes the uncovered abdomen.
def object_from_arrays(name,arrays):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(arrays['positions'],[],arrays['triangles']);mesh.update();o=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(o);return o
base=object_from_arrays('STM shoulder arms hips',stm);bridge=object_from_arrays('Abdominal bridge',full)
bm=bmesh.new();bm.from_mesh(bridge.data)
for axis,value,sign in [(1,.97,1),(1,1.30,-1),(0,-.205,1),(0,.205,-1)]:
    origin=[0,0,0];normal=[0,0,0];origin[axis]=value;normal[axis]=sign
    bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),dist=1e-6,plane_co=origin,plane_no=normal,clear_inner=True)
    boundary=[e for e in bm.edges if e.is_boundary]
    if boundary:bmesh.ops.holes_fill(bm,edges=boundary,sides=0)
bmesh.ops.triangulate(bm,faces=list(bm.faces));bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(bridge.data);bm.free()
bpy.ops.object.select_all(action='DESELECT');base.select_set(True);bpy.context.view_layer.objects.active=base
union=base.modifiers.new('Join abdominal coverage','BOOLEAN');union.operation='UNION';union.solver='EXACT';union.object=bridge
bpy.ops.object.modifier_apply(modifier=union.name)
bm=bmesh.new();bm.from_mesh(base.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6);bmesh.ops.triangulate(bm,faces=list(bm.faces));bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(base.data);bm.free()
points=np.array([v.co[:] for v in base.data.vertices]);tri=np.array([p.vertices[:] for p in base.data.polygons],dtype=np.uint32)
# Transfer the authored STM weights on its surface; the abdominal bridge uses the body.
source_tree=BVHTree.FromPolygons(stm['positions'].tolist(),stm['triangles'].tolist(),all_triangles=True)
body_tree=BVHTree.FromPolygons(body['positions'].tolist(),body['triangles'].tolist(),all_triangles=True)
ids=[];weights=[]
for point in points:
    q,n,face,d=source_tree.find_nearest(Vector(point));data=stm
    if d>.002:q,n,face,d=body_tree.find_nearest(Vector(point));data=body
    vertices=data['triangles'][face];a,b,c=data['positions'][vertices]
    uv=np.linalg.lstsq(np.column_stack((b-a,c-a)),np.array(q)-a,rcond=None)[0];bc=np.maximum([1-uv.sum(),*uv],0);bc/=bc.sum();influences={}
    for v,w in zip(vertices,bc):
        for bone,weight in zip(data['bone_ids'][v],data['weights'][v]):influences[int(bone)]=influences.get(int(bone),0)+float(w*weight)
    selected=sorted(influences.items(),key=lambda x:-x[1])[:32];total=sum(w for _,w in selected);selected += [(0,0)]*(32-len(selected));ids.append([i for i,w in selected]);weights.append([w/total for i,w in selected])
normals=np.zeros_like(points);fn=np.cross(points[tri[:,1]]-points[tri[:,0]],points[tri[:,2]]-points[tri[:,0]])
for face,n in zip(tri,fn):
    for v in face:normals[v]+=n
normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1);_,counts=np.unique(edges,axis=0,return_counts=True)
if not (counts==2).all():raise RuntimeError('Hybrid STM is not closed')
pack(OUT/'collision-hybrid.dmp',dict(influences=32,closed=True,source=source,abdomen='closed Boolean union with full body bridge'),positions=points,normals=normals,uv=np.zeros((len(points),2)),triangles=tri,
     material=np.zeros(len(tri),dtype=np.uint32),bone_ids=np.array(ids,dtype=np.uint32),weights=np.array(weights))
# Fit radii to body samples around each bone section, instead of using the wider ragdoll asset.
inv=np.concatenate([body['inverse_bind'],np.tile([[[0,0,0,1]]],(len(names),1,1))],axis=1);bind=np.linalg.inv(inv)
cap_bones=[];cap_a=[];cap_b=[];radii=[];caps=[];caps_world=[]
for side in ['L','R']:
    for bone_name,end_name,sections in [('Hip_'+side,'Knee_'+side,[(.42,.72),(.67,1.0)]),('Knee_'+side,'Ankle_'+side,[(0,.4),(.35,.7),(.65,1)])]:
        bone=names.index(bone_name);end=names.index(end_name);a=bind[bone,:3,3];b=bind[end,:3,3];axis=b-a;length=np.linalg.norm(axis);axis/=length
        vertices=body['positions'];along=(vertices-a)@axis/length;radial=vertices-a-(along*length)[:,None]*axis
        same_side=(vertices[:,0]*a[0]>0)&(np.linalg.norm(radial,axis=1)<.13)
        for first,last in sections:
            selected=same_side&(along>=first)&(along<=last)
            if selected.sum()<4:raise RuntimeError('Missing body samples for leg section')
            offset=np.median(radial[selected],axis=0);offset-=axis*(offset@axis)
            distances=np.linalg.norm(radial[selected]-offset,axis=1);radius=float(np.quantile(distances,.98)+.001)
            start=a+axis*(first*length)+offset;finish=a+axis*(last*length)+offset
            cap_bones.append(bone);cap_a.append((inv[bone]@np.r_[start,1])[:3]);cap_b.append((inv[bone]@np.r_[finish,1])[:3]);radii.append(radius)
            caps_world.append((bone,start,finish,radius))
            caps.append(dict(bone=bone_name,interval=[first,last],radius_m=radius,samples=int(selected.sum())))
pack(OUT/'leg-capsules.dmp',dict(version_note='full-skeleton bone-local cm; world radius metres; fitted multi-section legs'),bone=np.array(cap_bones,dtype=np.uint32),a=np.array(cap_a),b=np.array(cap_b),radius=np.array(radii))
report=dict(stm_vertices=len(points),stm_triangles=len(tri),closed=True,source=source,source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),capsules=caps,status='development hybrid collision; quality and realtime performance qualification pending')
(OUT/'collision-hybrid.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
manifest=json.loads((OUT/'demo-attached.json').read_text(encoding='utf-8'));manifest['collision']='collision-hybrid.dmp';manifest['capsules']='leg-capsules.dmp'
manifest['collision_strategy']=dict(type='stm_and_capsules',stm_regions=['shoulders','arms','hips','abdomen'],capsule_regions=['segmented thighs','segmented calves'],source_report='collision-hybrid.json')
(OUT/'demo-hybrid-collision.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
bpy.data.objects.remove(bridge,do_unlink=True);obj.hide_set(True);obj.hide_render=True;base.name='C10032_STM_Hybrid'
# Keep the editable Blender collision objects bound to the source's complete rig.
# Pack points use render-world metres; the imported FBX rig uses Blender world.
render_from_ue=Matrix(((.01,0,0,0),(0,0,.01,0),(0,-.01,0,0),(0,0,0,1)))
blender_from_render=f.inverted() @ render_from_ue.inverted()
def bind_object(o,vertex_ids,vertex_weights):
    o.matrix_world=blender_from_render
    groups={}
    for v,(bones,values) in enumerate(zip(vertex_ids,vertex_weights)):
        for bone,value in zip(bones,values):
            if value<=1e-8:continue
            name=names[int(bone)]
            if name not in rig.data.bones:raise RuntimeError('Collision influence missing from source rig: '+name)
            if name not in groups:groups[name]=o.vertex_groups.new(name=name)
            groups[name].add([v],float(value),'REPLACE')
    modifier=o.modifiers.new('Full skeleton skinning','ARMATURE');modifier.object=rig
bind_object(base,ids,weights)
for index,(bone,a,b,radius) in enumerate(caps_world):
    axis=(b-a)/np.linalg.norm(b-a);u=np.cross(axis,[1,0,0]);u/=np.linalg.norm(u);v=np.cross(axis,u)
    vertices=[];faces=[];segments=16
    # Separate equator rings preserve the cylindrical middle section.
    for center,angles in [(a,np.linspace(-np.pi/2,0,7)),(b,np.linspace(0,np.pi/2,7))]:
        for latitude in angles:
            for longitude in np.arange(segments)*2*np.pi/segments:
                vertices.append(center+radius*(np.sin(latitude)*axis+np.cos(latitude)*(np.cos(longitude)*u+np.sin(longitude)*v)))
    for ring in range(13):
        for j in range(segments):
            k=ring*segments+j;n=ring*segments+(j+1)%segments
            faces.extend([[k,n,n+segments],[k,n+segments,k+segments]])
    visual=object_from_arrays('Leg capsule %02d %s'%(index,names[bone]),dict(positions=vertices,triangles=faces))
    visual.display_type='WIRE';visual.hide_render=True
    bind_object(visual,[[bone]]*len(vertices),[[1.]]*len(vertices))
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'C10032_HybridCollision.blend'))
print(json.dumps(report,indent=2),flush=True)

