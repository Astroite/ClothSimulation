"""Offline triangle correspondence for the original decorated display garment."""
import bpy
import json
import sys
import struct
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from bake_demo_blender import pack,reference,calibration,WORLD

def main():
    records,rest,_=reference()
    f,_,_=calibration(bpy.data.objects['Root'],rest)
    data=(ROOT/'.work/demo/cloth.dmp').read_bytes()
    size=struct.unpack_from('<Q',data,8)[0];meta=json.loads(data[16:16+size]);payload=data[16+size:]
    def array(key):
        a=meta['arrays'][key]
        return np.frombuffer(payload,dtype=a['dtype'],count=a['bytes']//4,offset=a['offset']).reshape(a['shape'])
    positions=array('positions');triangles=array('triangles')
    bvh=BVHTree.FromPolygons(positions.tolist(),triangles.tolist(),all_triangles=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source='F:/ArtWorks/Cloth/CH_10032_V3/SM_C10032_Cloth_01.fbx'
    bpy.ops.import_scene.fbx(filepath=source,use_anim=False)
    obj=bpy.data.objects['SM_C10032_Cloth_01'];mesh=obj.data;mesh.calc_loop_triangles()
    matrix=WORLD@f@obj.matrix_world
    # The local original is ankle length; the locked model is a calf-length garment.
    # Tailor only the lower pattern in the generated Blender file, retaining its UVs.
    inverse=matrix.inverted()
    for vertex in mesh.vertices:
        point=matrix@vertex.co
        if point.y<1.05:
            point.y=.24+(point.y-.01)*(1.05-.24)/(1.05-.01)
        vertex.co=inverse@point
    mapping=[];barycentric=[];uv=[];indices=[];materials=[];distances=[];cache={};reference_positions=[]
    for triangle in mesh.loop_triangles:
        face=[]
        for lid,vid in zip(triangle.loops,triangle.vertices):
            tex=mesh.uv_layers.active.data[lid].uv
            key=(vid,tuple(tex),triangle.material_index)
            if key not in cache:
                point=matrix@mesh.vertices[vid].co
                q,n,ti,distance=bvh.find_nearest(point)
                a,b,c=positions[triangles[ti]]
                e1=b-a;e2=c-a
                bc=np.linalg.lstsq(np.column_stack((e1,e2)),np.array(q)-a,rcond=None)[0]
                weights=np.array([1-bc.sum(),bc[0],bc[1]])
                normal=np.cross(e1,e2);normal/=np.linalg.norm(normal)
                offset=float(np.dot(np.array(point)-np.array(q),normal))
                reconstructed=weights@positions[triangles[ti]]+normal*offset
                # Closest points on triangle boundaries can have tangential residual:
                # store a full local offset in a rest triangle orthonormal frame.
                tangent=e1/np.linalg.norm(e1);bitangent=np.cross(normal,tangent)
                residual=np.array(point)-weights@positions[triangles[ti]]
                local=[np.dot(residual,tangent),np.dot(residual,bitangent),offset]
                cache[key]=len(mapping);mapping.append([*triangles[ti],triangle.material_index])
                barycentric.append([*weights,0]);uv.append([tex.x,tex.y,*local[:2]])
                reference_positions.append([*local,0]);distances.append(distance)
            face.append(cache[key])
        if matrix.to_3x3().determinant()<0:face[1],face[2]=face[2],face[1]
        indices.append(face)
    for m in mesh.materials:
        color=list(m.diffuse_color)
        if m.use_nodes:
            bsdf=next((n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
            if bsdf:color=list(bsdf.inputs['Base Color'].default_value)
        materials.append(dict(name=m.name,base_color=color,source_channel='FBX constant base color'))
    pack(ROOT/'.work/demo/display.dmp',dict(source=source,materials=materials,
         tailoring='Lower pattern shortened from 0.01 m to 0.24 m hem; waist 1.05 m remains fixed',
         mapping='offline closest triangle, barycentric coordinates and rest tangent-frame offset',
         distance_p95_m=float(np.percentile(distances,95)),distance_max_m=float(max(distances))),
         mapping=np.array(mapping,dtype=np.uint32),barycentric=np.array(barycentric),
         uv=np.array(uv),offset=np.array(reference_positions),triangles=np.array(indices,dtype=np.uint32))
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'.work/demo/C10032_Display.blend'))
    print('DISPLAY_BAKE_COMPLETE',len(mapping),len(indices),np.percentile(distances,[50,95,100]),flush=True)
if __name__=='__main__':main()
