"""Fit the existing closed STM topology to the actual render body; preserve originals."""
import json
import struct
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bake_demo_blender import pack

directory=Path(__file__).resolve().parents[1]/'.work/demo'
parser=argparse.ArgumentParser();parser.add_argument('--source',default='collision.dmp');parser.add_argument('--output',default='collision-fit.dmp');parser.add_argument('--offset',type=float,default=.001)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
def read(name):
    raw=(directory/name).read_bytes();size=struct.unpack_from('<Q',raw,8)[0];header=json.loads(raw[16:16+size]);payload=raw[16+size:]
    return header,{k:np.frombuffer(payload,dtype=e['dtype'],count=e['bytes']//4,offset=e['offset']).reshape(e['shape']).copy() for k,e in header['arrays'].items()}
_,body=read('character.dmp');meta,proxy=read(args.source)
tree=BVHTree.FromPolygons(body['positions'].tolist(),body['triangles'].tolist(),all_triangles=True)
positions=[];ids=[];weights=[];errors=[]
for vertex in proxy['positions']:
    q,normal,face,distance=tree.find_nearest(Vector(vertex));errors.append(distance)
    triangle=body['triangles'][face];a,b,c=body['positions'][triangle]
    uv=np.linalg.lstsq(np.column_stack((b-a,c-a)),np.array(q)-a,rcond=None)[0]
    bary=np.maximum([1-uv.sum(),uv[0],uv[1]],0);bary/=bary.sum()
    influences={}
    for v,w in zip(triangle,bary):
        for bone,weight in zip(body['bone_ids'][v],body['weights'][v]):influences[int(bone)]=influences.get(int(bone),0.)+float(w*weight)
    selected=sorted(influences.items(),key=lambda entry:-entry[1])[:32];total=sum(w for _,w in selected)
    selected += [(0,0)]*(32-len(selected));ids.append([b for b,w in selected]);weights.append([w/total for b,w in selected])
    positions.append(tuple(q+normal*args.offset))
old=proxy['positions'];new=np.array(positions,dtype=np.float32);tri=proxy['triangles']
old_normal=np.cross(old[tri[:,1]]-old[tri[:,0]],old[tri[:,2]]-old[tri[:,0]])
projected=new.copy();fraction=np.ones(len(new),dtype=np.float32)
for iteration in range(20):
    normals=np.cross(new[tri[:,1]]-new[tri[:,0]],new[tri[:,2]]-new[tri[:,0]])
    invalid=np.einsum('ij,ij->i',old_normal,normals)<=0
    if not invalid.any():break
    affected=np.unique(tri[invalid]);fraction[affected]*=.5
    new=old+(projected-old)*fraction[:,None]
new_normal=np.cross(new[tri[:,1]]-new[tri[:,0]],new[tri[:,2]]-new[tri[:,0]])
flips=int((np.einsum('ij,ij->i',old_normal,new_normal)<0).sum())
proxy['positions']=new;proxy['bone_ids']=np.array(ids,dtype=np.uint32);proxy['weights']=np.array(weights,dtype=np.float32)
proxy['normals']=np.zeros_like(new)
for face,n in zip(tri,new_normal):
    for vertex in face:proxy['normals'][vertex]+=n
proxy['normals']/=np.maximum(np.linalg.norm(proxy['normals'],axis=1)[:,None],1e-10)
report={'source':args.source,'body':'character.dmp','vertices':len(new),'triangles':len(tri),'surface_offset_m':args.offset,
        'fit_distance_p95_m':float(np.quantile(errors,.95)),'fit_distance_max_m':max(errors),'flipped_faces':flips,
        'limited_vertices':int((fraction<1).sum()),
        'status':'candidate; requires animated and visible-contact validation'}
pack(directory/args.output,dict(influences=32,closed=True,source=report['source'],fit=report),**proxy)
(directory/Path(args.output).with_suffix('.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
