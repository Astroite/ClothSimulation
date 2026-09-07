"""Blender-side geometric diagnostics for an explicit runtime physics snapshot.

blender --factory-startup -b --python-exit-code 1 -P tools/measure_demo_snapshot.py
Nearest outward surface signs are a diagnostic, not a substitute for rollout/visual QA.
"""
import json
import struct
import sys
import argparse
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from demo_contact_metrics import solid_metrics

directory=Path(__file__).resolve().parents[1]/'.work/demo'
parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,default=directory/'physics-snapshot.json');parser.add_argument('--output',type=Path,default=directory/'snapshot-quality.json')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
snapshot=json.loads(args.snapshot.read_text(encoding='utf8'))
raw=(directory/'cloth.dmp').read_bytes();size=struct.unpack_from('<Q',raw,8)[0]
header=json.loads(raw[16:16+size]);payload=raw[16+size:]
def array(key):
    entry=header['arrays'][key]
    return np.frombuffer(payload,dtype=entry['dtype'],count=entry['bytes']//4,offset=entry['offset']).reshape(entry['shape'])
rest=np.array(snapshot['rest']) if 'rest' in snapshot else array('positions')
pinned=np.array(snapshot['pinned']).ravel() if 'pinned' in snapshot else array('pinned').ravel()
tri=np.array(snapshot['triangles']).reshape(-1,3)
edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0)
rest_length=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
parent=np.arange(len(rest))
def find(v):
    while parent[v]!=v:
        parent[v]=parent[parent[v]];v=parent[v]
    return v
for a,b in edges:
    ra,rb=find(a),find(b)
    if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
regions=np.array([find(i) for i in range(len(rest))])
report={'qualification':False,'method':'legacy nearest-face plane depth plus independent three-ray solid classification and nearest-surface distance; open/intersecting bodies require review',
        'physics':snapshot.get('physics',{}),'actors':[]}
for actor in snapshot['actors']:
    cloth=np.array(actor['cloth']);ratios=np.linalg.norm(cloth[edges[:,0]]-cloth[edges[:,1]],axis=1)/rest_length
    entry={'algorithm':actor['algorithm'],'time':actor['time'],'finite':bool(np.isfinite(cloth).all()),
           'edge_ratio_min':float(ratios.min()),'edge_ratio_p05':float(np.quantile(ratios,.05)),
           'edge_ratio_p95':float(np.quantile(ratios,.95)),'edge_ratio_max':float(ratios.max())}
    entry['components']=[]
    for region in np.unique(regions):
        mask=regions==region;edge_mask=mask[edges[:,0]];lengths=rest_length[edge_mask]
        entry['components'].append(dict(first_vertex=int(region),vertices=int(mask.sum()),pins=int(pinned[mask].sum()),
            edge_ratio_p95=float(np.quantile(ratios[edge_mask],.95)),
            mean_absolute_edge_strain=float(np.sum(np.abs(ratios[edge_mask]-1)*lengths)/lengths.sum())))
    for surface in ['body','proxy']:
        faces=np.array(snapshot[surface+'_triangles']).reshape(-1,3)
        if not actor.get(surface) or not len(faces):
            entry[surface]={'available':False};continue
        tree=BVHTree.FromPolygons(actor[surface],faces.tolist(),all_triangles=True)
        depths=[];distances=[]
        for vertex in cloth:
            point=Vector(vertex);q,n,face,distance=tree.find_nearest(point)
            depths.append(max(0.,-float((point-q).dot(n))))
            distances.append(distance)
        depths=np.array(depths);positive=depths[depths>1e-6]
        entry[surface]={'penetrating_vertices':int(len(positive)), 'penetrating_pins':int(((depths>1e-6)&(pinned>0)).sum()),
                        'penetration_p95_m':float(np.quantile(positive,.95)) if len(positive) else 0.,
                        'penetration_max_m':float(depths.max()),
                        'worst':[dict(vertex=int(i),depth_m=float(depths[i]),pinned=bool(pinned[i]),position=cloth[i].tolist(),rest=rest[i].tolist()) for i in np.argsort(depths)[-8:][::-1] if depths[i]>1e-6]}
        entry[surface]['solid']=solid_metrics(tree,cloth,actor[surface],distances,pinned)
    capsule_depth=np.zeros(len(cloth))
    for capsule in actor.get('capsules',[]):
        a=np.array(capsule['a']);b=np.array(capsule['b']);axis=b-a
        along=np.clip((cloth-a)@axis/max(float(axis@axis),1e-16),0,1)
        capsule_depth=np.maximum(capsule_depth,capsule['radius']-np.linalg.norm(cloth-(a+along[:,None]*axis),axis=1))
    positive=capsule_depth[capsule_depth>1e-6]
    entry['capsules']={'count':len(actor.get('capsules',[])), 'penetrating_vertices':len(positive),
                       'penetration_p95_m':float(np.quantile(positive,.95)) if len(positive) else 0.,
                       'penetration_max_m':float(capsule_depth.max())}
    report['actors'].append(entry)
args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2),flush=True)
