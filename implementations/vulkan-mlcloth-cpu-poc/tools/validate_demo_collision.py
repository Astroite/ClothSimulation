"""Independent animation-reference checks for the full-body proxy and physical attachments."""
from pathlib import Path
import json,struct,sys,argparse
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
OUT=Path(__file__).resolve().parents[1]/'.work/demo'
parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=OUT/'demo-hybrid-collision.json');parser.add_argument('--output',type=Path,default=OUT/'hybrid-collision-animation-validation.json')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
OUT=args.manifest.resolve().parent
def read(name):
    raw=(OUT/name).read_bytes();n=struct.unpack_from('<Q',raw,8)[0];meta=json.loads(raw[16:16+n]);payload=raw[16+n:]
    return meta,{k:np.frombuffer(payload,dtype=e['dtype'],offset=e['offset'],count=e['bytes']//4).reshape(e['shape']).copy() for k,e in meta['arrays'].items()}
manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
_,body=read(manifest['body']);_,proxy=read(manifest['collision']);_,cloth=read(manifest['cloth']);_,capsules=read(manifest['capsules'])
world=np.array([[.01,0,0,0],[0,0,.01,0],[0,-.01,0,0],[0,0,0,1]],dtype=np.float64)
bind=np.concatenate([body['inverse_bind'],np.tile([[[0,0,0,1]]],(len(body['inverse_bind']),1,1))],axis=1)
ids=proxy['bone_ids'];weights=proxy['weights'];points=np.column_stack([proxy['positions'],np.ones(len(proxy['positions']))])
tri=proxy['triangles'];below_neck=(body['positions'][:,1]<1.45)&(body['positions'][:,1]>.12)
report={'qualification':False,'reference':'independent Blender evaluated full skeleton and body, three samples per animation','clips':[]}
for clip in ['idle','walk','run','sprint','jump','turn','complex']:
    _,ref=read(clip+'.reference.dmp');measurements=[]
    for index,frame in enumerate(ref['frames'].ravel()):
        matrices=world@ref['bones'][index]@bind
        skin=np.einsum('nkij,nj,nk->ni',matrices[ids],points,weights)[:,:3]
        if not np.isfinite(skin).all():raise RuntimeError('Non-finite proxy skin')
        tree=BVHTree.FromPolygons(skin.tolist(),tri.tolist(),all_triangles=True)
        reference=ref['skin'][index];samples=reference[below_neck];distances=np.array([tree.find_nearest(Vector(v))[3] for v in samples])
        rest_samples=body['positions'][below_neck];regional={}
        masks={'arms':(np.abs(rest_samples[:,0])>.2)&(rest_samples[:,1]>.6),
               'hips':(np.abs(rest_samples[:,0])<=.2)&(rest_samples[:,1]>.6)&(rest_samples[:,1]<1.03),
               'torso':(np.abs(rest_samples[:,0])<=.2)&(rest_samples[:,1]>=1.03)}
        for region,mask in masks.items():
            selected=distances[mask]
            regional[region]=dict(vertices=int(mask.sum()),distance_p95_m=float(np.quantile(selected,.95)),distance_max_m=float(selected.max()))
        for bone,a,b,radius in zip(capsules['bone'].ravel(),capsules['a'],capsules['b'],capsules['radius'].ravel()):
            transform=world@ref['bones'][index,bone];a=(transform@np.r_[a,1])[:3];b=(transform@np.r_[b,1])[:3];axis=b-a
            along=np.clip((samples-a)@axis/max(float(axis@axis),1e-16),0,1)
            signed=np.linalg.norm(samples-(a+along[:,None]*axis),axis=1)-radius
            distances=np.minimum(distances,np.abs(signed))
        selected=cloth['pinned']>0;faces=reference[cloth['attachment_vertices'][selected]];normal=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]);areas=np.linalg.norm(normal,axis=1)
        if areas.min()<1e-10:raise RuntimeError('Degenerate animated pin attachment')
        measurements.append(dict(frame=int(frame),body_below_neck_surface_p95_m=float(np.quantile(distances,.95)),body_below_neck_surface_max_m=float(max(distances)),stm_regions=regional,min_attachment_double_area_m2=float(areas.min())))
    report['clips'].append(dict(clip=clip,samples=measurements))
args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
