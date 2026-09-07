"""Read-only animation contact diagnostics from exported skinning, in world metres.

Run with a Python environment containing numpy. This diagnoses foot motion;
the separate Blender reference tests validate coordinate and skinning accuracy.
"""
import argparse
import json
import struct
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'.work/demo/demo.json')
parser.add_argument('--output',type=Path)
args=parser.parse_args()
directory=args.manifest.parent
manifest=json.loads(args.manifest.read_text(encoding='utf8'))

def pack(path):
    raw=path.read_bytes()
    if raw[:8]!=b'DMPACK01':raise ValueError('Unexpected asset container')
    size=struct.unpack_from('<Q',raw,8)[0];h=json.loads(raw[16:16+size]);payload=raw[16+size:]
    return h,{key:np.frombuffer(payload,dtype=e['dtype'],offset=e['offset'],count=int(np.prod(e['shape']))).reshape(e['shape']).copy() for key,e in h['arrays'].items()}

body_meta,body=pack(directory/manifest['body'])
inverse=np.tile(np.eye(4),(len(body_meta['bones']),1,1));inverse[:,:3]=body['inverse_bind']
positions=np.column_stack((body['positions'],np.ones(len(body['positions']))))
convert=np.array([[.01,0,0,0],[0,0,.01,0],[0,-.01,0,0],[0,0,0,1]])
report=dict(qualification=False,method='60 Hz exported skinning; sole vertices within 15 mm of ground and vertical speed below 0.1 m/s; horizontal speed is a sliding diagnostic, not a foot-lock certificate',clips=[])
for clip in manifest['clips']:
    meta,data=pack(directory/clip['animation']);local=data['local'].astype(float);parents=meta['parents'];cache={}
    def world(index):
        if index not in cache:
            values=local[:,index];q=values[:,3:7];q=q/np.linalg.norm(q,axis=1)[:,None];x,y,z,w=q.T
            m=np.tile(np.eye(4),(len(values),1,1));m[:,:3,3]=values[:,:3]
            m[:,0,:3]=np.column_stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)))
            m[:,1,:3]=np.column_stack((2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)))
            m[:,2,:3]=np.column_stack((2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
            cache[index]=world(parents[index])@m if parents[index]>=0 else convert@m
        return cache[index]
    result=dict(clip=clip['id'],duration=clip['duration'],feet=[])
    for side in ('L','R'):
        ankle=meta['bones'].index('Ankle_'+side);foot_bones={ankle}
        for i,parent in enumerate(parents):
            if parent in foot_bones:foot_bones.add(i)
        mask=np.sum(body['weights']*np.isin(body['bone_ids'],list(foot_bones)),axis=1)>.5
        floor=float(body['reference_skin'][mask,1].min());mask&=body['reference_skin'][:,1]<floor+.01
        selected=np.where(mask)[0];skin=np.zeros((len(local),len(selected),3))
        for slot in range(body['weights'].shape[1]):
            ids=body['bone_ids'][selected,slot];weights=body['weights'][selected,slot]
            for bone in np.unique(ids[weights>0]):
                which=np.where((ids==bone)&(weights>0))[0];matrix=world(int(bone))@inverse[bone]
                transformed=np.einsum('fij,vj->fvi',matrix,positions[selected[which]])[:,:,:3]
                skin[:,which]+=transformed*weights[which][None,:,None]
        velocity=np.diff(skin,axis=0)*60;height=(skin[:-1,:,1]+skin[1:,:,1])*.5
        contact=(height>=-.015)&(height<=.015)&(np.abs(velocity[:,:,1])<.1)
        speed=np.linalg.norm(velocity[:,:,[0,2]],axis=2);frame_speed=[]
        for i in range(len(speed)):
            if np.count_nonzero(contact[i])>=3:frame_speed.append(float(np.median(speed[i,contact[i]])))
        result['feet'].append(dict(side=side,sole_vertices=len(selected),min_height_m=float(skin[:,:,1].min()),
            max_minimum_height_m=float(skin[:,:,1].min(axis=1).max()),contact_seconds=len(frame_speed)/60,
            contact_horizontal_speed_p50_mps=float(np.median(frame_speed)) if frame_speed else None,
            contact_horizontal_speed_p95_mps=float(np.quantile(frame_speed,.95)) if frame_speed else None))
    report['clips'].append(result)
    print(json.dumps(result),flush=True)
(args.output or directory/'foot-contact-audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
