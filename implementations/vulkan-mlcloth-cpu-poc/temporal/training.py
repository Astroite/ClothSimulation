"""Differentiable four-tick rollout through the Demo reference, with explicit data gates.

Sequence NPZ schema (SI units): rest, triangles, mass, pinned, x/v [F,N,3],
body_positions/body_normals [F,M,3], pin_targets [F-1,8,N,3],
collider_positions [F-1,9,C,3], collider_triangles [T,3],
capsules [F-1,9,P,7] (Axyz,Bxyz,radius). Adjacent intervals share endpoints.
The manifest supplies split, clip, config, path and SHA-256 for every sequence.
"""
from pathlib import Path
import json,hashlib,math
import numpy as np
import torch
from .model import GnnHistoryState,normalize_history
from real_scene.fine15 import Fine15

SPLITS={'idle':'train','walk':'train','run':'train','sprint':'train','jump':'train','turn':'validation','complex':'test'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_dataset(manifest):
    manifest=Path(manifest);meta=json.loads(manifest.read_text())
    gate=manifest.parent/meta['reference_gate'];report=json.loads(gate.read_text())
    if meta.get('version')!=1 or not report.get('passed') or not report.get('training_allowed'):
        raise ValueError('reference quality gate did not authorize quality training')
    if digest(gate)!=meta['reference_gate_sha256']:
        raise ValueError('reference gate fingerprint mismatch')
    if not report.get('reference_forward_passed') or not report.get('contact_gradients_passed'):
        raise ValueError('solver forward/contact gradient gate is not complete')
    rows=meta['sequences'];seen={}
    for row in rows:
        if SPLITS.get(row['clip'])!=row['split']:
            raise ValueError('clip/variant split leakage')
        path=manifest.parent/row['path']
        if digest(path)!=row['sha256']:
            raise ValueError('sequence fingerprint mismatch')
        if row['sha256'] in seen and seen[row['sha256']]!=row['split']:
            raise ValueError('same sequence occurs in different splits')
        seen[row['sha256']]=row['split'];row=dict(row);row['resolved_path']=str(path)
        yield row

def load_sequence(row,device):
    with np.load(row['resolved_path'],allow_pickle=False) as data:
        values={key:torch.as_tensor(data[key].copy(),device=device) for key in data.files}
    for key in ('rest','x','v','mass','body_positions','body_normals','pin_targets','collider_positions','capsules'):
        if key not in values or not torch.isfinite(values[key]).all():raise ValueError('missing/nonfinite sequence '+key)
    f,n,_=values['x'].shape;m=values['body_positions'].shape[1];c=values['collider_positions'].shape[2]
    expected={'rest':(n,3),'v':(f,n,3),'mass':(n,),'pinned':(n,),'body_positions':(f,m,3),'body_normals':(f,m,3),
              'pin_targets':(f-1,8,n,3),'collider_positions':(f-1,9,c,3)}
    if f<8:raise ValueError('sequence too short for history and rollout')
    for key,shape in expected.items():
        if tuple(values[key].shape)!=shape:raise ValueError('sequence shape mismatch '+key)
    cp=values['collider_positions']
    if not torch.allclose(cp[:-1,-1],cp[1:,0],atol=1e-6,rtol=0):raise ValueError('body sequence discontinuity')
    for key in ('triangles','collider_triangles'):values[key]=values[key].long().reshape(-1,3)
    values['pinned']=values['pinned'].bool()
    tri=values['triangles'];edges=torch.cat((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]))
    edges=torch.unique(torch.cat((edges,edges.flip(-1))),dim=0,sorted=True)
    order=torch.argsort(edges[:,1]*n+edges[:,0]);edges=edges[order]
    values['senders'],values['receivers']=edges[:,0],edges[:,1]
    lengths=torch.linalg.vector_norm(values['rest'][edges[:,0]]-values['rest'][edges[:,1]],dim=-1)
    if (lengths<=0).any():raise ValueError('degenerate rest edge')
    values['length_scale']=lengths.median().detach()
    values['min_edge']=torch.full((n,),torch.inf,device=device).scatter_reduce(0,edges[:,1],lengths,reduce='amin')
    return values

def make_graph(builder,s,x,v,t):
    return builder.prepare_graph(position=x,previous=x-v/30,rest_position=s['rest'],triangles=s['triangles'],
        mesh_senders=s['senders'],mesh_receivers=s['receivers'],mass=s['mass'],pin_mask=s['pinned'],
        pin_target=s['pin_targets'][t,-1],obstacle_position=s['body_positions'][t],obstacle_target=s['body_positions'][t+1],
        obstacle_normals=s['body_normals'][t],timestep=1/30)

def push_history(history,builder,s,g,v,t):
    raw=builder.weights if hasattr(builder,'weights') else None
    # Fine15 stores the supplied weights as .weights in the existing reference.
    mean,std=raw.normalizer('node');mean,std=mean.reshape(-1),std.reshape(-1)
    cloth_normal=g.cloth_nodes[:,12:15]*std[12:15]+mean[12:15]
    body_delta=torch.zeros_like(s['body_positions'][t]) if t==0 else s['body_positions'][t]-s['body_positions'][t-1]
    h=history.push(torch.cat((v/30,body_delta)),torch.cat((cloth_normal,s['body_normals'][t])))
    # Fine15 compacts active obstacle nodes; history itself always uses stable,
    # full surface IDs, so contact-neighbor changes cannot transfer history.
    selected=torch.cat((torch.arange(len(v),device=v.device),len(v)+g.active_obstacle))
    return normalize_history(h,mean,std)[selected]

def collider_at(s,t,k):
    caps=s['capsules'][t];rows=[]
    for i in range(caps.shape[1]):
        rows.append(dict(previousA=caps[k,i,:3],previousB=caps[k,i,3:6],currentA=caps[k+1,i,:3],currentB=caps[k+1,i,3:6],radius=float(caps[k+1,i,6])))
    return dict(previous=s['collider_positions'][t,k],current=s['collider_positions'][t,k+1],triangles=s['collider_triangles'],capsules=rows)

def rollout(model,weights,solver,s,start,steps=4,trust=2.,history=None,position=None,velocity=None):
    builder=Fine15(weights);history=history or GnnHistoryState()
    x=s['x'][start] if position is None else position;v=s['v'][start] if velocity is None else velocity
    if history.samples is None:
        for t in range(max(0,start-3),start):
            g=make_graph(builder,s,s['x'][t],s['v'][t],t);push_history(history,builder,s,g,s['v'][t],t)
    predictions=[];loss=x.new_zeros(());length=s['length_scale'];tri=s['triangles'];mass=s['mass']/s['mass'].sum()
    gravity=x.new_tensor([0,solver.config.get('gravity',-9.81),0])
    def area(p):return torch.linalg.vector_norm(torch.linalg.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]],dim=-1),dim=-1)*.5
    rest_area=area(s['rest']).clamp_min(1e-12)
    for t in range(start,start+steps):
        g=make_graph(builder,s,x,v,t);h=push_history(history,builder,s,g,v,t)
        raw=weights.inverse('output',model(g,h));valid=torch.isfinite(raw).all(-1,keepdim=True)
        safe=torch.where(valid,raw,gravity/900)
        confidence=(2-torch.linalg.vector_norm(safe-gravity/900,dim=-1,keepdim=True)/(trust*s['min_edge'][:,None]).clamp_min(1e-6)).clamp(0,1)*valid
        acceleration=gravity+confidence*(safe*900-gravity)
        for k in range(8):x,v=solver.step(x,v,1/240,s['pin_targets'][t,k],collider_at(s,t,k),acceleration)
        tx,tv=s['x'][t+1],s['v'][t+1]
        pos_loss=(((x-tx)/length).square().sum(-1)*mass).sum()
        vel_loss=(((v-tv)/(30*length)).square().sum(-1)*mass).sum()
        send,recv=s['senders'],s['receivers']
        edge_loss=((torch.linalg.vector_norm(x[send]-x[recv],dim=-1)-torch.linalg.vector_norm(tx[send]-tx[recv],dim=-1))/length).square().mean()
        area_loss=(.05-area(x)/rest_area).clamp_min(0).square().mean()
        # Local surface-plane penalty is only a training surrogate; reports use the actual collider.
        body=s['body_positions'][t+1];normals=s['body_normals'][t+1]
        if len(body):
            with torch.no_grad():nearest=torch.cdist(x.detach(),body).argmin(-1)
            delta=x-body[nearest];near=torch.linalg.vector_norm(delta,dim=-1)<.03
            penetration=(solver.config.get('thickness',.003)-(delta*normals[nearest]).sum(-1)).clamp_min(0)
            contact_loss=(penetration[near]/length).square().mean() if near.any() else x.new_zeros(())
        else:contact_loss=x.new_zeros(())
        loss=loss+pos_loss+.1*vel_loss+.1*edge_loss+area_loss+.1*contact_loss+1e-4*(safe/length).square().mean()
        predictions.append((x,v))
    return loss/steps,predictions,history
