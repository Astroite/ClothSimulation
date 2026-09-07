"""Vectorized moving-capsule projections; detached hit selection, live projection."""
import torch

def project_capsules(s,pos,previous,capsules,thickness,dt,pins,contacts,swept=True,filter_mask=None):
    if not capsules:return pos,contacts
    lookup={c['vertex']:i for i,c in enumerate(contacts)}
    def norm(x):return torch.linalg.vector_norm(x,dim=-1)
    def closest(p,a,b):
        axis=b-a;length2=(axis*axis).sum(-1);along=((p-a)*axis).sum(-1)/length2.clamp_min(1e-16)
        along=torch.where(length2>1e-16,along.clamp(0,1),torch.zeros_like(along));return a+axis*along[...,None],along
    def normalized(v):return v/norm(v).clamp_min(1e-30)[...,None]
    for phase in range(2):
        for cap in capsules:
            ca,cb,pa,pb=(torch.as_tensor(cap[k],device=pos.device,dtype=pos.dtype) for k in ('currentA','currentB','previousA','previousB'))
            radius=cap['radius']+thickness;end=pos;start=previous;q,along=closest(end,ca,cb);oldq,_=closest(start,pa,pb)
            distance=norm(end-q);oldgap=norm(start-oldq)-radius;recovery=oldgap<0
            with torch.no_grad():
                motion=end-start;bound=torch.maximum(norm(ca-pa-motion),norm(cb-pb-motion))
                crossing=torch.zeros_like(pins,dtype=torch.bool);time=torch.zeros(len(pos),device=pos.device,dtype=pos.dtype)
                searching=(oldgap>0)&(bound>1e-9)&(distance<=bound+radius)&(~pins) if swept and phase==0 else torch.zeros_like(pins,dtype=torch.bool)
                for _ in range(32):
                    if not searching.any():break
                    a=pa+(ca-pa)*time[:,None];b=pb+(cb-pb)*time[:,None];point=start+(end-start)*time[:,None]
                    feature,_=closest(point,a,b);gap=norm(point-feature)-radius;hit=searching&(gap<=1e-5)
                    crossing|=hit;searching=searching&~hit;time=torch.where(searching,time+gap/bound.clamp_min(1e-30),time);searching&=time<=1
                hit=(distance<radius)|crossing
            at=pa+(ca-pa)*time[:,None];bt=pb+(cb-pb)*time[:,None];point=start+(end-start)*time[:,None]
            feature,along_hit=closest(point,at,bt);normal=torch.where(crossing[:,None],point-feature,end-q)
            along=torch.where(crossing,along_hit,along);q=ca+(cb-ca)*along[:,None]
            normal=torch.where((norm(normal)<1e-8)[:,None],start-oldq,normal)
            fallback=torch.linalg.cross(cb-ca,pos.new_tensor([1,0,0]),dim=-1)
            fallback=torch.where(norm(fallback)<1e-8,pos.new_tensor([0,0,1]),fallback)
            normal=torch.where((norm(normal)<1e-8)[:,None],fallback,normal);normal=normalized(normal)
            residual=((end-q)*normal).sum(-1)-radius;active=hit&(residual<0)&~pins
            if filter_mask is not None:active=active&filter_mask
            pos=torch.where(active[:,None],end-normal*residual[:,None],end)
            bodyvel=((ca-pa)*(1-along[:,None])+(cb-pb)*along[:,None])/dt
            for vertex in active.nonzero().flatten().tolist():
                row=dict(vertex=vertex,normal=normal[vertex],impulse=-residual[vertex]/dt,body_vel=bodyvel[vertex],recovery=bool(recovery[vertex]))
                if vertex in lookup:contacts[lookup[vertex]]=row
                else:lookup[vertex]=len(contacts);contacts.append(row)
    return pos,contacts
