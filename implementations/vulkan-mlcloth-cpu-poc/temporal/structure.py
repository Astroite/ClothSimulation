"""Functional, color-batched structural projections in Demo CPU visitation order."""
import torch

def color_records(records,device):
    occupied=[];groups=[]
    for i,row in enumerate(records):
        vertices=set(row);group=next((j for j,used in enumerate(occupied) if not used&vertices),len(groups))
        if group==len(groups):groups.append([]);occupied.append(set())
        groups[group].append(i);occupied[group].update(vertices)
    return [torch.tensor(g,device=device,dtype=torch.long) for g in groups]

def build_structure(s):
    records=[s.stretch_pairs,[(r.a,r.b,r.c) for r in s.shear_records],[(r.a,r.b,r.c,r.d) for r in s.bend_records]]
    rests=[s.stretch_rest,[r.rest_dot for r in s.shear_records],[r.rest_angle for r in s.bend_records]]
    s.structure=[]
    for k,(ids,rest) in enumerate(zip(records,rests)):
        s.structure.append((torch.tensor(ids,device=s.device,dtype=torch.long).reshape(-1,k+2),s.rest.new_tensor(rest),color_records(ids,s.device)))

def project(s,pos,kind,compliance,dt,lambdas):
    records,rest,groups=s.structure[kind];alpha=max(0.,compliance)/(dt*dt)
    def dot(a,b):return (a*b).sum(-1)
    def norm(a):return torch.linalg.vector_norm(a,dim=-1)
    for group in groups:
        ids=records[group];p=pos[ids];mass=s.inv_mass[ids];e=p[:,1]-p[:,0]
        if kind==0:
            length=norm(e);active=length>=1e-12
            grad=e/length.clamp_min(1e-30)[:,None];g=torch.stack((-grad,grad),1);residual=length-rest[group]
        elif kind==1:
            f=p[:,2]-p[:,0];g=torch.stack((-e-f,f,e),1);residual=dot(e,f)-rest[group];active=torch.ones_like(residual,dtype=torch.bool)
        else:
            f=p[:,2]-p[:,0];h=p[:,3]-p[:,0];n1=torch.linalg.cross(e,f,dim=-1);n2=torch.linalg.cross(e,h,dim=-1)
            el=norm(e);l1=norm(n1);l2=norm(n2);active=(el>=1e-12)&(l1>=1e-12)&(l2>=1e-12)
            el=el.clamp_min(1e-30);l1=l1.clamp_min(1e-30);l2=l2.clamp_min(1e-30)
            u=n1/l1[:,None];v=n2/l2[:,None];axis=e/el[:,None]
            sin=dot(torch.linalg.cross(u,v,dim=-1),axis);cos=dot(u,v).clamp(-1,1)
            angle=torch.atan2(torch.where(active,sin,torch.zeros_like(sin)),torch.where(active,cos,torch.ones_like(cos)))
            residual=torch.atan2(torch.sin(angle-rest[group]),torch.cos(angle-rest[group]))
            gc=-u*(el/l1)[:,None];gd=v*(el/l2)[:,None];tc=dot(f,e)/el.square();td=dot(h,e)/el.square()
            ga=-(1-tc)[:,None]*gc-(1-td)[:,None]*gd;gb=-tc[:,None]*gc-td[:,None]*gd;g=torch.stack((ga,gb,gc,gd),1)
        denominator=(mass*(g*g).sum(-1)).sum(-1)+alpha
        if kind==1:active=active&(denominator>=1e-12)
        dl=torch.where(active,(-residual-alpha*lambdas[group])/denominator.clamp_min(1e-20),torch.zeros_like(residual))
        lambdas=lambdas.index_copy(0,group,lambdas[group]+dl)
        # No vertex repeats within a color. Functional updates preserve the
        # complete rollout gradient and avoid a clone per individual constraint.
        correction=g*(mass*dl[:,None])[:,:,None]
        pos=pos.index_add(0,ids.reshape(-1),correction.reshape(-1,3))
    return pos,lambdas
