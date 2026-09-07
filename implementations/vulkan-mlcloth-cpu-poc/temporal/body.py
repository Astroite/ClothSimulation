"""Body triangle selection and live projections, following the Demo contact order."""
import torch
from .geometry import closest_triangle,norm,dot,distance_normal,SurfaceNormals
from .capsule import project_capsules

class BodyProjector:
    def __init__(self,body):
        self.current=body['current'];self.previous=body['previous'];self.tri=body['triangles']
        self.topology=SurfaceNormals(self.tri,len(self.current));self.normals=self.topology.at(self.current)
    def select(self,pos,start,thickness,pins,recovery_only=False,eligible=None):
        tri=self.tri;cur=self.current[tri];old=self.previous[tri];choices=[]
        if not len(tri):return choices
        with torch.no_grad():
            for first in range(0,len(pos),64):
                ids=torch.arange(first,min(first+64,len(pos)),device=pos.device);ids=ids[~pins[ids]]
                if eligible is not None:ids=ids[eligible[ids]]
                if not len(ids):continue
                end=pos[ids];begin=start[ids];q,bary=closest_triangle(end[:,None],cur[None,:,0],cur[None,:,1],cur[None,:,2])
                distance=norm(end[:,None]-q);ts=torch.arange(len(tri),device=pos.device)[None].expand(len(ids),-1)
                normal=distance_normal(end[:,None]-q,self.topology.feature(ts,bary,self.normals));signed=dot(end[:,None]-q,normal)
                distance=torch.where((norm(self.normals[1])>=.5)[None],distance,torch.full_like(distance,float('inf')))
                minimum=distance.min(-1).values;nearest=(distance<=minimum[:,None]+1e-6).long().argmax(-1)
                selected={}
                if not recovery_only:
                    motion=end-begin
                    bound=norm(cur[None]-old[None]-motion[:,None,None]).max(-1).values
                    row,col=torch.where(distance<=bound+thickness+1e-5)
                    if len(row):
                        a,b,c=cur[col].unbind(1);oa,ob,oc=old[col].unbind(1);p0,p1=begin[row],end[row]
                        rate=bound[row,col];time=torch.zeros_like(rate);hit=torch.zeros_like(time,dtype=torch.bool);bary_hit=bary[row,col];search=rate>1e-9
                        for _ in range(32):
                            if not search.any():break
                            t=time[:,None];point=p0+(p1-p0)*t;cl,bc=closest_triangle(point,oa+(a-oa)*t,ob+(b-ob)*t,oc+(c-oc)*t);gap=norm(point-cl)
                            found=search&(gap<=thickness+1e-5);bary_hit=torch.where(found[:,None],bc,bary_hit);hit|=found;search=search&~found
                            time=torch.where(search,time+(gap-thickness)/rate.clamp_min(1e-30),time);search&=time<=1
                        swept=hit.clone();time=torch.where(hit,time,torch.where(rate<=1e-9,torch.zeros_like(time),torch.ones_like(time)))
                        hit|=distance[row,col]<thickness
                        qcur=(cur[col]*bary_hit[:,:,None]).sum(1);point=p0+(p1-p0)*time[:,None]
                        at=old[col]+(cur[col]-old[col])*time[:,None,None];qat=(at*bary_hit[:,:,None]).sum(1)
                        pseudo=self.topology.feature_at_time(col,bary_hit,time,self.previous,self.current)
                        nn=torch.where(swept[:,None],distance_normal(point-qat,pseudo),normal[row,col])
                        active=hit&(dot(p1-qcur,nn)<thickness)
                        for r in range(len(ids)):
                            candidates=((row==r)&active).nonzero().flatten()
                            if not len(candidates):continue
                            earliest=time[candidates].min();candidates=candidates[time[candidates]<=earliest+1e-6]
                            best=distance[row[candidates],col[candidates]].min();candidates=candidates[distance[row[candidates],col[candidates]]<=best+1e-6]
                            k=candidates[col[candidates].argmin()];selected[r]=(int(col[k]),float(time[k]),bool(swept[k]),False)
                for r,v in enumerate(ids.tolist()):
                    if r in selected:choices.append((v,*selected[r]))
                    elif torch.isfinite(minimum[r]) and signed[r,nearest[r]]<(-1e-6 if recovery_only else 0):
                        choices.append((v,int(nearest[r]),1.,False,True))
        return choices
    def apply(self,pos,start,thickness,dt,choices,contacts):
        if not choices:return pos,contacts
        ids=torch.tensor([r[0] for r in choices],device=pos.device);tri_ids=torch.tensor([r[1] for r in choices],device=pos.device)
        time=pos.new_tensor([r[2] for r in choices]);swept=torch.tensor([r[3] for r in choices],device=pos.device,dtype=torch.bool)
        current=self.current[self.tri[tri_ids]];previous=self.previous[self.tri[tri_ids]];point=start[ids]+(pos[ids]-start[ids])*time[:,None]
        at=previous+(current-previous)*time[:,None,None]
        q,bary=closest_triangle(point,at[:,0],at[:,1],at[:,2]);qcurrent=(current*bary[:,:,None]).sum(1)
        # Selected times/IDs are fixed. Closest points, normals, residual and
        # friction impulse remain in the differentiable graph.
        pseudo=self.topology.feature_at_time(tri_ids,bary,time,self.previous,self.current)
        normal=distance_normal(point-q,pseudo);residual=dot(pos[ids]-qcurrent,normal)-thickness
        correction=normal*(-residual).clamp_min(0)[:,None];new_pos=pos.index_add(0,ids,correction)
        lookup={r['vertex']:i for i,r in enumerate(contacts)};bodyvel=((current-previous)*bary[:,:,None]).sum(1)/dt
        for i,row in enumerate(choices):
            if float(residual[i])>=0:continue
            contact=dict(vertex=row[0],normal=normal[i],body_vel=bodyvel[i],impulse=(-residual[i]/dt).clamp_min(0),recovery=row[4])
            if row[0] in lookup:contacts[lookup[row[0]]]=contact
            else:lookup[row[0]]=len(contacts);contacts.append(contact)
        return new_pos,contacts

def project_body(s,pos,previous,body,thickness,dt,pins):
    contacts=[];projector=BodyProjector(body)
    choices=projector.select(pos,previous,thickness,pins);pos,contacts=projector.apply(pos,previous,thickness,dt,choices,contacts)
    pos,contacts=project_capsules(s,pos,previous,body.get('capsules',[]),thickness,dt,pins,contacts)
    for _ in range(2):
        eligible=torch.zeros_like(pins);eligible[[c['vertex'] for c in contacts]]=True
        choices=projector.select(pos,previous,thickness,pins,True,eligible)
        if not choices:break
        pos,contacts=projector.apply(pos,previous,thickness,dt,choices,contacts)
        changed=torch.zeros_like(pins);changed[[r[0] for r in choices]]=True
        pos,contacts=project_capsules(s,pos,previous,body.get('capsules',[]),thickness,dt,pins,contacts,swept=False,filter_mask=changed)
    return pos,contacts
