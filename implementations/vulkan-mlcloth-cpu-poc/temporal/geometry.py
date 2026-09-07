"""Broadcast geometry operations with differentiable continuous branches."""
import torch
def dot(a,b):return (a*b).sum(-1)
def norm(a):return torch.linalg.vector_norm(a,dim=-1)
def normalize(a):return a/norm(a).clamp_min(1e-30)[...,None]
def div(a,b):
    # Avoid evaluating 0/0 on inactive torch.where branches.
    return a/torch.where(b.abs()>1e-30,b,torch.ones_like(b))

def closest_triangle(p,a,b,c):
    p,a,b,c=torch.broadcast_tensors(p,a,b,c);ab=b-a;ac=c-a;ap=p-a
    d1=dot(ab,ap);d2=dot(ac,ap);bp=p-b;d3=dot(ab,bp);d4=dot(ac,bp);cp=p-c;d5=dot(ab,cp);d6=dot(ac,cp)
    vc=d1*d4-d3*d2;vb=d5*d2-d1*d6;va=d3*d6-d5*d4;den=va+vb+vc
    u=div(vb,den).clamp(0,1);v=torch.minimum(div(vc,den).clamp_min(0),1-u);bary=torch.stack((1-u-v,u,v),-1)
    done=torch.zeros_like(d1,dtype=torch.bool)
    def choose(mask,weights):
        nonlocal bary,done
        mask=mask&~done;bary=torch.where(mask[...,None],weights,bary);done|=mask
    zero=torch.zeros_like(d1);one=torch.ones_like(d1)
    choose((d1<=0)&(d2<=0),torch.stack((one,zero,zero),-1))
    choose((d3>=0)&(d4<=d3),torch.stack((zero,one,zero),-1))
    t=div(d1,d1-d3);choose((vc<=0)&(d1>=0)&(d3<=0),torch.stack((1-t,t,zero),-1))
    choose((d6>=0)&(d5<=d6),torch.stack((zero,zero,one),-1))
    t=div(d2,d2-d6);choose((vb<=0)&(d2>=0)&(d6<=0),torch.stack((1-t,zero,t),-1))
    t=div(d4-d3,d4-d3+d5-d6);choose((va<=0)&(d4-d3>=0)&(d5-d6>=0),torch.stack((zero,1-t,t),-1))
    area=torch.linalg.cross(ab,ac,dim=-1);edge2=torch.maximum(torch.maximum(dot(ab,ab),dot(ac,ac)),dot(c-b,c-b))
    degenerate=dot(area,area)<=1e-12*edge2.square()
    # Same ordered segment fallback as the Demo, including degenerate triangles.
    best=torch.full_like(d1,float('inf'));fallback=bary
    for x,y,wx,wy in ((a,b,(1,0,0),(0,1,0)),(b,c,(0,1,0),(0,0,1)),(c,a,(0,0,1),(1,0,0))):
        edge=y-x;e2=dot(edge,edge);t=torch.where(e2>1e-20,div(dot(p-x,edge),e2).clamp(0,1),zero)
        q=x+edge*t[...,None];distance=dot(p-q,p-q);improves=distance<best
        weight=p.new_tensor(wx)+(p.new_tensor(wy)-p.new_tensor(wx))*t[...,None]
        fallback=torch.where(improves[...,None],weight,fallback);best=torch.minimum(best,distance)
    bary=torch.where(degenerate[...,None],fallback,bary)
    return a*bary[...,:1]+b*bary[...,1:2]+c*bary[...,2:3],bary

def closest_edges(a,b,c,d):
    d1=b-a;d2=d-c;r=a-c;aa=dot(d1,d1);ee=dot(d2,d2);ff=dot(d2,r);cc=dot(d1,r);bb=dot(d1,d2)
    den=aa*ee-bb*bb;z=torch.zeros_like(den)
    s=torch.where(den.abs()>1e-16,div(bb*ff-cc*ee,den).clamp(0,1),z);t=div(bb*s+ff,ee)
    s=torch.where(t<0,div(-cc,aa).clamp(0,1),torch.where(t>1,div(bb-cc,aa).clamp(0,1),s));t=t.clamp(0,1)
    s=torch.where(aa<=1e-16,z,torch.where(ee<=1e-16,div(-cc,aa).clamp(0,1),s))
    t=torch.where(ee<=1e-16,z,torch.where(aa<=1e-16,div(ff,ee).clamp(0,1),t))
    return a+d1*s[...,None],c+d2*t[...,None],s,t

class SurfaceNormals:
    def __init__(self,triangles,count):
        self.tri=triangles;rows=triangles.detach().cpu().tolist();inc=[[] for _ in range(count)];neighbors=[[i]*3 for i in range(len(rows))];edges={}
        for t,row in enumerate(rows):
            for k,v in enumerate(row):
                inc[v].append((t,k));key=tuple(sorted((v,row[(k+1)%3])))
                if key in edges:ot,ok=edges[key];neighbors[t][k]=ot;neighbors[ot][ok]=t
                else:edges[key]=(t,k)
        self.inc=inc;self.neighbors=torch.tensor(neighbors,device=triangles.device,dtype=torch.long).reshape(-1,3)
    def at(self,points):
        a,b,c=(points[self.tri[:,i]] for i in range(3));face=normalize(torch.linalg.cross(b-a,c-a,dim=-1));vertex=torch.zeros_like(points)
        for k in range(3):
            e1=points[self.tri[:,(k+1)%3]]-points[self.tri[:,k]];e2=points[self.tri[:,(k+2)%3]]-points[self.tri[:,k]]
            angle=torch.atan2(norm(torch.linalg.cross(e1,e2,dim=-1)),dot(e1,e2));vertex=vertex.index_add(0,self.tri[:,k],face*angle[:,None])
        return normalize(vertex),face,normalize(face[:,None]+face[self.neighbors])
    def feature(self,t,bary,normals):
        vertex,face,edge=normals;answer=face[t];done=torch.zeros_like(t,dtype=torch.bool)
        for k in range(3):
            active=(bary[...,k]>=1-1e-6)&~done;chosen=vertex[self.tri[t,k]]
            answer=torch.where((active&(norm(chosen)>.5))[...,None],chosen,answer);done|=active
        for k in range(3):
            active=(bary[...,(k+2)%3]<=1e-6)&~done;chosen=edge[t,k]
            answer=torch.where((active&(norm(chosen)>.5))[...,None],chosen,answer);done|=active
        return answer
    def feature_at_time(self,t,bary,time,previous,current):
        # Only selected swept contacts reach this path. Construct incident normals
        # at each fixed candidate time; topology/candidate choice is discrete.
        def point(v):return previous[v]+(current[v]-previous[v])*time[...,None]
        def face(ids):
            aa,bb,cc=(point(self.tri[ids,k]) for k in range(3));return normalize(torch.linalg.cross(bb-aa,cc-aa,dim=-1))
        answer=face(t);done=torch.zeros_like(t,dtype=torch.bool)
        for k in range(3):
            active=(bary[:,k]>=1-1e-6)&~done
            for index in active.nonzero().flatten().tolist():
                vertex=int(self.tri[t[index],k]);total=answer.new_zeros(3);clock=time[index]
                def vpos(v):return previous[v]+(current[v]-previous[v])*clock
                for it,corner in self.inc[vertex]:
                    row=self.tri[it];a=vpos(row[corner]);e1=vpos(row[(corner+1)%3])-a;e2=vpos(row[(corner+2)%3])-a
                    if float(dot(e1,e1)*dot(e2,e2))>1e-20:total=total+normalize(torch.linalg.cross(e1,e2,dim=-1))*torch.atan2(norm(torch.linalg.cross(e1,e2,dim=-1)),dot(e1,e2))
                if float(norm(total))>1e-12:answer=answer.index_copy(0,answer.new_tensor([index],dtype=torch.long),normalize(total)[None])
            done|=active
        for k in range(3):
            active=(bary[:,(k+2)%3]<=1e-6)&~done;chosen=normalize(face(t)+face(self.neighbors[t,k]));answer=torch.where((active&(norm(chosen)>.5))[:,None],chosen,answer);done|=active
        return answer

def distance_normal(delta,pseudo):
    distance=norm(delta);sign=torch.where(dot(delta,pseudo)<0,-1.,1.)
    return torch.where((distance>1e-8)[...,None],delta*(sign/distance.clamp_min(1e-30))[...,None],pseudo)
