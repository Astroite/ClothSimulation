"""Blender pose transitions in parent-local space, followed by forward kinematics."""
import numpy as np
from mathutils import Matrix,Quaternion

def local_pose(pose,parents):
    matrices=[Matrix(m.tolist()) for m in pose]
    return [matrices[parent].inverted()@m if parent>=0 else m.copy() for m,parent in zip(matrices,parents)]

def compose(local,parents):
    world=[]
    for matrix,parent in zip(local,parents):
        world.append(world[parent]@matrix if parent>=0 else matrix)
    return np.array([np.array(m) for m in world])

def mix_pose_local(a,b,t,parents):
    local=[]
    for x,y in zip(local_pose(a,parents),local_pose(b,parents)):
        rotation=x.to_quaternion().slerp(y.to_quaternion(),t)
        matrix=rotation.to_matrix().to_4x4();matrix.translation=x.translation.lerp(y.translation,t)
        local.append(matrix)
    return compose(local,parents)

def close_loop_local(frames,root,parents,blend_frames=18):
    """Fade endpoint pose error into a moving source, not toward a frozen end pose."""
    result=frames.copy();target=frames[0].copy()
    delta=frames[-1,root,:3,3]-frames[0,root,:3,3];delta[2]=0
    moving={root}
    for i,parent in enumerate(parents):
        if parent in moving:moving.add(i)
    for bone in moving:target[bone,:3,3]+=delta
    end=local_pose(frames[-1],parents);goal=local_pose(target,parents)
    rotations=[b.to_quaternion()@a.to_quaternion().inverted() for a,b in zip(end,goal)]
    shifts=[b.translation-a.translation for a,b in zip(end,goal)]
    count=min(blend_frames,len(frames)//3)
    for k in range(count+1):
        u=k/max(1,count);u=u*u*(3-2*u);index=len(frames)-count-1+k
        local=local_pose(frames[index],parents)
        for bone,matrix in enumerate(local):
            correction=Quaternion((1,0,0,0)).slerp(rotations[bone],u)
            rotation=correction@matrix.to_quaternion();position=matrix.translation+shifts[bone]*u
            local[bone]=rotation.to_matrix().to_4x4();local[bone].translation=position
        result[index]=compose(local,parents)
    return result
