"""Synthetic bone-length and root-velocity regression, executed in Blender."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from mathutils import Matrix,Vector
from demo_pose_blend import compose,mix_pose_local,close_loop_local

parents=[-1,0,1]
root=Matrix.Identity(4);joint=Matrix.Identity(4);joint.translation=Vector((0,0,1));tip=joint.copy()
a=compose([root,joint,tip],parents)
b=compose([root,Matrix.Rotation(np.pi/2,4,'Y')@joint,tip],parents)
mixed=mix_pose_local(a,b,.5,parents)
assert abs(np.linalg.norm(mixed[2,:3,3]-mixed[1,:3,3])-1)<1e-6
# A fully translating loop needs no pose correction; preserve every root sample
# and its endpoint speed, including the tail previously blended to a fixed pose.
frames=[]
for i in range(61):
    r=root.copy();r.translation=Vector((i/60,0,0));frames.append(compose([r,joint,tip],parents))
frames=np.array(frames);closed=close_loop_local(frames,0,parents)
assert np.max(np.abs(closed-frames))<1e-6
assert abs((closed[-1,0,0,3]-closed[-2,0,0,3])*60-1)<1e-5
print('PASS: local-space transition preserves child length and moving-loop root velocity')
