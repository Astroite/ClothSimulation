"""Diagnose residual body contact after a saved solve; does not modify runtime assets."""
import json, numpy as np, sys, argparse
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
root=Path(__file__).resolve().parents[1]/'.work/demo'
parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,default=root/'rollouts-20260907-161741/jump/physics-snapshot.json')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
snapshot=json.loads(args.snapshot.read_text(encoding='utf-8'))
tris=np.array(snapshot['proxy_triangles']).reshape(-1,3);body_tris=np.array(snapshot['body_triangles']).reshape(-1,3)
pins=np.array(snapshot['pinned'])>0;report=[]
for actor in snapshot['actors'][1:]:
    cloth=np.array(actor['cloth']);tree=BVHTree.FromPolygons(actor['proxy'],tris.tolist(),all_triangles=True)
    body=BVHTree.FromPolygons(actor['body'],body_tris.tolist(),all_triangles=True);entries=[]
    for iteration in range(5):
        depths=[]
        for point in cloth:
            q,n,_,_=body.find_nearest(Vector(point));depths.append(max(0.,-(Vector(point)-q).dot(n)))
        positive=np.array(depths);positive=positive[positive>1e-6]
        entries.append(dict(iteration=iteration,p95_mm=float(np.quantile(positive,.95)*1000) if len(positive) else 0.,max_mm=max(depths)*1000,vertices=len(positive)))
        for i,point in enumerate(cloth):
            if pins[i]:continue
            q,n,_,distance=tree.find_nearest(Vector(point));signed=(Vector(point)-q).dot(n)
            if signed<0 or distance<.003:cloth[i]+=np.array(n)*(.003-signed)
            for capsule in actor['capsules']:
                a=np.array(capsule['a']);b=np.array(capsule['b']);axis=b-a;t=np.clip((cloth[i]-a)@axis/max(axis@axis,1e-16),0,1)
                delta=cloth[i]-(a+axis*t);distance=np.linalg.norm(delta);radius=capsule['radius']+.003
                if distance<radius and distance>1e-8:cloth[i]+=delta*(radius/distance-1)
    report.append(dict(algorithm=actor['algorithm'],static_recovery=entries))
(root/'collision-recovery-probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
