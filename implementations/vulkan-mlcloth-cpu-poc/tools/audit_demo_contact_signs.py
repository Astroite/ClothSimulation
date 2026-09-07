"""Independent multi-ray winding audit of nearest-face penetration diagnostics."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);snapshot=json.loads(args.snapshot.read_text(encoding='utf8'))
directions=[Vector(v).normalized() for v in [(1,.371,.127),(-.247,.613,1),(.439,1,-.293)]]
report={'qualification':False,'method':'signed ray crossing counts in three directions; nonzero positive winding denotes material, discrepancies remain ambiguous','actors':[]}
def winding(tree,point,direction):
    origin=point.copy();count=0
    for _ in range(64):
        hit,normal,face,distance=tree.ray_cast(origin,direction,10.)
        if hit is None:return count
        dot=normal.dot(direction)
        if abs(dot)<1e-7:return None
        count+=1 if dot>0 else -1
        origin=hit+direction*.00001
    return None
for actor in snapshot['actors'][1:]:
    entry={'actor':actor['algorithm'],'surfaces':{}}
    for surface in ['body','proxy']:
        tree=BVHTree.FromPolygons(actor[surface],np.array(snapshot[surface+'_triangles']).reshape(-1,3).tolist(),all_triangles=True)
        rows=[]
        for vertex,p in enumerate(actor['cloth']):
            point=Vector(p);q,normal,face,distance=tree.find_nearest(point);depth=-(point-q).dot(normal)
            if depth<=.001:continue
            counts=[winding(tree,point,d) for d in directions]
            consensus=all(v is not None and v==counts[0] for v in counts)
            rows.append(dict(vertex=vertex,nearest_face_depth_m=depth,nearest_distance_m=distance,winding=counts,
                consensus=consensus,inside=bool(consensus and counts[0]>0)))
        positive=[r['nearest_distance_m'] for r in rows if r['inside']]
        entry['surfaces'][surface]=dict(nearest_face_candidates=len(rows),inside=sum(r['inside'] for r in rows),
            outside=sum(r['consensus'] and not r['inside'] for r in rows),ambiguous=sum(not r['consensus'] for r in rows),
            confirmed_inside_distance_p95_m=float(np.quantile(positive,.95)) if positive else 0,
            worst=sorted(rows,key=lambda r:-r['nearest_face_depth_m'])[:20])
    report['actors'].append(entry)
args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2),flush=True)
