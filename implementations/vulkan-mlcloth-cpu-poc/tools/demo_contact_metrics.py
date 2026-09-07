"""Blender reference solid classification, independent of runtime pseudonormals."""
import numpy as np
from mathutils import Vector
DIRECTIONS=[Vector(v).normalized() for v in [(1,.371,.127),(-.247,.613,1),(.439,1,-.293)]]
def winding(tree,point,direction):
    origin=point.copy();count=0
    for _ in range(64):
        hit,normal,face,distance=tree.ray_cast(origin,direction,10.)
        if hit is None:return count
        dot=normal.dot(direction)
        if abs(dot)<1e-7:return None
        count+=1 if dot>0 else -1;origin=hit+direction*.00001
    return None
def solid_metrics(tree,cloth,surface,distances,pinned):
    surface=np.asarray(surface);lo=surface.min(axis=0);hi=surface.max(axis=0)
    inside=[];ambiguous=[];overlapping=[]
    for vertex,p in enumerate(cloth):
        if np.any(p<lo) or np.any(p>hi):continue
        counts=[winding(tree,Vector(p),direction) for direction in DIRECTIONS]
        if any(n is None or n!=counts[0] for n in counts) or counts[0]<0:ambiguous.append(vertex);continue
        if counts[0]>0 and distances[vertex]>1e-6:inside.append(vertex)
        if counts[0]>1:overlapping.append(vertex)
    depths=np.asarray(distances)[inside]
    return dict(method='three signed ray crossing counts; positive unanimous winding; Euclidean distance to nearest surface',
        penetrating_vertices=len(inside),penetrating_pins=int(np.count_nonzero(np.asarray(pinned)[inside])),
        ambiguous_vertices=len(ambiguous),overlapping_volume_vertices=len(overlapping),
        penetration_p95_m=float(np.quantile(depths,.95)) if len(depths) else 0.,penetration_max_m=float(depths.max()) if len(depths) else 0.,
        worst=[dict(vertex=int(v),depth_m=float(distances[v]),pinned=bool(pinned[v])) for v in sorted(inside,key=lambda v:-distances[v])[:8]],
        limitation='Open/intersecting body surfaces and ambiguous rays require review; this is not a quality qualification.')
