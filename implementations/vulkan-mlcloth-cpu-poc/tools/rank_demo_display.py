"""Compare local garment sources with the locked simulation surface before selecting a mapping."""
import bpy,json,struct,sys
from pathlib import Path
import numpy as np
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from bake_demo_blender import reference,calibration,WORLD
_,rest,_=reference();f,_,_=calibration(bpy.data.objects['Root'],rest)
data=(ROOT/'.work/demo/cloth.dmp').read_bytes();size=struct.unpack_from('<Q',data,8)[0];meta=json.loads(data[16:16+size]);payload=data[16+size:]
def array(key):
    a=meta['arrays'][key];return np.frombuffer(payload,dtype=a['dtype'],count=a['bytes']//4,offset=a['offset']).reshape(a['shape'])
tree=BVHTree.FromPolygons(array('positions').tolist(),array('triangles').tolist(),all_triangles=True)
result=[]
for directory in ['CH_10032','CH_10032_V2','CH_10032_V3','CH_Show_01']:
    for path in Path('F:/ArtWorks/Cloth',directory).glob('*.fbx'):
        bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.fbx(filepath=str(path),use_anim=False)
        for obj in bpy.context.scene.objects:
            if obj.type!='MESH' or any(k in obj.name for k in ['Body_Geo','Head_Geo']):continue
            matrix=WORLD@f@obj.matrix_world
            distances=[tree.find_nearest(matrix@v.co)[3] for v in obj.data.vertices]
            result.append(dict(source=str(path),object=obj.name,vertices=len(distances),
                distance_m=np.percentile(distances,[50,95,99,100]).tolist()))
result.sort(key=lambda a:a['distance_m'][1]);(ROOT/'.work/demo/display-candidates.json').write_text(json.dumps(result,indent=2))
print('DISPLAY_CANDIDATES_COMPLETE',flush=True)
