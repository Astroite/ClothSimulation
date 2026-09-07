"""Read-only inventory of display-cloth candidates in Blender."""
import bpy
import json
from pathlib import Path
from mathutils import Vector
result=[]
for source in ['F:/ArtWorks/Cloth/CH_10032_V3/SM_C10032_Cloth_01.fbx',
               'F:/ArtWorks/Cloth/CH_Show_01/SM_Cloth_Show_01.fbx']:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=source,use_anim=False)
    for obj in bpy.context.scene.objects:
        if obj.type!='MESH': continue
        points=[obj.matrix_world@v.co for v in obj.data.vertices]
        result.append(dict(source=source,name=obj.name,vertices=len(points),faces=len(obj.data.polygons),
            bounds=[[min(p[i] for p in points),max(p[i] for p in points)] for i in range(3)],
            materials=[m.name for m in obj.data.materials],uv=[u.name for u in obj.data.uv_layers]))
Path(__file__).resolve().parents[1].joinpath('.work/demo/cloth-inventory.json').write_text(json.dumps(result,indent=2))
