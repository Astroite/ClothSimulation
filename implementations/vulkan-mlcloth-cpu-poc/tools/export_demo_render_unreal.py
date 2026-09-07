"""Read-only export of the render/sim source meshes belonging to Cloth_2607.

Run through UnrealEditor-Cmd with -nullrhi. UE assets are never saved or modified.
"""
import json
from pathlib import Path
import unreal

out=Path(__file__).resolve().parents[1]/'.work/demo/source/render2607'
out.mkdir(parents=True,exist_ok=True)
base='/Game/Developers/jinzhao/AICloth/CH_10032/Model/V1/Cloth_2607/'
sources={'render':'RenderMesh/SM_CH10032_Cloth_2607_V1_R',
         'sim':'CCA_JZ_CH10032_2607_Import/USD_CH10032_Cloth_2607_V1_S_88AB03B7/StaticMeshes/SM_SimMesh'}
report={}
for label,relative in sources.items():
    asset=unreal.load_asset(base+relative)
    if not isinstance(asset,unreal.StaticMesh):raise RuntimeError('Static mesh missing: '+base+relative)
    task=unreal.AssetExportTask();task.object=asset;task.filename=str(out/(label+'.fbx'))
    task.automated=True;task.prompt=False;task.replace_identical=True
    task.exporter=unreal.StaticMeshExporterFBX();task.options=unreal.FbxExportOption()
    if not unreal.Exporter.run_asset_export_task(task):raise RuntimeError('Mesh export failed: '+label)
    materials=[]
    for slot in asset.get_editor_property('static_materials'):
        material=slot.material_interface
        entry=dict(slot=str(slot.material_slot_name),material=material.get_path_name() if material else None,textures=[],vectors={},scalars={})
        if material:
            lib=unreal.MaterialEditingLibrary;instance=isinstance(material,unreal.MaterialInstanceConstant)
            for name in lib.get_texture_parameter_names(material):
                texture=(lib.get_material_instance_texture_parameter_value if instance else lib.get_material_default_texture_parameter_value)(material,name)
                if texture:
                    filename=texture.get_name()+'.tga';path=out/filename
                    if not path.exists():
                        export=unreal.AssetExportTask();export.object=texture;export.filename=str(path);export.automated=True;export.prompt=False;export.replace_identical=True;export.exporter=unreal.TextureExporterTGA()
                        if not unreal.Exporter.run_asset_export_task(export):raise RuntimeError('Texture export failed: '+texture.get_path_name())
                    entry['textures'].append(dict(parameter=str(name),source=texture.get_path_name(),file=filename))
            for name in lib.get_vector_parameter_names(material):
                color=(lib.get_material_instance_vector_parameter_value if instance else lib.get_material_default_vector_parameter_value)(material,name)
                entry['vectors'][str(name)]=[color.r,color.g,color.b,color.a]
            for name in lib.get_scalar_parameter_names(material):
                entry['scalars'][str(name)]=(lib.get_material_instance_scalar_parameter_value if instance else lib.get_material_default_scalar_parameter_value)(material,name)
        materials.append(entry)
    report[label]=dict(source=asset.get_path_name(),file=task.filename,materials=materials)
(out/'sources.json').write_text(json.dumps(report,indent=2),encoding='utf8')
unreal.log('[DemoRenderExport] COMPLETE')
