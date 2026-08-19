# Processed cloth meshes

`CH10032_lower_sim` is the lower-body skirt extracted from
`USD_CH10032_Cloth_2607_V1_S.usd` at `/Garment/SimMesh`.

The conversion selects `pattern20-23` and `pattern26-29`, welds all internal
`SewingAPI` vertex pairs, converts the source millimeter stage to meters, and
places the waist near `Y=0` in a right-handed Y-up coordinate system.

Generated files:

- `CH10032_lower_sim.vcloth`: runtime-oriented VCLOTH v1 binary.
- `CH10032_lower_sim.obj`: geometry-only inspection/debug mesh.
- `CH10032_lower_sim.json`: provenance, hashes, topology, constraints, section
  layout, and pinned waist-loop indices.

The welded asset contains 1,377 vertices and 2,570 triangles. Its triangle
one-ring CSR contains 7,894 directed edges. The asset also stores 3,947
edge-length constraints, 3,763 four-vertex dihedral bending constraints, and a
72-vertex pinned waist loop. Constraints are ordered in vertex-disjoint color
batches; color offsets are stored in the binary.

Regenerate with Blender 4.5 or another Python runtime containing Pixar USD:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe' `
  --background --factory-startup `
  --python 'E:\Projects\cloth-simulation\implementations\vulkan-gnn-poc\tools\convert_usd_cloth.py' `
  -- `
  --input 'F:\AICloth\CH10032_2607\Mesh\USD_CH10032_Cloth_2607_V1_S.usd' `
  --output-dir 'E:\Projects\cloth-simulation\implementations\vulkan-gnn-poc\Assets\Meshes' `
  --name 'CH10032_lower_sim' `
  --patterns '20-23,26-29'
```

The original USD is not copied into this directory. The JSON sidecar records
its filename and SHA-256 so the generated asset can be traced and reproduced.

## Standalone simulation LOD1

`CH10032_lower_sim_lod1` is a separate VCLTH v2 asset generated from the baked
1,377-vertex runtime cloth. It does not replace `CH10032_lower.vcloth2` and is
not read by the student-distillation scripts or their existing goldens.

- 732 vertices and 1,280 triangles (46.8% / 50.2% reduction).
- Both source boundary loops are protected; the 72-vertex waist pin loop is
  unchanged.
- UV, 12-slot LBS, masses, CSR, stretch and dihedral-bend constraints are
  regenerated for the new topology.
- `CH10032_lower_sim_lod1.json` records geometry error, hashes and topology;
  the OBJ is an inspection copy of the exact VCLTH v2 geometry.

Regenerate without touching the training environment:

```powershell
.\tools\build_cloth_lod.ps1 -TargetTriangles 1280
```

This launches Blender's isolated Python runtime. The default output is always
the `_lod1` asset under `Assets/Meshes`; the script rejects an output path equal
to the source runtime cloth.
