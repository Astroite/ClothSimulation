# Vulkan MLCloth CPU inference upload PoC

Windows-only validation sample for the deliberately narrow path:

```text
CH10032 driver clip (30 Hz) -> AILab/MNN CPU inference
-> 5,294 Root_M-local UE-cm points -> per-frame Vulkan upload
-> compute root/axis/unit transform -> point-list rendering
```

It has an independent pinned Sascha Willems checkout under `.work/Vulkan` and
builds `mlclothcpu.exe`. Models and vendor DLLs are copied from the local
MLCloth plugin into ignored `.work/runtime`; no vendor binary is committed.

```powershell
pwsh ./prepare_runtime.ps1
pwsh ./bake_driver_clip.ps1
pwsh ./build.ps1
pwsh ./run.ps1 -Verify -Validation
```

The driver bake defaults to `E:\Main\Projects\Z2Game\Z2Game.uproject` and
`E:\Main\Engine\Binaries\Win64\UnrealEditor-Cmd.exe`; all asset paths and the
project/editor path remain CLI-overridable. Z2Game currently contains the
CH10032 mesh and animation but not the locked `.enc` model or vendor runtime
DLLs. Therefore `prepare_runtime.ps1` defaults to the legacy PaperGame MLCloth
plugin as the artifact source. Override `-MLClothRoot` after those artifacts
are synchronized into another workspace.

Use `run.ps1 -Benchmark -Threads 1` for a 200-frame warm-up and 1,000-frame
capture. `P` pauses, `R` deterministically resets and `Esc` exits.

This PoC excludes XPBD, collisions, triangle topology, 5,294-to-1,377 mapping,
normals/materials and MNN GPU execution.
