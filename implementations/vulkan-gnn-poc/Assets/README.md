# PoC assets

This directory keeps runtime and research inputs separate from source code:

- `Meshes/`: the processed CH10032 lower-body cloth mesh used by the custom
  mesh integration work.
- `TrainingSamples/`: a compact six-motion subset copied from the local
  `F:\Projects\Anim` export, with raw motion, a convenience 60 fps resample,
  and a validated SMPL body-22 skeleton FBX for each motion.
- `Characters/CH10032/`: the body reference FBX and sidecar used for alignment,
  skinning, and future collision-proxy work.

Run `tools/import_motion_samples.py` to reproduce the local motion subset. The
checked-in `TrainingSamples/manifest.json` records relative paths, file sizes,
SHA-256 hashes, source motion metadata, and the original validation results.

## Data terms

The PoC source code is MIT licensed, but that does **not** relicense these
motion, body, or garment assets. HumanML3D's repository code is MIT licensed;
its own documentation states that AMASS-derived data cannot be redistributed
directly. Keep these copied assets in an authorized/private workspace and
review the HumanML3D, AMASS, KIT, BMLmovi, SMPL, and original character/garment
terms before publishing or redistributing them.
