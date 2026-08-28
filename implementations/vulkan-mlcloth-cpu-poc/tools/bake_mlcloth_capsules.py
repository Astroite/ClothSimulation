#!/usr/bin/env python3
"""Bake the body physics asset's capsules into `MLCAP001` for the XPBD contact pass.

Why capsules, and why this is nearly free
-----------------------------------------
The sibling GNN PoC collides against a 4,096-vertex body point proxy and resolves
contacts as a half-plane at the nearest proxy. `results/PROGRESS.md` section 6
records the defect that comes with it: the criterion is *blind to tunnelling* --
pushing the hem patch 0.12 m into the body reports 28.5% penetration while 0.25 m
reports 1%, because once a vertex is out the far side its nearest proxy is the far
surface and the signed distance is positive again. The solver uses the same test,
so it is blind in the same way.

A capsule has a closed-form signed distance, and inside is unambiguously inside, so
this is a correction to that design rather than a port of it.

It is also almost free here. `body_physics_runtime.json` in the sibling PoC records
that the physics asset the character actually references at runtime -- "therefore
the collision the cloth solver must respect" -- has 14 bodies, every one of them a
`SphylElems` capsule, on these bones:

    Root_M  Spine1_M  Chest_M  Neck1_M  Shoulder_L/R  Wrist_L/R
    Hip_L/R  Knee_L/R  KneePart2_L/R

All fourteen are among the model's 45 driver bones, which was checked against the
`.enc` driver list, so the `.mldrv` clip already carries their per-frame component
transforms. No new per-frame data, no 4,096-point proxy, and no runtime skinning:
the capsules are a one-time static bake and the animation is already in hand.

What is *not* solved by this: self-collision, and cloth passing between two capsules
that do not cover the gap between them. The 14 capsules approximate a body; for a
skirt the load-bearing ones are Root_M, Hip_L/R, Knee_L/R and KneePart2_L/R.

Unreal conventions this depends on, both verified against the 5.8 headers:

* `FKSphylElem` -- "Z axis is capsule axis", and `Length` is the length of the
  *line segment*, so the total height is `Length + 2 * Radius`
  (`Engine/Classes/PhysicsEngine/SphylElem.h`).
* `FRotationMatrix` builds its rows as the local basis from (Pitch, Yaw, Roll) in
  degrees, so the capsule axis is the third row
  (`Core/Public/Math/RotationTranslationMatrix.h`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
GNN_POC = POC_ROOT.parent / "vulkan-gnn-poc"
DEFAULT_T3D = GNN_POC / ".work/ch10032_library/data/body_physics_runtime.t3d"

MAGIC = b"MLCAP001"
VERSION = 1
DRIVER_COUNT = 45

_BODY = re.compile(
    r'Begin Object Name="(?P<object>[^"]+)".*?'
    r"AggGeom=\((?P<geom>.*?)\)\s*\n"
    r'.*?BoneName="(?P<bone>[^"]+)"',
    re.DOTALL,
)
_SPHYL_BLOCK = re.compile(r"SphylElems=\((?P<body>.*)\)\s*$", re.DOTALL)
_SPHYL = re.compile(
    r"Center=\(X=(?P<cx>[-\d.eE+]+),Y=(?P<cy>[-\d.eE+]+),Z=(?P<cz>[-\d.eE+]+)\),"
    r"Rotation=\(Pitch=(?P<pitch>[-\d.eE+]+),Yaw=(?P<yaw>[-\d.eE+]+),Roll=(?P<roll>[-\d.eE+]+)\),"
    r"Radius=(?P<radius>[-\d.eE+]+),Length=(?P<length>[-\d.eE+]+)"
)
_OTHER_GEOM = re.compile(r"(?P<kind>[A-Za-z]+Elems)=")


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"bake_mlcloth_capsules: {message}")


def rotator_axis(pitch_deg: float, yaw_deg: float, roll_deg: float) -> tuple[float, float, float]:
    """The local Z axis of Unreal's FRotator, which is the capsule's axis.

    Third row of FRotationMatrix. Written out rather than composed from three
    separate rotations because Unreal's rotator is not a plain XYZ Euler triple and
    an independently "reasonable" composition order gives a different axis.
    """
    sp, cp = math.sin(math.radians(pitch_deg)), math.cos(math.radians(pitch_deg))
    sy, cy = math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg))
    sr, cr = math.sin(math.radians(roll_deg)), math.cos(math.radians(roll_deg))
    axis = (-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp)
    norm = math.sqrt(sum(component * component for component in axis))
    if norm < 1.0e-9:
        fail(f"rotator ({pitch_deg}, {yaw_deg}, {roll_deg}) produced a degenerate axis")
    return tuple(component / norm for component in axis)  # type: ignore[return-value]


def capsule_signed_distance(
    point: tuple[float, float, float],
    center: tuple[float, float, float],
    axis: tuple[float, float, float],
    radius: float,
    half_length: float,
) -> float:
    """Signed distance from `point` to the capsule surface; negative means inside.

    The reference implementation for the HLSL contact block. Clamping the projection
    to the segment is what makes this exact for the spherical caps as well as the
    cylinder, and it is why a vertex that has passed through the middle of a limb
    still reports negative -- the property the nearest-proxy half-plane lacks.
    """
    dx, dy, dz = (point[i] - center[i] for i in range(3))
    projection = dx * axis[0] + dy * axis[1] + dz * axis[2]
    projection = max(-half_length, min(half_length, projection))
    ox = dx - axis[0] * projection
    oy = dy - axis[1] * projection
    oz = dz - axis[2] * projection
    return math.sqrt(ox * ox + oy * oy + oz * oz) - radius


def read_driver_names(path: Path) -> list[str]:
    data = path.read_bytes()
    if len(data) < 5:
        fail(f"encoded model is truncated: {path}")
    json_bytes = struct.unpack_from("<I", data, 0)[0]
    if json_bytes <= 1 or 4 + json_bytes >= len(data):
        fail(f"encoded model JSON length is invalid: {path}")
    config = json.loads(data[4 : 4 + json_bytes].decode("utf-8"))
    names = config.get("driverNames")
    if not isinstance(names, list) or len(names) != DRIVER_COUNT:
        fail(f"encoded model must declare exactly {DRIVER_COUNT} driverNames")
    if names[0] != "Root_M":
        fail("the first driver must be Root_M")
    return [str(name) for name in names]


def parse_t3d(path: Path) -> tuple[list[dict], list[str]]:
    """Every capsule in the physics asset, plus a note of any geometry that is not one."""
    text = path.read_text(encoding="utf-8", errors="replace")
    capsules: list[dict] = []
    skipped: list[str] = []
    for match in _BODY.finditer(text):
        geometry = match.group("geom")
        bone = match.group("bone")
        for kind in {other.group("kind") for other in _OTHER_GEOM.finditer(geometry)}:
            if kind != "SphylElems":
                # Reported rather than ignored: silently dropping a collision primitive
                # would understate the body and make the penetration column optimistic.
                skipped.append(f"{bone}:{kind}")
        block = _SPHYL_BLOCK.search(geometry)
        if not block:
            continue
        for element in _SPHYL.finditer(block.group("body")):
            radius = float(element.group("radius"))
            length = float(element.group("length"))
            if radius <= 0.0 or length < 0.0:
                fail(f"{bone} has a capsule with radius {radius} and length {length}")
            capsules.append(
                {
                    "bone": bone,
                    "object": match.group("object"),
                    "center_cm": (float(element.group("cx")), float(element.group("cy")), float(element.group("cz"))),
                    "rotation_pyr_deg": (
                        float(element.group("pitch")),
                        float(element.group("yaw")),
                        float(element.group("roll")),
                    ),
                    "radius_cm": radius,
                    "segment_length_cm": length,
                }
            )
    if not capsules:
        fail(f"no SphylElems capsules found in {path}")
    return capsules, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--physics-t3d",
        type=Path,
        default=DEFAULT_T3D,
        help="text export of the runtime physics asset. Defaults to the sibling GNN PoC's copy, "
             "which is the one its results are measured against.",
    )
    parser.add_argument("--model", type=Path, default=POC_ROOT / ".work/runtime/model_NeuralRes4_NeuralRes4_final.enc")
    parser.add_argument("--output", type=Path, default=POC_ROOT / ".work/mesh/ch10032_body_capsules.mlcap")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (GNN_POC / "real_scene" / "formats.py").is_file():
        fail(f"the sibling GNN PoC supplies the container writer but was not found at {GNN_POC}")
    sys.path.insert(0, str(GNN_POC))
    from real_scene.formats import Section, write_sectioned  # noqa: E402

    if not args.physics_t3d.is_file():
        fail(f"physics asset export is missing: {args.physics_t3d}")
    if not args.model.is_file():
        fail(f"encoded model is missing: {args.model}. Run prepare_runtime.ps1 first.")

    driver_names = read_driver_names(args.model)
    driver_index = {name: index for index, name in enumerate(driver_names)}
    capsules, skipped = parse_t3d(args.physics_t3d)

    # A capsule on a bone the model does not drive cannot be placed, because the clip
    # only carries the 45 driver transforms. Refusing is right: quietly dropping it
    # would remove collision the solver is supposed to respect.
    orphans = sorted({capsule["bone"] for capsule in capsules if capsule["bone"] not in driver_index})
    if orphans:
        fail(
            f"these physics bones are not among the model's {DRIVER_COUNT} drivers, so their capsules "
            f"cannot be placed from a .mldrv clip: {orphans}"
        )

    for capsule in capsules:
        capsule["driver_index"] = driver_index[capsule["bone"]]
        capsule["axis"] = rotator_axis(*capsule["rotation_pyr_deg"])
        capsule["half_length_cm"] = 0.5 * capsule["segment_length_cm"]
        capsule["total_height_cm"] = capsule["segment_length_cm"] + 2.0 * capsule["radius_cm"]
    capsules.sort(key=lambda capsule: (capsule["driver_index"], capsule["object"]))

    count = len(capsules)
    sections = [
        Section("info", 4, 4, struct.pack("<4I", count, DRIVER_COUNT, 0, 0)),
        Section("driver", count, 4, struct.pack(f"<{count}I", *(c["driver_index"] for c in capsules))),
        Section("center", count, 12, struct.pack(f"<{count * 3}f", *(v for c in capsules for v in c["center_cm"]))),
        Section("axis", count, 12, struct.pack(f"<{count * 3}f", *(v for c in capsules for v in c["axis"]))),
        Section(
            "size",
            count,
            8,
            struct.pack(f"<{count * 2}f", *(v for c in capsules for v in (c["radius_cm"], c["half_length_cm"]))),
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = write_sectioned(
        args.output.resolve(), MAGIC, VERSION, sections, source_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest()
    )

    report = {
        "output": str(args.output.resolve()),
        "physics_t3d": str(args.physics_t3d.resolve()),
        "physics_t3d_sha256": hashlib.sha256(args.physics_t3d.read_bytes()).hexdigest(),
        "model": str(args.model.resolve()),
        "capsules": count,
        "driver_count": DRIVER_COUNT,
        "skipped_non_capsule_geometry": skipped,
        "bones": [
            {
                "bone": capsule["bone"],
                "driver_index": capsule["driver_index"],
                "radius_cm": round(capsule["radius_cm"], 5),
                "segment_length_cm": round(capsule["segment_length_cm"], 5),
                "total_height_cm": round(capsule["total_height_cm"], 5),
                "axis": [round(component, 6) for component in capsule["axis"]],
                "center_cm": [round(component, 5) for component in capsule["center_cm"]],
            }
            for capsule in capsules
        ],
        "payload_sha256": written["payload_sha256"],
        "file_bytes": written["file_bytes"],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if skipped:
        print(f"\nWARNING: {len(skipped)} non-capsule collision primitives were skipped: {skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
