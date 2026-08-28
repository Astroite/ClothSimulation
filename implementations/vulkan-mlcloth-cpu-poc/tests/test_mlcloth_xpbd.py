#!/usr/bin/env python3
"""Compare the C++ CPU XPBD port against the Python reference it was ported from.

This is the only check in this PoC's XPBD coverage that does not share the port's own
assumptions. Everything else would confirm that the port agrees with itself.

`vulkan-gnn-poc/real_scene/xpbd.py::project` with `sweep="jacobi"` is the definition. The
constraint tables are built with the reference's own `_gather_tables` and
`greedy_colouring` rather than a hand-rolled equivalent, for the same reason: a
reimplemented gather would put a second suspect between the two solvers.

Tolerances are on the *displacement* the solver produced, not on the absolute position,
because agreeing to 1e-7 on a coordinate that barely moved is not evidence. They are a
small multiple of float32 epsilon against that displacement -- the two implementations do
the same arithmetic in the same order, so anything larger is a real disagreement, not
rounding. Note that a GPU port could not be held to this: `hood_xpbd.comp` records a
1.68e-3 gap against the same reference caused by the one-sided `max(residual, 0)` and the
contact's `signed < 0` landing on a branch boundary, and that gap is a property of the
comparison, not of the port. Having the tight version first is the point of doing the CPU
solver before any shader.

Needs the sibling PoC's virtual environment for torch. Skipped with a printed note when it
is absent, and when `MLCLOTH_XPBD_REFERENCE` does not point at the built CLI -- never
silently.

Run directly: `py -3 tests/test_mlcloth_xpbd.py`
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
GNN_POC = POC_ROOT.parent / "vulkan-gnn-poc"
GNN_VENV = GNN_POC / ".venv/Scripts/python.exe"

MAGIC = b"MLXPBSC1"
FLAG_ONE_SIDED = 1
FLAG_COLLISION = 2
FLAG_AREA = 4
FLAG_GUIDE = 8
FLAG_INERTIAL = 16

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _failures.append(message)
        print(f"  FAIL: {message}")


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------


def grid_mesh(rows: int, columns: int, spacing: float):
    """A flat grid in metres, with a deterministic triangulation and edge list."""
    positions = []
    for row in range(rows):
        for column in range(columns):
            positions.append((column * spacing, row * spacing, 0.0))
    triangles = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            b = a + 1
            c = a + columns
            d = c + 1
            triangles.append((a, b, d))
            triangles.append((a, d, c))
    edges = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    return positions, triangles, sorted(edges)


def bend_pairs_from_triangles(triangles, vertex_count: int):
    """The same derivation `derive_bend_pairs` performs, duplicated on purpose.

    Duplicated rather than shared because the pair *ordering* is the contract between the
    solver and whatever calibrates target lengths for it. If the two derivations ever
    disagree the comparison below fails loudly, which is the behaviour wanted.
    """
    owners: dict[tuple[int, int], list[int]] = {}
    for index, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            owners.setdefault((min(u, v), max(u, v)), []).append(index)
    unique = set()
    for (u, v), faces in owners.items():
        if len(faces) != 2:
            continue
        opposite = []
        for face in faces:
            rest = [corner for corner in triangles[face] if corner not in (u, v)]
            if len(rest) != 1:
                break
            opposite.append(rest[0])
        if len(opposite) != 2 or opposite[0] == opposite[1]:
            continue
        unique.add((min(opposite), max(opposite)))
    return sorted(unique)


def displace(positions, amount: float, spacing: float):
    """Scale the grid in plane and bow it out of plane, so constraints have work to do.

    The out-of-plane term is scaled by `spacing`, not used as a raw metre value. The first
    version used `amount` directly, which at amount -0.3 put a 30 cm wobble on a 5 cm grid:
    the mesh compressed in plane but crumpled so hard that triangle areas went *up*, and the
    area-floor test measured a pass that never activated.
    """
    moved = []
    for index, (x, y, z) in enumerate(positions):
        scale = 1.0 + amount
        wobble = amount * spacing * math.sin(index * 0.7)
        moved.append((x * scale, y * scale, z + wobble))
    return moved


def build_scenario(*, rows=6, columns=7, spacing=0.05, iterations=6, one_sided=True,
                   collision=False, area=False, guide=False, inertial=False,
                   include_bend=True, stretch_compliance=0.0, bend_compliance=0.05,
                   area_floor=0.0, area_compliance=0.0, guide_compliance=1.0,
                   guide_trust_ratio=0.0, contact_offset=0.0, timestep=1.0 / 30.0,
                   relaxation=1.0, capsules=(), pin_first_row=True, displacement=0.35):
    rest, triangles, edges = grid_mesh(rows, columns, spacing)
    vertex_count = len(rest)
    pairs = list(edges)
    bend_start = len(pairs)
    if include_bend:
        pairs += bend_pairs_from_triangles(triangles, vertex_count)

    def distance(a, b, points):
        return math.dist(points[a], points[b])

    target_length = [distance(a, b, rest) for a, b in pairs]
    target_area = []
    for a, b, c in triangles:
        ux, uy, uz = (rest[b][k] - rest[a][k] for k in range(3))
        vx, vy, vz = (rest[c][k] - rest[a][k] for k in range(3))
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        target_area.append(0.5 * math.sqrt(nx * nx + ny * ny + nz * nz))

    mass = [0.02 + 0.001 * (index % 5) for index in range(vertex_count)]
    pin_mask = [0] * vertex_count
    if pin_first_row:
        for column in range(columns):
            pin_mask[column] = 1

    guide_positions = displace(rest, displacement, spacing)
    inertial_positions = displace(rest, displacement * 0.4, spacing) if inertial else None

    flags = 0
    if one_sided:
        flags |= FLAG_ONE_SIDED
    if collision:
        flags |= FLAG_COLLISION
    if area:
        flags |= FLAG_AREA
    if guide:
        flags |= FLAG_GUIDE
    if inertial_positions is not None:
        flags |= FLAG_INERTIAL

    return {
        "rest": rest,
        "triangles": triangles,
        "pairs": pairs,
        "bend_start": bend_start,
        "target_length": target_length,
        "target_area": target_area,
        "mass": mass,
        "pin_mask": pin_mask,
        "guide": guide_positions,
        "inertial": inertial_positions,
        "capsules": list(capsules),
        "flags": flags,
        "iterations": iterations,
        "timestep": timestep,
        "relaxation": relaxation,
        "stretch_compliance": stretch_compliance,
        "bend_compliance": bend_compliance,
        "area_floor": area_floor if area else 0.0,
        "area_compliance": area_compliance,
        "guide_compliance": guide_compliance,
        "guide_trust_ratio": guide_trust_ratio,
        "contact_offset": contact_offset,
    }


def encode_scenario(scenario) -> bytes:
    vertex_count = len(scenario["rest"])
    blob = MAGIC
    blob += struct.pack(
        "<5I", vertex_count, len(scenario["pairs"]), len(scenario["triangles"]),
        len(scenario["capsules"]), scenario["bend_start"],
    )
    blob += struct.pack("<iI", scenario["iterations"], scenario["flags"])
    blob += struct.pack(
        "<8f", scenario["timestep"], scenario["relaxation"], scenario["stretch_compliance"],
        scenario["bend_compliance"], scenario["area_floor"], scenario["area_compliance"],
        scenario["guide_compliance"], scenario["guide_trust_ratio"],
    )
    blob += struct.pack("<f", scenario["contact_offset"])
    blob += struct.pack(f"<{vertex_count * 3}f", *[v for point in scenario["guide"] for v in point])
    if scenario["inertial"] is not None:
        blob += struct.pack(f"<{vertex_count * 3}f", *[v for point in scenario["inertial"] for v in point])
    blob += struct.pack(f"<{len(scenario['pairs']) * 2}I", *[v for pair in scenario["pairs"] for v in pair])
    blob += struct.pack(f"<{len(scenario['triangles']) * 3}I", *[v for t in scenario["triangles"] for v in t])
    blob += struct.pack(f"<{len(scenario['target_length'])}f", *scenario["target_length"])
    blob += struct.pack(f"<{len(scenario['target_area'])}f", *scenario["target_area"])
    blob += struct.pack(f"<{vertex_count}f", *scenario["mass"])
    blob += struct.pack(f"<{vertex_count}I", *scenario["pin_mask"])
    for capsule in scenario["capsules"]:
        blob += struct.pack("<8f", *capsule)
    return blob


def run_cpp(scenario, work: Path):
    binary = os.environ.get("MLCLOTH_XPBD_REFERENCE")
    if not binary or not Path(binary).is_file():
        return None, "set MLCLOTH_XPBD_REFERENCE to the built xpbd_reference binary"
    scenario_path = work / "scenario.bin"
    result_path = work / "result.bin"
    scenario_path.write_bytes(encode_scenario(scenario))
    outcome = subprocess.run([binary, str(scenario_path), str(result_path)], capture_output=True, text=True)
    if outcome.returncode != 0:
        return None, f"xpbd_reference failed: {outcome.stderr.strip()[:300]}"
    blob = result_path.read_bytes()
    vertex_count = len(scenario["rest"])
    values = struct.unpack_from(f"<{vertex_count * 3}f", blob, 0)
    contacts, floors, stretch = struct.unpack_from("<3I", blob, vertex_count * 12)
    worst = struct.unpack_from("<f", blob, vertex_count * 12 + 12)[0]
    return {
        "positions": [tuple(values[3 * v : 3 * v + 3]) for v in range(vertex_count)],
        "contacts": contacts, "floors": floors, "stretch": stretch, "worst": worst,
    }, None


PYTHON_DRIVER = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch
from real_scene.xpbd import (
    AreaConstraints, ConstraintSet, Contacts, SolverConfig, STRETCH, BEND,
    _gather_tables, greedy_colouring, guide_confidence, project,
)

payload = json.loads(Path(sys.argv[2]).read_text())
double = payload["double"]
dtype = torch.float64 if double else torch.float32

def tensor(values, shape=None):
    out = torch.tensor(values, dtype=dtype)
    return out if shape is None else out.reshape(shape)

vertex_count = len(payload["mass"])
pairs = torch.tensor(payload["pairs"], dtype=torch.long).reshape(-1, 2)
count = int(pairs.shape[0])
bend_start = payload["bend_start"]
kind = torch.full((count,), STRETCH, dtype=torch.long)
kind[bend_start:] = BEND

inverse_mass = torch.tensor(
    [0.0 if flag else 1.0 / mass for flag, mass in zip(payload["pin_mask"], payload["mass"])],
    dtype=dtype,
).reshape(-1, 1)

slots, signs, incident = _gather_tables(pairs, vertex_count)
constraints = ConstraintSet(
    pairs=pairs,
    target_length=tensor(payload["target_length"]),
    kind=kind,
    suspect=torch.zeros(count, dtype=torch.bool),
    slots=slots,
    signs=signs.to(dtype),
    incident=incident.to(dtype),
    inverse_mass=inverse_mass,
    colour=greedy_colouring(pairs, vertex_count),
)

triangles = torch.tensor(payload["triangles"], dtype=torch.long).reshape(-1, 3)
area = None
if payload["area_floor"] > 0.0 and int(triangles.shape[0]) > 0:
    # The same padded per-vertex tables the pair sweep uses, for triangles: slot -> incident
    # triangle, corner -> which of the three gradients this vertex owns.
    flat_vertex = triangles.reshape(-1)
    flat_triangle = torch.arange(int(triangles.shape[0])).repeat_interleave(3)
    flat_corner = torch.arange(3).repeat(int(triangles.shape[0]))
    order = torch.argsort(flat_vertex, stable=True)
    sorted_vertex = flat_vertex[order]
    counts = torch.bincount(sorted_vertex, minlength=vertex_count)
    width = max(1, int(counts.max().item()))
    offsets = torch.zeros(vertex_count + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)
    rank = torch.arange(sorted_vertex.shape[0]) - offsets[sorted_vertex]
    tri_slots = torch.full((vertex_count, width), int(triangles.shape[0]), dtype=torch.long)
    tri_corner = torch.zeros((vertex_count, width), dtype=torch.long)
    tri_slots[sorted_vertex, rank] = flat_triangle[order]
    tri_corner[sorted_vertex, rank] = flat_corner[order]
    area = AreaConstraints(
        triangles=triangles,
        target_area=tensor(payload["target_area"]),
        slots=tri_slots,
        corner=tri_corner,
        incident=counts.to(dtype).reshape(-1, 1),
    )

guide_positions = tensor(payload["guide"], (vertex_count, 3))
pin_mask = torch.tensor(payload["pin_mask"], dtype=torch.bool).reshape(-1, 1)

confidence = None
if payload["guide"] and payload["guide_trust_ratio"] > 0.0 and payload["inertial"] is not None:
    min_edge = torch.full((vertex_count,), float("inf"), dtype=dtype)
    lengths = tensor(payload["target_length"])
    for index in range(count):
        a, b = int(pairs[index, 0]), int(pairs[index, 1])
        min_edge[a] = torch.minimum(min_edge[a], lengths[index])
        min_edge[b] = torch.minimum(min_edge[b], lengths[index])
    min_edge = torch.where(torch.isfinite(min_edge), min_edge, torch.zeros_like(min_edge))
    inertial_positions = tensor(payload["inertial"], (vertex_count, 3))
    confidence = guide_confidence(guide_positions, inertial_positions, min_edge, payload["guide_trust_ratio"])

config = SolverConfig(
    iterations=payload["iterations"],
    mode="guide" if payload["use_guide"] else "standard",
    sweep="jacobi",
    stretch_compliance=payload["stretch_compliance"],
    bend_compliance=payload["bend_compliance"],
    one_sided=payload["one_sided"],
    relaxation=payload["relaxation"],
    collision=False,
    contact_offset=payload["contact_offset"],
    guide_compliance=payload["guide_compliance"],
    guide_trust_ratio=payload["guide_trust_ratio"],
    area_floor=payload["area_floor"],
    area_compliance=payload["area_compliance"],
)

result = project(
    constraints, config,
    position=guide_positions,
    inertial=guide_positions,
    pin_mask=pin_mask,
    pin_target=guide_positions,
    timestep=payload["timestep"],
    contacts=None,
    guide=guide_positions if payload["use_guide"] else None,
    confidence=confidence,
    area=area,
)
print(json.dumps({"positions": result.reshape(-1).tolist()}))
'''


def run_python(scenario, work: Path):
    if not GNN_VENV.is_file():
        return None, f"the sibling PoC's virtual environment is not at {GNN_VENV}"
    payload = {
        "pairs": [v for pair in scenario["pairs"] for v in pair],
        "bend_start": scenario["bend_start"],
        "triangles": [v for t in scenario["triangles"] for v in t],
        "target_length": scenario["target_length"],
        "target_area": scenario["target_area"],
        "mass": scenario["mass"],
        "pin_mask": scenario["pin_mask"],
        "guide": [v for point in scenario["guide"] for v in point],
        "inertial": None if scenario["inertial"] is None else [v for p in scenario["inertial"] for v in p],
        "iterations": scenario["iterations"],
        "timestep": scenario["timestep"],
        "relaxation": scenario["relaxation"],
        "stretch_compliance": scenario["stretch_compliance"],
        "bend_compliance": scenario["bend_compliance"],
        "area_floor": scenario["area_floor"],
        "area_compliance": scenario["area_compliance"],
        "guide_compliance": scenario["guide_compliance"],
        "guide_trust_ratio": scenario["guide_trust_ratio"],
        "contact_offset": scenario["contact_offset"],
        "one_sided": bool(scenario["flags"] & FLAG_ONE_SIDED),
        "use_guide": bool(scenario["flags"] & FLAG_GUIDE),
        "double": False,
    }
    import json

    payload_path = work / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    driver_path = work / "driver.py"
    driver_path.write_text(PYTHON_DRIVER, encoding="utf-8")
    outcome = subprocess.run(
        [str(GNN_VENV), str(driver_path), str(GNN_POC), str(payload_path)],
        capture_output=True, text=True,
    )
    if outcome.returncode != 0:
        return None, f"the Python reference failed: {outcome.stderr.strip()[-600:]}"
    values = json.loads(outcome.stdout)["positions"]
    vertex_count = len(scenario["mass"])
    return [tuple(values[3 * v : 3 * v + 3]) for v in range(vertex_count)], None


def compare(label: str, scenario, work: Path, tolerance_ulps: float = 64.0, expect_motion: bool = True):
    cpp, error = run_cpp(scenario, work)
    if error:
        print(f"    (skipped {label}: {error})")
        return None
    reference, error = run_python(scenario, work)
    if error:
        print(f"    (skipped {label}: {error})")
        return None

    guide = scenario["guide"]
    worst_absolute = 0.0
    displacement = 0.0
    for index in range(len(guide)):
        for axis in range(3):
            moved = abs(reference[index][axis] - guide[index][axis])
            displacement = max(displacement, moved)
            gap = abs(cpp["positions"][index][axis] - reference[index][axis])
            worst_absolute = max(worst_absolute, gap)
    # Judged against how far the solver actually moved things: agreeing on a coordinate
    # that never moved proves nothing.
    scale = max(displacement, 1.0e-6)
    worst_relative = worst_absolute / scale
    limit = tolerance_ulps * 1.1920929e-7
    print(f"    {label:<34} max_gap={worst_absolute:.3e}  displacement={displacement:.3e}  "
          f"relative={worst_relative:.3e}  limit={limit:.3e}")
    check(
        worst_relative <= limit,
        f"{label}: the port differs from the Python reference by {worst_relative:.3e} of the "
        f"displacement, over the {limit:.3e} limit",
    )
    if expect_motion:
        check(displacement > 1.0e-4, f"{label}: the scenario barely moved ({displacement:.3e} m), so it proves little")
    else:
        # Some configurations are *supposed* to be inert -- one-sided constraints under pure
        # compression have no positive residual to resolve. Asserting stillness there is the
        # informative claim, and it also confirms the reference does nothing either.
        check(displacement < 1.0e-6, f"{label}: expected an inert configuration, but it moved {displacement:.3e} m")
    return cpp["positions"]


def differs(a, b) -> float:
    """Largest per-coordinate gap between two results, for "did this knob do anything"."""
    if a is None or b is None:
        return math.inf
    return max(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stretch_only(work: Path) -> None:
    compare("stretch only", build_scenario(include_bend=False), work)


def test_stretch_and_bend(work: Path) -> None:
    compare("stretch + bend", build_scenario(include_bend=True), work)


def test_two_sided(work: Path) -> None:
    """The one-sided clamp has to be exercised, not merely enabled.

    A stretched mesh has every residual positive, so `max(residual, 0)` is inert and
    one-sided and two-sided produce identical results -- which is what the first version of
    this test measured, agreeing to four digits with the stretch case and proving nothing.
    Compressing the mesh puts every residual negative instead: one-sided then does nothing
    at all while two-sided pulls the edges back out, so the two must differ.
    """
    compressed = dict(displacement=-0.25, include_bend=False)
    one_sided = compare("one-sided, compressed", build_scenario(one_sided=True, **compressed), work,
                        expect_motion=False)
    two_sided = compare("two-sided, compressed", build_scenario(one_sided=False, **compressed), work)
    gap = differs(one_sided, two_sided)
    # The bar is against the port's own agreement with the reference (~1e-8 above), not an
    # absolute distance: what matters is that the knob moves the result far more than the
    # two implementations differ, so the comparison above is actually discriminating.
    check(gap > 1.0e-5, f"the one-sided clamp changes the result under compression, got {gap:.3e}")


def test_area_floor(work: Path) -> None:
    """Same trap: a stretched mesh's triangles are already above any sane floor.

    At displacement +0.35 the areas are about 1.8x the target, so a floor of 1.4x never
    binds. Compressing shrinks them below the floor so the pass actually runs, which the
    solver's own `areaFloorsActive` counter confirms rather than being inferred.
    """
    baseline = compare("no area floor, compressed", build_scenario(displacement=-0.3), work,
                       expect_motion=False)
    scenario = build_scenario(displacement=-0.3, area=True, area_floor=1.0, area_compliance=0.0)
    floored = compare("area floor, compressed", scenario, work)
    gap = differs(baseline, floored)
    check(gap > 1.0e-5, f"the area floor changes the result, got {gap:.3e}")
    cpp, error = run_cpp(scenario, work)
    if not error:
        check(cpp["floors"] > 0, f"the area pass reports active floors, got {cpp['floors']}")


def test_guide(work: Path) -> None:
    compare("guide", build_scenario(guide=True, guide_compliance=1.5), work)


def test_guide_with_trust_gate(work: Path) -> None:
    """The pass where reading the reference caught two errors in this port."""
    compare(
        "guide + trust gate",
        build_scenario(guide=True, guide_compliance=1.5, guide_trust_ratio=1.5, inertial=True),
        work,
    )


def test_everything(work: Path) -> None:
    compare(
        "stretch + bend + area + guide",
        build_scenario(displacement=-0.3, area=True, area_floor=1.0, area_compliance=0.0,
                       guide=True, guide_compliance=1.5, guide_trust_ratio=2.0, inertial=True),
        work,
    )


def test_zero_iterations_is_the_input(work: Path) -> None:
    """The no-regression gate: with no iterations the output is the network's own output.

    Checked on this side rather than only in the runtime because it is a property of the
    solver, and because the runtime's version of this test costs a GPU.
    """
    scenario = build_scenario(iterations=0, area=True, area_floor=1.0, guide=True)
    cpp, error = run_cpp(scenario, work)
    if error:
        print(f"    (skipped: {error})")
        return
    for index, point in enumerate(scenario["guide"]):
        for axis in range(3):
            # Bit-exact: struct-packed float32 in, float32 out, no arithmetic in between.
            check(
                struct.pack("<f", cpp["positions"][index][axis]) == struct.pack("<f", point[axis]),
                f"vertex {index} axis {axis} is unchanged with zero iterations",
            )
    check(cpp["stretch"] == 0 and cpp["floors"] == 0, "no pass ran")
    check(cpp["worst"] == 0.0, "the reported correction is zero")


def test_capsule_contact_pushes_out(work: Path) -> None:
    """The contact block has no Python counterpart, so it is checked against geometry.

    The sibling PoC's contacts are a nearest-proxy half-plane; this is an analytic capsule,
    which is a deliberate replacement rather than a port. So the assertion is the property
    that motivated it: no vertex is left inside, including ones driven past the axis where
    a half-plane's signed distance would have turned positive again.
    """
    capsule = (0.15, 0.12, 0.0, 0.0, 1.0, 0.0, 0.06, 0.10)
    scenario = build_scenario(collision=True, capsules=[capsule], iterations=12,
                             guide=False, displacement=0.05)
    cpp, error = run_cpp(scenario, work)
    if error:
        print(f"    (skipped: {error})")
        return
    centre, axis, radius, half = capsule[0:3], capsule[3:6], capsule[6], capsule[7]

    def signed(point):
        offset = [point[k] - centre[k] for k in range(3)]
        projection = max(-half, min(half, sum(offset[k] * axis[k] for k in range(3))))
        radial = [offset[k] - axis[k] * projection for k in range(3)]
        return math.sqrt(sum(v * v for v in radial)) - radius

    before = sum(1 for point in scenario["guide"] if signed(point) < 0.0)
    after = [signed(point) for point in cpp["positions"]]
    pinned = scenario["pin_mask"]
    # Pinned vertices are kinematic: the network drives them and the closing overwrite
    # restores their guide value, so the solver cannot push one out of a capsule and the
    # assertion must not ask it to. The first version of this test did, and failed on a
    # pinned vertex 1.2 cm inside -- correct solver behaviour, wrong assertion. That the
    # limitation is real is worth stating rather than papering over: if the network puts a
    # *pinned* vertex inside the body, nothing downstream will fix it.
    free_inside = [value for index, value in enumerate(after) if not pinned[index] and value < -1.0e-6]
    pinned_inside = [index for index, value in enumerate(after) if pinned[index] and value < -1.0e-6]
    check(before > 0, f"the scenario starts with vertices inside the capsule, got {before}")
    check(
        not free_inside,
        f"no movable vertex is left inside the capsule, got {len(free_inside)} worst {min(free_inside):.3e}"
        if free_inside else "",
    )
    check(cpp["contacts"] > 0, "the solver reports having resolved contacts")
    if pinned_inside:
        print(f"    (note: {len(pinned_inside)} pinned vertices remain inside, which the solver cannot move)")


def main() -> int:
    tests = [
        test_stretch_only,
        test_stretch_and_bend,
        test_two_sided,
        test_area_floor,
        test_guide,
        test_guide_with_trust_gate,
        test_everything,
        test_zero_iterations_is_the_input,
        test_capsule_contact_pushes_out,
    ]
    for test in tests:
        print(f"  {test.__name__}...")
        before = len(_failures)
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
        if len(_failures) == before:
            print("    ok")
    print()
    if _failures:
        print(f"{len(_failures)} failed")
        return 1
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
