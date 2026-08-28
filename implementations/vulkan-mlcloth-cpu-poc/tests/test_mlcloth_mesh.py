#!/usr/bin/env python3
"""Round-trip the MLCloth training export into `.mlmesh`, with no Unreal involved.

The fixtures here are synthesised, because at the time this was written no training
export existed on disk yet. That is the point: the converter and the container it
produces are pinned down *before* the real export arrives, so when it does the only
new variable is the data.

The validator below re-checks the produced file against the same rules the C++
`mlcloth_formats.cpp::parse_mesh` enforces -- section shapes and alignment, strictly
ascending edges, exact CSR totals, the pin count, the Euler characteristic. Two
independent implementations of the same contract is the whole reason a scrambled
asset would fail to load rather than load wrongly, so the test asserts the contract
rather than asserting "the writer wrote what the writer wrote".

Run directly: `py -3 tests/test_mlcloth_mesh.py`
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
CONVERTER = POC_ROOT / "tools" / "bake_cloth_topology.py"
CLOTH_VERTEX_COUNT = 5294
DRIVER_COUNT = 45
SECTION_HEADER = struct.Struct("<8sIIQQ32s32s")
SECTION_ENTRY = struct.Struct("<16sQII")

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _failures.append(message)
        print(f"  FAIL: {message}")


def check_equal(actual, expected, message: str) -> None:
    check(actual == expected, f"{message}: got {actual!r}, expected {expected!r}")


# ---------------------------------------------------------------------------
# Fixture synthesis
# ---------------------------------------------------------------------------


def make_cylinder(rows: int, columns: int) -> tuple[list[float], list[int]]:
    """`rows` rings of `columns` vertices, columns wrapping, in Root_M-local cm.

    5294 = 2 * 2647 with 2647 prime, so a two-ring cylinder is the only grid shape
    that hits the model's vertex count exactly. Ring 0 sits at Z = 0 and is the loop
    the pin rule must choose; ring 1 hangs below it, the way a skirt hangs off a
    waist in this space.
    """
    positions: list[float] = []
    for row in range(rows):
        for column in range(columns):
            angle = 2.0 * math.pi * column / columns
            positions.extend((20.0 * math.cos(angle), 20.0 * math.sin(angle), -20.0 * row))
    triangles: list[int] = []
    for row in range(rows - 1):
        for column in range(columns):
            nxt = (column + 1) % columns
            v00 = row * columns + column
            v01 = row * columns + nxt
            v10 = (row + 1) * columns + column
            v11 = (row + 1) * columns + nxt
            triangles.extend((v00, v10, v11))
            triangles.extend((v00, v11, v01))
    return positions, triangles


def write_model(path: Path) -> str:
    config = {
        "modelType": 2,
        "driverFeatureLen": 1969,
        "drivenFeatureLen": 16394,
        "pcaDim": 512,
        "driverNames": ["Root_M"] + [f"Driver_{i}" for i in range(1, 45)],
    }
    blob = json.dumps(config, separators=(",", ":")).encode("utf-8")
    data = struct.pack("<I", len(blob)) + blob + b"\x00\x01\x02\x03"
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_topology(path: Path, vertex_count: int, indices: list[int], asset_index: int = 0, lod: int = 0) -> None:
    payload = b"AICCTOP1" + struct.pack("<ii", 1, 1)
    payload += struct.pack("<iiiiI", asset_index, lod, vertex_count, len(indices), 0xDEADBEEF)
    payload += struct.pack(f"<{len(indices)}I", *indices)
    path.write_bytes(payload)


def write_sim(
    path: Path,
    positions: list[float],
    frames: int = 3,
    compact: bool = False,
    include_normals: bool = True,
    asset_index: int = 0,
    lod: int = 0,
) -> None:
    vertex_count = len(positions) // 3
    identity = struct.pack("<10f", 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)
    if compact:
        payload = b"AICCLTH2" + struct.pack("<iiiii", 2, 30, frames, 1, 1 if include_normals else 0)
    else:
        payload = b"AICCLTH1" + struct.pack("<iiii", 1, 30, frames, 1)
        include_normals = True
    payload += struct.pack("<ii", asset_index, vertex_count)
    for frame in range(frames):
        payload += struct.pack("<i", lod) + identity + identity
        # Later frames are displaced so a converter that silently read the wrong
        # frame would produce a different rest pose and fail the assertions below.
        offset = 5.0 * frame
        payload += struct.pack(
            f"<{len(positions)}f", *[value + (offset if index % 3 == 2 else 0.0) for index, value in enumerate(positions)]
        )
        if include_normals:
            payload += struct.pack(f"<{len(positions)}f", *([0.0] * len(positions)))
    path.write_bytes(payload)


def write_attachment_clip(
    directory: Path,
    positions: list[float],
    attached_vertices: set[int],
    frames: int = 8,
    bone_count: int = 45,
    swing_cm: float = 20.0,
) -> None:
    """A clip where `attached_vertices` ride bone 0 rigidly and everything else swings.

    The cloth is written in reference-bone-local space with an identity reference
    transform, so component space and local space coincide and bone 0 sits at the origin.
    Attached vertices therefore never move at all in bone 0's frame, while free vertices
    are displaced by a per-frame amount -- which is exactly the separation
    `measure_loop_attachment` is supposed to find. Making the two classes differ by
    construction is the point: a fixture where everything moved together could not tell a
    working measurement from one that returns a constant.
    """
    directory.mkdir(parents=True, exist_ok=True)
    vertex_count = len(positions) // 3
    identity = struct.pack("<10f", 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)

    sim = b"AICCLTH2" + struct.pack("<iiiii", 2, 30, frames, 1, 0)
    sim += struct.pack("<ii", 0, vertex_count)
    for frame in range(frames):
        sim += struct.pack("<i", 0) + identity + identity
        swing = swing_cm * math.sin(2.0 * math.pi * frame / frames)
        moved = list(positions)
        for v in range(vertex_count):
            if v not in attached_vertices:
                moved[3 * v + 1] += swing
        sim += struct.pack(f"<{len(moved)}f", *moved)
    (directory / "cloth_sim.bin").write_bytes(sim)

    # Every bone is a stationary identity, so no bone can "explain" the swinging
    # vertices and they must score as free.
    bones = b"AICBCMP2" + struct.pack("<iiiii", 2, 30, frames, bone_count, 1)
    bones += identity * (frames * bone_count)
    (directory / "bones_component.bin").write_bytes(bones)


def write_manifest(path: Path, convention: str | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "clothVertexIndexConvention": convention
                if convention is not None
                else "cloth_sim.bin positions use zero-based array order; "
                     "cloth_topology.bin indices address that same array order",
                "clothPositionSpace": "FClothSimulData::Positions as returned by UChaosClothComponent "
                                      "simulation proxy after Chaos writeback",
                "chaosClothAsset": "/Game/Test/CCA_Fixture",
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Container validator -- mirrors mlcloth_formats.cpp::parse_mesh
# ---------------------------------------------------------------------------


def read_mlmesh(path: Path, expected_model_sha256: str) -> dict:
    blob = path.read_bytes()
    magic, version, section_count, file_bytes, payload_offset, payload_sha, source_sha = SECTION_HEADER.unpack_from(blob)
    check_equal(magic, b"MLMSH001", "mesh magic")
    check_equal(version, 1, "mesh version")
    check_equal(file_bytes, len(blob), "mesh fileBytes")
    check_equal(source_sha.hex(), expected_model_sha256, "mesh source digest is the model digest")
    check_equal(hashlib.sha256(blob[payload_offset:]).digest(), payload_sha, "mesh payload digest")
    directory = SECTION_HEADER.size + SECTION_ENTRY.size * section_count
    check_equal(payload_offset, (directory + 15) // 16 * 16, "mesh payloadOffset follows the directory")

    sections: dict[str, tuple[int, int, int]] = {}
    for index in range(section_count):
        name, offset, count, stride = SECTION_ENTRY.unpack_from(blob, SECTION_HEADER.size + SECTION_ENTRY.size * index)
        label = name.rstrip(b"\x00").decode("ascii")
        check(label not in sections, f"duplicate section {label}")
        check(offset % 16 == 0, f"section {label} is 16-byte aligned")
        check(offset >= payload_offset and offset + count * stride <= len(blob), f"section {label} is in bounds")
        sections[label] = (offset, count, stride)

    def read(label: str, fmt: str, expected_stride: int) -> tuple:
        offset, count, stride = sections[label]
        check_equal(stride, expected_stride, f"section {label} stride")
        total = count * stride // 4
        return struct.unpack_from(f"<{total}{fmt}", blob, offset)

    info = read("info", "I", 4)
    vertices, triangles, edges, boundary_edges, boundary_loops, max_valence, pinned, reserved = info
    check_equal(reserved, 0, "reserved info word")
    check_equal(vertices, CLOTH_VERTEX_COUNT, "vertex count")

    positions = read("positions", "f", 12)
    triangle_indices = read("triangles", "I", 12)
    edge_pairs = read("edges", "I", 8)
    edge_offsets = read("edge_csr_offs", "I", 4)
    edge_neighbours = read("edge_csr_nbr", "I", 4)
    tri_offsets = read("tri_csr_offs", "I", 4)
    tri_indices = read("tri_csr_idx", "I", 4)
    mass = read("vertex_mass", "f", 4)
    pin_mask = read("pin_mask", "I", 4)

    check_equal(len(positions), vertices * 3, "positions length")
    check_equal(len(triangle_indices), triangles * 3, "triangles length")
    check_equal(len(edge_pairs), edges * 2, "edges length")
    check_equal(len(edge_offsets), vertices + 1, "edge CSR offsets length")
    check_equal(edge_offsets[0], 0, "edge CSR starts at zero")
    check_equal(edge_offsets[vertices], 2 * edges, "edge CSR total")
    check_equal(len(edge_neighbours), 2 * edges, "edge CSR neighbour length")
    check_equal(tri_offsets[vertices], 3 * triangles, "triangle CSR total")
    check_equal(len(tri_indices), 3 * triangles, "triangle CSR index length")
    check_equal(sum(1 for flag in pin_mask if flag), pinned, "pin_mask agrees with the pinned count")
    check(all(flag in (0, 1) for flag in pin_mask), "pin_mask is boolean")
    check(0 < pinned < vertices, "some but not all vertices are pinned")
    check(all(value > 0.0 for value in mass), "every vertex mass is positive")
    check(all(math.isfinite(value) for value in positions), "every position is finite")

    ascending = all(
        (edge_pairs[2 * i], edge_pairs[2 * i + 1]) < (edge_pairs[2 * i + 2], edge_pairs[2 * i + 3])
        for i in range(edges - 1)
    )
    check(ascending, "edges are strictly ascending")
    check(all(edge_pairs[2 * i] < edge_pairs[2 * i + 1] for i in range(edges)), "edges are ordered pairs")

    # The coverage rule: the edge list is exactly the triangle edge set, and every
    # edge carries one or two triangles.
    lookup = {(edge_pairs[2 * i], edge_pairs[2 * i + 1]): i for i in range(edges)}
    use = [0] * edges
    for t in range(triangles):
        corners = triangle_indices[3 * t : 3 * t + 3]
        for k in range(3):
            a, b = corners[k], corners[(k + 1) % 3]
            key = (min(a, b), max(a, b))
            check(key in lookup, f"triangle {t} edge {key} is in the edge list")
            if key in lookup:
                use[lookup[key]] += 1
    check(all(1 <= count <= 2 for count in use), "every edge carries one or two triangles")
    check_equal(sum(1 for count in use if count == 1), boundary_edges, "boundary edge count")
    check_equal(vertices - edges + triangles, 0, "Euler characteristic of an annulus")
    check_equal(max(tri_offsets[v + 1] - tri_offsets[v] for v in range(vertices)), max_valence, "max triangle valence")

    # The pin bind is optional, so it is read only when present -- a bake with no reference
    # clip has nothing to measure it from and writes neither section.
    pin_driver = read("pin_driver", "I", 4) if "pin_driver" in sections else None
    pin_local_cm = read("pin_local_cm", "f", 12) if "pin_local_cm" in sections else None
    check_equal(pin_driver is None, pin_local_cm is None, "the two pin bind sections travel together")
    if pin_driver is not None:
        check_equal(len(pin_driver), vertices, "pin_driver length")
        check_equal(len(pin_local_cm), vertices * 3, "pin_local_cm length")
        bound = [v for v in range(vertices) if pin_driver[v] != 0xFFFFFFFF]
        check_equal(len(bound), pinned, "every pinned vertex and only those carry a bind")
        check(all(pin_mask[v] == 1 for v in bound), "no free vertex carries a driver")
        check(all(pin_driver[v] < DRIVER_COUNT for v in bound), "every bind names a model driver")
        check(all(math.isfinite(value) for value in pin_local_cm), "every bind offset is finite")

    return {
        "vertices": vertices,
        "triangles": triangles,
        "edges": edges,
        "boundary_edges": boundary_edges,
        "boundary_loops": boundary_loops,
        "max_valence": max_valence,
        "pinned": pinned,
        "pin_mask": pin_mask,
        "mass": mass,
        "positions": positions,
        "pin_driver": pin_driver,
        "pin_local_cm": pin_local_cm,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def run_converter(work: Path, *extra: str, expect_failure: bool = False) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(CONVERTER),
        "--topology", str(work / "cloth_topology.bin"),
        "--sim", str(work / "cloth_sim.bin"),
        "--model", str(work / "model.enc"),
        "--manifest", str(work / "manifest.json"),
        "--output", str(work / "out.mlmesh"),
        "--report", str(work / "report.json"),
        # Explicit, because the converter's own default is -1 (the settled last frame of a
        # T-pose reference). These fixtures displace later frames on purpose, so the tests
        # below name the frame they mean rather than inheriting a default that could
        # change under them.
        "--rest-frame", "0",
        *extra,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if expect_failure:
        check(result.returncode != 0, f"converter should have refused: {' '.join(extra) or '(defaults)'}")
    elif result.returncode != 0:
        check(False, f"converter failed: {result.stderr.strip()[:400]}")
    return result


def cross_check_with_cpp(work: Path) -> None:
    """Hand the produced file to the real C++ parser, when the build provides it.

    This is the assertion that matters most in this file: everything else checks the
    container against a Python re-derivation, which shares this converter's idea of
    what the topology is. `mesh_validate` does not -- it re-derives the edge set and
    the boundary structure in C++ and refuses the file if the two disagree. Skipped
    with a printed note rather than silently when the binary is absent, so a run that
    did not actually cross-check never reads as if it had.
    """
    validator = os.environ.get("MLCLOTH_MESH_VALIDATE")
    if not validator or not Path(validator).is_file():
        print("    (skipped C++ cross-check: set MLCLOTH_MESH_VALIDATE to the mesh_validate binary)")
        return
    result = subprocess.run(
        [validator, str(work / "model.enc"), str(work / "out.mlmesh")],
        capture_output=True,
        text=True,
    )
    check(
        result.returncode == 0,
        f"C++ parse_mesh rejected the Python-written mesh: {result.stderr.strip()[:400]}",
    )
    if result.returncode == 0:
        print(f"    C++ cross-check: {result.stdout.strip()}")


def build_fixture(work: Path, rows: int = 2, columns: int = CLOTH_VERTEX_COUNT // 2, **sim: object) -> tuple[str, list[int]]:
    positions, triangles = make_cylinder(rows, columns)
    model_sha = write_model(work / "model.enc")
    write_topology(work / "cloth_topology.bin", len(positions) // 3, triangles)
    write_sim(work / "cloth_sim.bin", positions, **sim)  # type: ignore[arg-type]
    write_manifest(work / "manifest.json")
    return model_sha, triangles


def test_full_conversion(work: Path) -> None:
    columns = CLOTH_VERTEX_COUNT // 2
    model_sha, _ = build_fixture(work)
    run_converter(work)
    cross_check_with_cpp(work)
    mesh = read_mlmesh(work / "out.mlmesh", model_sha)
    check_equal(mesh["triangles"], 2 * columns, "triangle count")
    check_equal(mesh["edges"], 4 * columns, "edge count")
    check_equal(mesh["boundary_edges"], 2 * columns, "boundary edge count")
    check_equal(mesh["boundary_loops"], 2, "boundary loop count")
    check_equal(mesh["max_valence"], 3, "max triangle valence")
    check_equal(mesh["pinned"], columns, "pinned vertex count")
    # The pin rule must pick the ring at Z = 0, not the one hanging below it.
    check(all(mesh["pin_mask"][v] == 1 for v in range(columns)), "the top ring is pinned")
    check(all(mesh["pin_mask"][v] == 0 for v in range(columns, 2 * columns)), "the lower ring is free")
    # Rest frame 0 must be the one that was read: later frames are offset in Z.
    check(abs(mesh["positions"][2]) < 1.0e-4, "rest frame 0 was used, not a displaced later frame")

    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    check_equal(report["euler_characteristic"], 0, "report Euler characteristic")
    check_equal(report["boundary_loops"], 2, "report boundary loops")
    check_equal(report["model_sha256"], model_sha, "report model digest")
    check_equal(report["sim_frames"], 3, "report sim frame count")
    check_equal(report["topology_hash_ue_crc32"], 0xDEADBEEF, "report carries UE's topology CRC")
    # Cylinder of radius 20 cm and height 20 cm: lateral area 2*pi*r*h = 0.2513 m^2.
    expected_area = 2.0 * math.pi * 0.20 * 0.20
    check(
        abs(report["surface_area_m2"] - expected_area) < 0.02 * expected_area,
        f"surface area {report['surface_area_m2']} is within 2% of {expected_area:.4f} m^2",
    )
    check(
        abs(report["total_mass_kg"] - report["surface_area_m2"] * report["density_kg_m2"]) < 1.0e-6,
        "total mass is area times density",
    )
    # Loop summary must let a reader confirm the pin choice rather than trust it.
    ranks = sorted(entry["rank_by_mean"] for entry in report["boundary_loop_summary"])
    check_equal(ranks, [0, 1], "every boundary loop is ranked")
    pinned_loops = [entry for entry in report["boundary_loop_summary"] if entry["pinned"]]
    check_equal(len(pinned_loops), 1, "exactly one loop is pinned")
    check(pinned_loops[0]["mean_cm"] > -1.0, "the pinned loop is the higher one")


def test_up_axis_is_derived_not_configured(work: Path) -> None:
    """The up axis comes from the reference transform, and a wrong assertion is refused.

    The fixture's reference transform is identity, so up is local Z. Asserting X or Y has
    to fail: on the real garment the reference bone maps local X to world up, and the
    plausible-looking `--up-axis z` produced a well-formed mesh with a meaningless pin
    set and no diagnostic at all. This test exists because that failure was silent.
    """
    build_fixture(work)
    run_converter(work, "--up-axis", "z")
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    check_equal(report["up_axis"], "z", "derived up axis")
    check(
        abs(report["up_axis_alignment_with_component_z"] - 1.0) < 1.0e-6,
        f"identity reference transform aligns perfectly, got {report['up_axis_alignment_with_component_z']}",
    )
    for wrong in ("x", "y"):
        result = run_converter(work, "--up-axis", wrong, expect_failure=True)
        check(
            "contradicts the reference transform" in result.stderr,
            f"--up-axis {wrong} is refused for the stated reason, got: {result.stderr.strip()[:200]}",
        )


def test_attachment_measurement_overrides_height(work: Path) -> None:
    """Pins come from measured attachment, and the measurement beats the height rule.

    The clip is built so the *lower* ring rides bone 0 and the upper ring swings -- the
    opposite of what height would say. A converter that quietly kept ranking by height
    would still emit a valid mesh here, so the assertion is on which ring ends up pinned.
    """
    columns = CLOTH_VERTEX_COUNT // 2
    model_sha, _ = build_fixture(work)
    lower_ring = set(range(columns, 2 * columns))
    write_attachment_clip(work / "clip_a", make_cylinder(2, columns)[0], lower_ring)
    run_converter(work, "--attachment-clip", str(work / "clip_a"))
    cross_check_with_cpp(work)

    mesh = read_mlmesh(work / "out.mlmesh", model_sha)
    check_equal(mesh["pinned"], columns, "one ring is pinned")
    check(all(mesh["pin_mask"][v] == 1 for v in lower_ring), "the measured-attached lower ring is pinned")
    check(all(mesh["pin_mask"][v] == 0 for v in range(columns)), "the swinging upper ring is free")

    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    check(report["pin_rule_warning"] is None, "no fallback warning when clips were supplied")
    check("normalised residual" in report["pin_rule"], f"pin rule names the measurement: {report['pin_rule']}")
    pinned = [entry for entry in report["boundary_loop_summary"] if entry["pinned"]]
    free = [entry for entry in report["boundary_loop_summary"] if not entry["pinned"]]
    check_equal(len(pinned), 1, "exactly one loop measured as attached")
    check(
        pinned[0]["attachment_residual"] < free[0]["attachment_residual"],
        "the pinned loop has the lower residual",
    )
    # The disagreement with the height rule must be visible, not just resolved.
    check(
        pinned[0]["pinned_by_height_rule"] is False,
        "the report records that the height rule would have chosen differently",
    )

    # The same measurement also produces the rigid bind the network-free comparison branch
    # drives its pins with. Every bone in this fixture is a stationary identity and the
    # attached ring never moves in bone 0's frame, so the bind must be bone 0 at the
    # vertex's own rest position -- anything else means the offset was taken in the wrong
    # frame, which is the failure that would place the anchor somewhere plausible but wrong.
    check(mesh["pin_driver"] is not None, "the bake wrote the pin bind sections")
    check(all(mesh["pin_driver"][v] == 0 for v in lower_ring), "the bind names bone 0")
    worst = max(
        abs(mesh["pin_local_cm"][3 * v + axis] - mesh["positions"][3 * v + axis])
        for v in lower_ring
        for axis in range(3)
    )
    check(worst < 1.0e-3, f"the bind offset reproduces the rest position (worst {worst:.2e} cm)")
    check_equal(report["pin_bind"]["written"], True, "the report records that the bind was written")
    check_equal(len(report["pin_bind"]["loops"]), 1, "one pinned loop is bound")
    check_equal(report["pin_bind"]["loops"][0]["bone_index"], 0, "the report names bone 0")


def test_missing_attachment_clips_warn(work: Path) -> None:
    """Falling back to the height rule is allowed but must announce itself."""
    build_fixture(work)
    run_converter(work)
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    check(report["pin_rule_warning"] is not None, "the height-rule fallback carries a warning")
    check("attachment-clip" in report["pin_rule_warning"], "the warning says what to pass instead")
    check_equal(report["attachment_clips"], [], "no attachment clips recorded")


def test_undecidable_attachment_band_is_refused(work: Path) -> None:
    """A loop between the two thresholds is an error rather than a coin flip."""
    columns = CLOTH_VERTEX_COUNT // 2
    build_fixture(work)
    write_attachment_clip(work / "clip_a", make_cylinder(2, columns)[0], set(range(columns, 2 * columns)))
    result = run_converter(
        work,
        "--attachment-clip", str(work / "clip_a"),
        "--attached-below", "0.0",
        "--free-above", "2.0",
        expect_failure=True,
    )
    check(
        "undecidable band" in result.stderr,
        f"the ambiguity is reported as such, got: {result.stderr.strip()[:200]}",
    )


def test_unattached_component_is_refused(work: Path) -> None:
    """If measurement attaches nothing, the bake fails instead of shipping loose cloth."""
    columns = CLOTH_VERTEX_COUNT // 2
    build_fixture(work)
    # Nothing rides the bone, so every loop swings and none can be pinned.
    write_attachment_clip(work / "clip_a", make_cylinder(2, columns)[0], set())
    result = run_converter(
        work,
        "--attachment-clip", str(work / "clip_a"),
        "--attached-below", "1.0e-6",
        "--free-above", "1.0e-5",
        expect_failure=True,
    )
    check(
        "unconstrained" in result.stderr or "undecidable band" in result.stderr,
        f"an unpinnable component is refused, got: {result.stderr.strip()[:200]}",
    )


def test_compact_sim_variant(work: Path) -> None:
    for include_normals in (True, False):
        model_sha, _ = build_fixture(work, compact=True, include_normals=include_normals)
        run_converter(work)
        mesh = read_mlmesh(work / "out.mlmesh", model_sha)
        check_equal(mesh["vertices"], CLOTH_VERTEX_COUNT, f"AICCLTH2 normals={include_normals} vertex count")


def test_rest_frame_selection(work: Path) -> None:
    model_sha, _ = build_fixture(work)
    run_converter(work, "--rest-frame", "2")
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    check_equal(report["rest_frame"], 2, "requested rest frame")
    run_converter(work, "--rest-frame", "3", expect_failure=True)

    # -1 is the last frame, which for a settle clip is the rested garment. The fixture
    # displaces frame n by 5n in Z, so the resolved frame is checkable rather than
    # merely reported.
    run_converter(work, "--rest-frame", "-1")
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    check_equal(report["rest_frame"], 2, "-1 resolves to the last frame, and is reported resolved")
    mesh = read_mlmesh(work / "out.mlmesh", model_sha)
    check(
        abs(mesh["positions"][2] - 10.0) < 1.0e-4,
        f"the last frame's 10 cm displacement is present, got {mesh['positions'][2]}",
    )
    run_converter(work, "--rest-frame", "-4", expect_failure=True)


def test_vertex_count_mismatch(work: Path) -> None:
    # A topology for a different garment parses cleanly and would be silently wrong,
    # so the count check is the thing standing between that and a scrambled asset.
    positions, triangles = make_cylinder(2, 100)
    write_model(work / "model.enc")
    write_topology(work / "cloth_topology.bin", len(positions) // 3, triangles)
    write_sim(work / "cloth_sim.bin", positions)
    write_manifest(work / "manifest.json")
    result = run_converter(work, expect_failure=True)
    check("expected" in result.stderr or "declares" in result.stderr, "mismatch is reported clearly")


def test_non_manifold_is_refused(work: Path) -> None:
    positions, triangles = make_cylinder(2, CLOTH_VERTEX_COUNT // 2)
    write_model(work / "model.enc")
    triangles = triangles + [triangles[0], triangles[1], CLOTH_VERTEX_COUNT // 2 + 7]
    write_topology(work / "cloth_topology.bin", len(positions) // 3, triangles)
    write_sim(work / "cloth_sim.bin", positions)
    write_manifest(work / "manifest.json")
    result = run_converter(work, expect_failure=True)
    check("manifold" in result.stderr, "non-manifold is named as the reason")


def test_manifest_convention_is_enforced(work: Path) -> None:
    build_fixture(work)
    write_manifest(work / "manifest.json", convention="positions are in some other order entirely")
    result = run_converter(work, expect_failure=True)
    check("array order" in result.stderr, "the convention guarantee is named as the reason")


def test_truncated_inputs(work: Path) -> None:
    build_fixture(work)
    topology = work / "cloth_topology.bin"
    topology.write_bytes(topology.read_bytes()[:-16])
    result = run_converter(work, expect_failure=True)
    check("truncated" in result.stderr, "a truncated topology is named as the reason")

    build_fixture(work)
    sim = work / "cloth_sim.bin"
    sim.write_bytes(sim.read_bytes() + b"\x00" * 8)
    result = run_converter(work, expect_failure=True)
    check("trailing" in result.stderr, "trailing bytes in cloth_sim.bin are named as the reason")


def main() -> int:
    tests = [
        test_full_conversion,
        test_up_axis_is_derived_not_configured,
        test_attachment_measurement_overrides_height,
        test_missing_attachment_clips_warn,
        test_undecidable_attachment_band_is_refused,
        test_unattached_component_is_refused,
        test_compact_sim_variant,
        test_rest_frame_selection,
        test_vertex_count_mismatch,
        test_non_manifold_is_refused,
        test_manifest_convention_is_enforced,
        test_truncated_inputs,
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
