"""Run the equal-budget substep / soft-guide matrix and aggregate it into one report.

Why this is a driver rather than a flag on `tools/recovery_probe.py`
-------------------------------------------------------------------
Each cell has to be its own OS process. `--repeats` averages inside one process, and
`results/PROGRESS.md` section 5.2 measured why that is useless as a noise estimate: `index_add_`'s
atomic accumulation order happens to reproduce within a process, so the in-process spread on
`ch10032_sprint` is 0.0007 against a cross-process 0.1208 -- an understatement of 10-30x. A matrix
whose rows differ by a few percent cannot be read against a noise floor that is measured 100x too
small, so `--processes` here means real subprocesses.

What the rows are, and why row 0 exists
---------------------------------------
Every row spends the same 128 constraint sweeps per visual frame, split differently between
substeps and iterations. The network, where present, still runs once per frame.

Row 0 is the one the plan this came from was missing, and it is the most important cell. The
repository's gate G0 table shows the hybrid beating both ablations, but `results/PROGRESS.md`
section 3 already carries the correction: branch B's score is 81-86% `over` -- edges stretched and
never pulled back, a pure convergence failure -- and it was measured at one step by 128 Jacobi
sweeps with one untuned configuration. Small steps is precisely the fix for a convergence failure.
So substepping is not only a hybrid improvement; it is also the control that can remove the
justification for having a network at all. Running D without running row 0 would produce a better
hybrid and leave the open question exactly as open. Row 0 is also the cheapest cell -- no GNN.

Rows B1 and 0 differ only in the substep count, so their gap is what small steps buys pure XPBD.
Rows A and B differ only in how the network enters, so their gap is what the soft guide buys. Rows
B through E differ only in the substep split. Nothing is confounded with anything else.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]

# (label, branch, mode, substeps, iterations, description)
ROWS = [
    ("0_xpbd_4x32", "B", "guide", 4, 32, "pure XPBD, substepped -- the control that can kill the architecture"),
    ("B1_xpbd_1x128", "B", "standard", 1, 128, "pure XPBD as measured today, at equal sweeps"),
    ("A_hybrid_hard_1x128", "C", "standard", 1, 128, "today's hybrid: the network as a hard initial value"),
    ("B_hybrid_guide_1x128", "C", "guide", 1, 128, "soft guide, no substeps -- isolates the coupling change"),
    ("C_hybrid_guide_2x64", "C", "guide", 2, 64, "soft guide, 2 substeps"),
    ("D_hybrid_guide_4x32", "C", "guide", 4, 32, "soft guide, 4 substeps"),
    ("E_hybrid_guide_8x16", "C", "guide", 8, 16, "soft guide, 8 substeps"),
]

# `sprint_start` is where the hybrid is worst (a 10.4 overshoot against pure XPBD's 2.5, fully
# recovered by step 90). `guard_skirt` is where the bare network is best, so it is where a coupling
# change has the most to lose. `block_loop_skirt` is the single clip where the hybrid loses to the
# network today. Three clips chosen to disagree with each other rather than to average well.
DEFAULT_MOTIONS = ("sprint_start", "guard_skirt", "block_loop_skirt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motions", nargs="+", default=list(DEFAULT_MOTIONS))
    parser.add_argument("--rows", nargs="+", default=[row[0] for row in ROWS])
    parser.add_argument("--students", nargs="+",
                        default=["student32x12_r1.vhood", "student32x12_v3.vhood"],
                        help="cross-tested because gate G0 measured the closed-loop-searched r1 as "
                             "the WORST XPBD partner (0.2186 against v3's 0.1949): a weight picked "
                             "on its bare score is not the right weight for the hybrid.")
    parser.add_argument("--processes", type=int, default=3,
                        help="independent subprocesses per cell. Do not lower this below 2 -- see "
                             "the module docstring on why --repeats is not a substitute.")
    parser.add_argument("--first-process", type=int, default=0,
                        help="index of the first process, so a matrix can be extended incrementally "
                             "without rerunning the cells already in hand")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--guide-compliance", type=float, default=10.0,
                        help="m^2/N. Measured on sprint_start step 1, the guide closes roughly half "
                             "the gap to the prediction here; below ~0.5 it is a hard guide in all "
                             "but name and above ~100 it is barely present.")
    parser.add_argument("--guide-trust-ratio", type=float, default=0.0)
    parser.add_argument("--area-floor", type=float, default=0.0)
    parser.add_argument("--scene-root", type=Path, default=POC_ROOT / ".work/real_scene")
    parser.add_argument("--output", type=Path, default=POC_ROOT / "results/substep_guide_matrix.json")
    parser.add_argument("--work", type=Path, default=POC_ROOT / "results/.substep_matrix",
                        help="per-cell probe JSONs, kept so a crashed sweep can be resumed")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cell_command(args, row, motion: str, student: str, output: Path) -> list[str]:
    label, branch, mode, substeps, iterations, _ = row
    command = [
        sys.executable, str(POC_ROOT / "tools/recovery_probe.py"),
        "--scenes", motion,
        "--scene-root", str(args.scene_root),
        "--student", str(POC_ROOT / ".work/hood_data" / student),
        "--steps", str(args.steps),
        # No damage: this matrix is about clean high-inertia behaviour, and the corruption axis is
        # S10d's job. "none" is still passed because it is the in-process control row.
        "--corruptions", "none",
        "--branches", branch,
        "--mode", mode,
        "--substeps", str(substeps),
        "--iterations", str(iterations),
        # Branch B's own count has to match, or the equal-budget claim silently stops holding: the
        # historical 228 buys B the hybrid's whole wall-clock budget, which is a different question.
        "--b-iterations", str(iterations),
        "--sweep", "jacobi",
        "--one-sided", "1",
        "--output", str(output),
    ]
    if mode == "guide":
        command += ["--guide-compliance", str(args.guide_compliance)]
        if args.guide_trust_ratio > 0.0:
            command += ["--guide-trust-ratio", str(args.guide_trust_ratio)]
    if args.area_floor > 0.0:
        command += ["--area-floor", str(args.area_floor)]
    return command


def read_cell(path: Path, motion: str, branch: str) -> dict | None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    scene = report.get("scenes", {}).get(motion)
    if not scene:
        return None
    arm = scene["speeds"]["1x"].get(branch)
    if not arm:
        return None
    clean = arm["clean"]
    return {
        "edge_p95_max": clean["edge_p95_max"],
        "edge_p95_end": clean["edge_p95_end"],
        "edge_p95_at": clean["edge_p95_at"],
        "flipped_max": clean["flipped_max"],
        "collapsed_max": clean["collapsed_max"],
        "area_median_min": clean["area_median_min"],
        "pierced_max": clean["pierced_max"],
        "depth_max": clean["depth_max"],
        "completed_steps": clean["completed_steps"],
        "clip_exhausted_at": scene["speeds"]["1x"].get("clip_exhausted_at"),
    }


def main() -> int:
    args = parse_args()
    rows = [row for row in ROWS if row[0] in args.rows]
    if not rows:
        raise SystemExit(f"no rows matched {args.rows}")
    args.work.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "steps": args.steps,
        "guide_compliance": args.guide_compliance,
        "guide_trust_ratio": args.guide_trust_ratio,
        "area_floor": args.area_floor,
        "processes": args.processes,
        "rows": {label: {"branch": branch, "mode": mode, "substeps": substeps,
                         "iterations": iterations, "total_sweeps": substeps * iterations,
                         "note": note}
                 for label, branch, mode, substeps, iterations, note in rows},
        "cells": {},
    }

    total = len(rows) * len(args.motions) * len(args.students) * args.processes
    done = 0
    for motion in args.motions:
        if not (args.scene_root / motion).is_dir():
            print(f"{motion}: not baked, skipped", flush=True)
            continue
        for student in args.students:
            for row in rows:
                label, branch = row[0], row[1]
                samples = []
                for index in range(args.first_process, args.first_process + args.processes):
                    output = args.work / f"{motion}__{Path(student).stem}__{label}__p{index}.json"
                    command = cell_command(args, row, motion, student, output)
                    done += 1
                    if args.dry_run:
                        print(f"[{done}/{total}] {' '.join(command)}", flush=True)
                        continue
                    if not output.is_file():
                        print(f"[{done}/{total}] {motion} {Path(student).stem} {label} p{index}",
                              flush=True)
                        result = subprocess.run(command, cwd=POC_ROOT, capture_output=True, text=True)
                        if result.returncode != 0:
                            print(result.stdout[-2000:], flush=True)
                            print(result.stderr[-2000:], flush=True)
                            raise SystemExit(f"cell failed: {label} {motion} p{index}")
                    else:
                        print(f"[{done}/{total}] {motion} {Path(student).stem} {label} p{index} "
                              f"(cached)", flush=True)
                    cell = read_cell(output, motion, branch)
                    if cell is not None:
                        samples.append(cell)
                if samples:
                    report["cells"].setdefault(motion, {}).setdefault(Path(student).stem, {})[label] = {
                        "samples": samples,
                        # Reported as a spread rather than a mean, because the spread across
                        # processes IS the resolution of every comparison drawn from this table.
                        "edge_p95_max_range": [min(s["edge_p95_max"] for s in samples),
                                               max(s["edge_p95_max"] for s in samples)],
                        "collapsed_max_range": [min(s["collapsed_max"] for s in samples),
                                                max(s["collapsed_max"] for s in samples)],
                        "area_median_min_range": [min(s["area_median_min"] for s in samples),
                                                  max(s["area_median_min"] for s in samples)],
                    }

    if args.dry_run:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    print_table(report)
    return 0


def print_table(report: dict) -> None:
    for motion, students in sorted(report["cells"].items()):
        for student, rows in sorted(students.items()):
            print(f"\n{motion}  {student}   (equal budget: "
                  f"{next(iter(report['rows'].values()))['total_sweeps']} sweeps/frame)")
            print(f"  {'row':22} {'edgeMAX':>16} {'edgeEnd':>8} {'collMAX':>16} {'areaMIN':>16} "
                  f"{'flip':>6} {'pierce':>6}")
            for label in report["rows"]:
                cell = rows.get(label)
                if cell is None:
                    continue
                low, high = cell["edge_p95_max_range"]
                closed, copen = cell["collapsed_max_range"]
                alow, ahigh = cell["area_median_min_range"]
                end = cell["samples"][0]["edge_p95_end"]
                print(f"  {label:22} {low:7.3f}-{high:7.3f} {end:8.3f} "
                      f"{closed:7.3f}-{copen:7.3f} {alow:7.3f}-{ahigh:7.3f} "
                      f"{cell['samples'][0]['flipped_max']:6.3f} "
                      f"{cell['samples'][0]['pierced_max']:6d}")


if __name__ == "__main__":
    raise SystemExit(main())
