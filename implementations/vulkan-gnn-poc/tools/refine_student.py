#!/usr/bin/env python3
"""Refine a trained student by hill-climbing the closed-loop metric itself.

Why this exists instead of more gradient descent. The 32x12 student in
results/STUDENT_32X12_RESULTS.md holds teacher-level structure for about 119 steps and then
drifts. Continuing to train it with train_student.py makes it WORSE, and measurement says the
reason is not the learning rate:

* Phase 1 (single-step fit) at 3e-4 moved the 150-step edge P95 from 2.4 to 50 while single-step
  variance explained barely moved (0.980 -> 0.975).
* Phase 2 (16-step unroll) at 1e-5 -- fifteen Adam steps, a weight change of at most 1.5e-4 --
  moved the selection score from 0.43 to 1.03.
* Adding ISOTROPIC RANDOM NOISE at ten times that magnitude (1e-3 of each tensor's own standard
  deviation) IMPROVED the score to 0.274 and 0.280 on two seeds, dropping edge P95 at 150 steps
  from 2.49 to 1.39.

Random perturbations help while gradient steps hurt, so the delivered weights are not a fragile
knife-edge and they are not a local optimum of closed-loop stability either. The proxy objective
is what is wrong: matching the teacher's trajectory over 8-16 steps is anti-correlated with
surviving 150. A trajectory-matching loss cannot see an instability whose whole character is that
it takes a hundred steps to show up.

A 150-step rollout costs about two seconds, so the metric that actually matters is cheap enough
to optimise directly. Two modes, both searching the space the metric is defined on rather than a
surrogate:

* `--mode hill` -- (1+1) evolution strategy with a 1/5-success step-size rule. Cheap and it moves
  fast at first: 0.622 to 0.255 on the 180-step score in 26 minutes. Then it stalls, because a
  single evaluation carries +/-0.035 of noise and the improvements still available are smaller
  than that, so the comparison that drives it becomes a coin flip.
* `--mode es` -- antithetic-pair evolution strategy. Each iteration evaluates a population of
  mirrored perturbations and combines them into one direction estimate, so the evaluation noise is
  averaged down across the population instead of being resolved by a single comparison. Slower per
  iteration, but it keeps making progress where the hill climb stops.

Two guards keep the search honest:

* The fit floor. The cheapest way to look stable is to predict almost nothing: HOOD's Verlet
  integration keeps constant velocity under zero acceleration, so a dead network drifts smoothly
  rather than exploding. A candidate whose single-step variance explained falls below the floor is
  rejected regardless of its rollout score. The floor is relative to the starting model because
  the absolute number is entirely a function of which states the probe holds -- the same 32x12
  weights score 0.980 on a probe weighted towards step 0 and grid64, and 0.614 on 40 states of the
  tpose trajectory.
* The incumbent is re-evaluated every iteration rather than cached. Scoring is nondeterministic
  (index_add_ has no deterministic CUDA float kernel) with a measured spread of 0.07 on a score
  of 0.40, so a cached incumbent that got a lucky draw would never be displaced.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))
sys.path.insert(0, str(POC_ROOT / "tools"))

from real_scene.fine15 import Fine15, Fine15Weights  # noqa: E402
from real_scene.formats import sha256_file  # noqa: E402
from real_scene.runtime_scene import RuntimeScene  # noqa: E402
from real_scene.tinyhood import TinyHood, export_tinyhood, load_tinyhood  # noqa: E402
from train_student import (  # noqa: E402
    StudentSample,
    make_graph,
    rollout_curve,
    stability_score,
    teacher_forced_fit,
    teacher_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", type=Path, required=True, help=".pt checkpoint to refine")
    parser.add_argument("--mode", choices=("hill", "es"), default="hill")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260819)

    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--steps", type=int, default=150, help="rollout horizon scored each iteration")
    parser.add_argument("--eval-repeats", type=int, default=2, help="rollouts averaged per candidate")
    parser.add_argument("--confirm-repeats", type=int, default=5, help="rollouts averaged before shipping a new best")
    parser.add_argument("--sigma", type=float, default=1.0e-3, help="initial perturbation, relative to each tensor's std")
    parser.add_argument("--sigma-min", type=float, default=3.0e-5)
    parser.add_argument("--sigma-max", type=float, default=2.0e-2)
    parser.add_argument("--sigma-window", type=int, default=12, help="iterations the 1/5 success rule looks back over")

    parser.add_argument(
        "--population",
        type=int,
        default=8,
        help="es mode: mirrored pairs per iteration. Each pair costs two rollouts and the pairing "
             "is what cancels the shared evaluation noise",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.5,
        help="es mode: step length as a multiple of sigma along the estimated direction. The "
             "advantage is standardised across the population, so this is scale-free",
    )

    parser.add_argument(
        "--var-floor-ratio",
        type=float,
        default=0.9,
        help="reject candidates whose single-step fit falls below this fraction of the starting "
             "model's. Relative rather than absolute because the absolute number depends "
             "entirely on which states the probe contains -- the same 32x12 weights score 0.980 "
             "on a probe weighted towards step 0 and grid64, and 0.614 on 40 states of the "
             "tpose trajectory",
    )
    parser.add_argument(
        "--var-floor",
        type=float,
        default=None,
        help="absolute override for --var-floor-ratio",
    )
    parser.add_argument("--probe-states", type=int, default=40, help="teacher-labelled states behind the fit floor")
    parser.add_argument("--stiff-weight", type=float, default=0.25)
    parser.add_argument("--drift-weight", type=float, default=2.0)
    parser.add_argument("--over-cap", type=float, default=2.0)

    parser.add_argument("--scene-root", type=Path, default=POC_ROOT / ".work/real_scene")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "ch10032_tpose:ch10032:180",
            "hood_grid64:hood_grid64:120",
            "ch10032_sprint:ch10032:61",
            "hml_001962:ch10032:136",
        ],
        help="key:asset_stem:steps triples. The score is the mean across all of them, because a "
             "single-scene objective overfits: refining on ch10032_tpose alone took its 360-step "
             "score from 2.743 to 0.259 while making hood_grid64 worse (3.050 to 4.066)",
    )
    parser.add_argument("--fine15", type=Path, default=POC_ROOT / ".work/hood_data/fine15.vhood")
    parser.add_argument("--tag", default="_refined")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


@dataclass
class RefineScene:
    key: str
    scene: RuntimeScene
    steps: int
    reference: dict
    positions: torch.Tensor


def perturb_state(
    state: dict[str, torch.Tensor],
    sigma: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Isotropic Gaussian noise scaled per tensor by that tensor's own standard deviation.

    Scaling per tensor rather than globally matters because a LayerNorm gain and a first-layer
    weight matrix live on different scales; one absolute sigma would be a no-op for one and a
    reset for the other.
    """
    result: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if value.dtype.is_floating_point and value.numel() > 1:
            scale = float(value.std().item()) * sigma
            noise = torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype)
            result[name] = value + noise * scale
        else:
            result[name] = value.clone()
    return result


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    weights = Fine15Weights.from_vhood(args.fine15.resolve(), device=device)
    teacher = Fine15(weights)
    mean, std = weights.normalizer("output")

    checkpoint = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
    architecture = checkpoint["architecture"]
    latent, blocks = int(architecture["latent"]), int(architecture["blocks"])
    label = f"{latent}x{blocks}{args.tag}"
    args.checkpoint = args.checkpoint or POC_ROOT / f".work/hood_data/student{label}.pt"
    args.output = args.output or POC_ROOT / f".work/hood_data/student{label}.vhood"
    args.report = args.report or POC_ROOT / f"results/student{label}_python.json"

    model = TinyHood(latent=latent, blocks=blocks).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    scenes: list[RefineScene] = []
    for entry in args.scenes:
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(f"--scenes entries must be key:asset_stem:steps, got {entry!r}")
        key, stem, steps = parts[0], parts[1], int(parts[2])
        loaded = RuntimeScene.load(args.scene_root / key, key, device=device, asset_stem=stem)
        # An animated scene has as many usable steps as it has frame transitions; asking for more
        # would score the student against a teacher that has run out of animation.
        usable = steps if loaded.frame_count == 1 else min(steps, loaded.frame_count - 1)
        curve = rollout_curve(teacher.predict_graph, teacher, loaded, usable, mean, std, store_positions=True)
        positions = curve.pop("positions")
        scenes.append(RefineScene(key, loaded, usable, curve, positions))
        print(json.dumps({"scene": key, "steps": usable, "cloth_vertices": int(loaded.cloth_rest.shape[0]),
                          "teacher_edge_p95_final": round(curve["edge_ratio_p95"], 4)}, sort_keys=True), flush=True)

    # A small teacher-labelled probe purely for the fit floor. Clean states off the teacher's own
    # trajectory are enough: the guard only has to notice a network going dead, not rank two
    # healthy networks. Spread across scenes so a candidate cannot go dead on one of them unseen.
    probe: list[StudentSample] = []
    per_scene = max(args.probe_states // len(scenes), 1)
    with torch.no_grad():
        for entry in scenes:
            for step, position, previous in teacher_trajectory(teacher, entry.scene, per_scene, mean, std):
                graph = make_graph(teacher, entry.scene, position, previous, step)
                target = teacher.predict_graph(graph)
                if torch.isfinite(target).all():
                    probe.append(StudentSample(entry.key, step, position, previous, target, 0.0))
    scene_by_key = {entry.key: entry.scene for entry in scenes}

    def score_of(repeats: int) -> tuple[float, dict]:
        """Mean score across every scene, averaged over `repeats` rollouts of each.

        Averaging across scenes rather than scoring one is what stops the search from buying a
        long tpose rollout at the expense of grid64, which is exactly what a single-scene run did.
        """
        per_scene_scores: dict[str, float] = {}
        detail: dict = {}
        for entry in scenes:
            total = 0.0
            for _ in range(repeats):
                trace = rollout_curve(
                    model, teacher, entry.scene, entry.steps, mean, std, reference=entry.positions
                )
                total += stability_score(trace, entry.reference, args)
                if entry is scenes[0]:
                    detail = {key: value for key, value in trace.items() if not key.endswith("_curve")}
            per_scene_scores[entry.key] = total / repeats
        detail["per_scene_score"] = per_scene_scores
        return sum(per_scene_scores.values()) / len(per_scene_scores), detail

    incumbent = {key: value.detach().clone() for key, value in model.state_dict().items()}
    baseline_score, baseline_detail = score_of(args.confirm_repeats)
    baseline_fit = teacher_forced_fit(model, probe, scene_by_key, teacher)
    baseline_variance = baseline_fit["variance_explained"]
    variance_floor = args.var_floor if args.var_floor is not None else baseline_variance * args.var_floor_ratio
    best = {"state": incumbent, "score": baseline_score, "iteration": 0, "detail": baseline_detail,
            "variance_explained": baseline_variance}
    print(json.dumps({
        "baseline": {
            "score": round(baseline_score, 5),
            "per_scene": {key: round(value, 5) for key, value in baseline_detail["per_scene_score"].items()},
            "variance_explained": round(baseline_variance, 4),
            "variance_floor": round(variance_floor, 4),
            "probe_states": len(probe),
        }
    }, sort_keys=True), flush=True)

    sigma = args.sigma
    outcomes: list[bool] = []
    history: list[dict] = []
    rejected_by_floor = 0
    started = time.perf_counter()

    def save_best() -> str:
        """Write the current best straight away.

        The search runs for over an hour, so keeping the only copy in memory until the loop ends
        means an interrupted run throws away everything it found.
        """
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "architecture": architecture,
            "seed": args.seed,
            "selected": {"phase": f"refine_{args.mode}", "epoch": best["iteration"], "score": best["score"]},
            "refined_from": {"path": str(args.resume), "score": baseline_score},
            "model": best["state"],
        }, args.checkpoint.resolve())
        return sha256_file(args.checkpoint.resolve())

    def promote(state: dict, iteration: int, variance: float) -> bool:
        """Confirm a candidate with more rollouts, then ship it if it still wins.

        The per-rollout spread is 0.07, so a candidate that won a single comparison may just have
        drawn a lucky rollout; without this the search would ratchet on noise.
        """
        model.load_state_dict(state, strict=True)
        confirmed, detail = score_of(args.confirm_repeats)
        # Re-measure the incumbent with the same budget in the same iteration. Comparing against a
        # cached best_score would let a lucky earlier draw block every later improvement, and the
        # score's spread is 0.07 on a score of 0.40.
        model.load_state_dict(best["state"], strict=True)
        reference_score, _ = score_of(args.confirm_repeats)
        best["score"] = reference_score
        if confirmed >= reference_score:
            return False
        best.update({
            "state": {key: value.detach().clone() for key, value in state.items()},
            "score": confirmed,
            "iteration": iteration,
            "detail": detail,
            "variance_explained": variance,
        })
        save_best()
        return True

    if args.mode == "hill":
        for iteration in range(1, args.iterations + 1):
            candidate = perturb_state(incumbent, sigma, generator)
            model.load_state_dict(candidate, strict=True)
            candidate_fit = teacher_forced_fit(model, probe, scene_by_key, teacher)
            if candidate_fit["variance_explained"] < variance_floor:
                # Not scored at all: a dead network can post an excellent rollout score, so letting
                # it compete and filtering afterwards would waste the rollouts and bias the 1/5 rule.
                rejected_by_floor += 1
                outcomes.append(False)
                accepted = False
                candidate_score = float("nan")
            else:
                candidate_score, _ = score_of(args.eval_repeats)
                model.load_state_dict(incumbent, strict=True)
                incumbent_score, _ = score_of(args.eval_repeats)
                # Elitist: the incumbent IS the best-known state, and it only moves when a
                # candidate survives confirmation. A non-elitist walk drifts -- once the incumbent
                # has wandered to a worse score, mediocre candidates beat it and get accepted, the
                # success rate stays high, the 1/5 rule pushes sigma to its ceiling, and the search
                # never returns. That is exactly what a multi-scene run did for 37 iterations
                # without once improving on its starting point.
                accepted = candidate_score < incumbent_score and promote(
                    candidate, iteration, candidate_fit["variance_explained"]
                )
                outcomes.append(accepted)
                if accepted:
                    incumbent = best["state"]
            model.load_state_dict(incumbent, strict=True)

            window = outcomes[-args.sigma_window:]
            if len(window) >= args.sigma_window:
                rate = sum(window) / len(window)
                sigma = sigma * 1.3 if rate > 0.2 else sigma / 1.3
                sigma = min(max(sigma, args.sigma_min), args.sigma_max)
                outcomes = outcomes[-args.sigma_window:]

            history.append({
                "iteration": iteration,
                "sigma": sigma,
                "candidate_score": None if candidate_score != candidate_score else candidate_score,
                "accepted": accepted,
                "best_score": best["score"],
            })
            if iteration % 10 == 0 or accepted:
                print(json.dumps({
                    "iteration": iteration,
                    "sigma": round(sigma, 8),
                    "candidate": None if candidate_score != candidate_score else round(candidate_score, 5),
                    "accepted": accepted,
                    "best": round(best["score"], 5),
                    "best_iteration": best["iteration"],
                    "floor_rejects": rejected_by_floor,
                    "elapsed_minutes": round((time.perf_counter() - started) / 60.0, 1),
                }, sort_keys=True), flush=True)
    else:
        pairs = max(args.population // 2, 1)
        names = [name for name, value in incumbent.items()
                 if value.dtype.is_floating_point and value.numel() > 1]
        scales = {name: float(incumbent[name].std().item()) for name in names}
        for iteration in range(1, args.iterations + 1):
            directions: list[dict[str, torch.Tensor]] = []
            advantages: list[float] = []
            for _ in range(pairs):
                direction = {
                    name: torch.randn(incumbent[name].shape, generator=generator,
                                      device=device, dtype=incumbent[name].dtype)
                    for name in names
                }
                scored = []
                for sign in (1.0, -1.0):
                    mirrored = {key: value.clone() for key, value in incumbent.items()}
                    for name in names:
                        mirrored[name] = incumbent[name] + direction[name] * (sign * sigma * scales[name])
                    model.load_state_dict(mirrored, strict=True)
                    fit = teacher_forced_fit(model, probe, scene_by_key, teacher)
                    if fit["variance_explained"] < variance_floor:
                        # Charge a dead mirror the worst score in the population rather than
                        # dropping the pair: dropping it would silently bias the direction estimate
                        # towards whichever half of the space still passes the floor.
                        rejected_by_floor += 1
                        scored.append(None)
                    else:
                        scored.append(score_of(args.eval_repeats)[0])
                if scored[0] is None and scored[1] is None:
                    continue
                penalty = max(value for value in scored if value is not None) * 2.0
                plus = scored[0] if scored[0] is not None else penalty
                minus = scored[1] if scored[1] is not None else penalty
                directions.append(direction)
                # Lower is better, so a negative advantage means the +sigma side was the good one.
                advantages.append(minus - plus)
            if not directions:
                print(json.dumps({"iteration": iteration, "stopped": "whole population failed the fit floor"}), flush=True)
                break

            values = torch.tensor(advantages, dtype=torch.float32)
            spread = float(values.std().item()) if values.numel() > 1 else 0.0
            # Standardising the advantage makes --learning-rate independent of how noisy this
            # particular population happened to be; without it the step length would track the
            # score's local scale rather than the direction's reliability.
            weights_es = values / spread if spread > 1.0e-9 else torch.zeros_like(values)
            updated = {key: value.clone() for key, value in incumbent.items()}
            for name in names:
                step = torch.zeros_like(incumbent[name])
                for index, direction in enumerate(directions):
                    step = step + direction[name] * float(weights_es[index])
                updated[name] = incumbent[name] + step * (args.learning_rate * sigma * scales[name] / len(directions))
            model.load_state_dict(updated, strict=True)
            updated_fit = teacher_forced_fit(model, probe, scene_by_key, teacher)
            if updated_fit["variance_explained"] < variance_floor:
                sigma = max(sigma / 1.3, args.sigma_min)
                accepted = False
                updated_score = float("nan")
            else:
                updated_score, _ = score_of(args.eval_repeats)
                model.load_state_dict(incumbent, strict=True)
                incumbent_score, _ = score_of(args.eval_repeats)
                accepted = updated_score < incumbent_score
                if accepted:
                    accepted = promote(updated, iteration, updated_fit["variance_explained"])
                if accepted:
                    incumbent = best["state"]
                else:
                    # A rejected step means sigma is too large for the local curvature, the same
                    # signal the 1/5 rule reads in hill mode.
                    sigma = max(sigma / 1.2, args.sigma_min)
            model.load_state_dict(incumbent, strict=True)

            history.append({
                "iteration": iteration,
                "sigma": sigma,
                "candidate_score": None if updated_score != updated_score else updated_score,
                "accepted": accepted,
                "advantage_spread": spread,
                "best_score": best["score"],
            })
            print(json.dumps({
                "iteration": iteration,
                "sigma": round(sigma, 8),
                "candidate": None if updated_score != updated_score else round(updated_score, 5),
                "accepted": accepted,
                "advantage_spread": round(spread, 5),
                "best": round(best["score"], 5),
                "best_iteration": best["iteration"],
                "floor_rejects": rejected_by_floor,
                "elapsed_minutes": round((time.perf_counter() - started) / 60.0, 1),
            }, sort_keys=True), flush=True)

    search_seconds = time.perf_counter() - started
    model.load_state_dict(best["state"], strict=True)
    model.eval()
    final_fit = teacher_forced_fit(model, probe, scene_by_key, teacher)

    checkpoint_hash = save_best()
    export_info = export_tinyhood(model, weights, args.output.resolve(), checkpoint_sha256=checkpoint_hash)

    reloaded = load_tinyhood(args.output.resolve(), device=device)
    saved, restored = model.state_dict(), reloaded.state_dict()
    mismatched = {
        name: float((saved[name] - restored[name]).abs().max().item())
        for name in saved
        if not torch.equal(saved[name], restored[name])
    }
    if mismatched:
        raise ValueError(f"VHOOD reload is not bit-identical: {mismatched}")

    report = {
        "architecture": architecture,
        "method": {"hill": "(1+1) evolution strategy with 1/5 success rule on the closed-loop score",
                   "es": "antithetic-pair evolution strategy on the closed-loop score"}[args.mode],
        "mode": args.mode,
        "population": args.population if args.mode == "es" else None,
        "learning_rate": args.learning_rate if args.mode == "es" else None,
        "rationale": "gradient steps on the 8-16 step trajectory proxy degrade 150-step stability "
                     "while random perturbations ten times larger improve it, so the proxy is the "
                     "problem rather than the step size",
        "refined_from": {"path": str(args.resume), "score": baseline_score,
                         "variance_explained": baseline_fit["variance_explained"]},
        "scenes": args.scenes,
        "score_weights": {"stiff": args.stiff_weight, "drift": args.drift_weight, "over_cap": args.over_cap},
        "eval_repeats": args.eval_repeats,
        "confirm_repeats": args.confirm_repeats,
        "variance_floor": variance_floor,
        "var_floor_ratio": args.var_floor_ratio,
        "baseline_variance_explained": baseline_variance,
        "iterations": args.iterations,
        "floor_rejects": rejected_by_floor,
        "search_seconds": search_seconds,
        "selected": {"iteration": best["iteration"], "score": best["score"]},
        "final_variance_explained": final_fit["variance_explained"],
        "final_rollout": best["detail"],
        "checkpoint_sha256": checkpoint_hash,
        "vhood": {key: value for key, value in export_info.items() if key != "tensors"},
        "vhood_parameters_bit_identical": True,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "selected", "refined_from", "final_variance_explained", "final_rollout", "floor_rejects", "search_seconds"
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
