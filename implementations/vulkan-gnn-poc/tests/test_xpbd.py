"""Unit tests for the deterministic XPBD projection used by gate G0.

These run on synthetic geometry plus the real CH10032 scene when it is present, and they exist to
separate "the hybrid does not help" from "the solver is wrong" before any gate result is read.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.xpbd import (  # noqa: E402
    BEND,
    DEFAULT_SEARCH_RADIUS,
    STRETCH,
    ConstraintSet,
    SolverConfig,
    bake_tables,
    bend_pairs,
    build_area_constraints,
    build_constraints,
    calibrate_area_from_trajectory,
    contacts_from_graph,
    contacts_from_search,
    load_vxpbd,
    primal_residual,
    project,
    step_substepped,
    stretch_residual,
    triangle_areas,
    undirected_edges,
)

SCENE_ROOT = POC_ROOT / ".work/real_scene"


class FakeScene:
    """The handful of `RuntimeScene` fields the solver reads, for tests that need no asset."""

    def __init__(self, rest: torch.Tensor, triangles: torch.Tensor, mass: torch.Tensor, pins: torch.Tensor):
        self.cloth_rest = rest
        self.cloth_triangles = triangles
        self.cloth_mass = mass
        self.cloth_pins = pins
        senders, receivers = [], []
        for i, j, k in triangles.tolist():
            for a, b in ((i, j), (j, k), (k, i)):
                senders += [a, b]
                receivers += [b, a]
        self.cloth_senders = torch.tensor(senders, dtype=torch.long)
        self.cloth_receivers = torch.tensor(receivers, dtype=torch.long)

    # `step_substepped` interpolates the body between animation frames. A synthetic scene has one
    # static pose, so both of these ignore the time and the substep tests isolate the solver rather
    # than the interpolation. tests/test_xpbd.py::RuntimeSceneInterpolationTests covers the real thing.
    def cloth_target_at(self, time: float) -> torch.Tensor:
        return self.cloth_rest.clone()

    def proxy_at(self, time: float) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((0, 3), dtype=self.cloth_rest.dtype),
            torch.zeros((0, 3), dtype=self.cloth_rest.dtype),
        )


class FakeGraph:
    """The `Fine15Graph` fields `step_substepped` and `contacts_from_graph` read."""

    def __init__(self, position, previous, pin_mask, pin_target):
        self.effective_position = position
        self.effective_previous = previous
        self.pin_mask = pin_mask
        self.pin_target = pin_target
        self.world_cloth = torch.zeros(0, dtype=torch.long)
        self.world_obstacle = torch.zeros(0, dtype=torch.long)
        self.active_obstacle = torch.zeros(0, dtype=torch.long)


def two_triangle_scene() -> FakeScene:
    """A unit square split into two triangles: 4 vertices, 5 edges, 1 interior edge."""
    rest = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    triangles = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    mass = torch.full((4,), 0.25)
    pins = torch.zeros(4, dtype=torch.bool)
    return FakeScene(rest, triangles, mass, pins)


class TopologyTests(unittest.TestCase):
    def test_undirected_edges_halves_the_csr(self) -> None:
        scene = two_triangle_scene()
        pairs = undirected_edges(scene.cloth_senders, scene.cloth_receivers)
        # Six directed halves of five unique edges, minus the duplicate diagonal shared by both
        # triangles, which the CSR stores once per triangle.
        self.assertEqual(pairs.shape[1], 2)
        self.assertTrue(bool((pairs[:, 0] < pairs[:, 1]).all()))
        self.assertEqual({tuple(pair) for pair in pairs.tolist()}, {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)})

    def test_bend_pair_spans_the_interior_edge(self) -> None:
        scene = two_triangle_scene()
        pairs = bend_pairs(scene.cloth_triangles)
        # The only interior edge is 1-2; the opposite corners are 0 and 3.
        self.assertEqual(pairs.tolist(), [[0, 3]])

    def test_gather_tables_match_a_naive_scatter(self) -> None:
        """The padded gather must reproduce index_add_, which is the thing it replaces."""
        scene = two_triangle_scene()
        constraints = build_constraints(scene, scene.cloth_rest)
        count = constraints.count
        gradient = torch.randn(count, 3, generator=torch.Generator().manual_seed(7))
        step = torch.randn(count, generator=torch.Generator().manual_seed(11))

        padded_gradient = torch.cat((gradient, torch.zeros_like(gradient[:1])), dim=0)
        padded_step = torch.cat((step, torch.zeros_like(step[:1])), dim=0)
        gathered = (
            constraints.signs.unsqueeze(-1)
            * padded_gradient[constraints.slots]
            * padded_step[constraints.slots].unsqueeze(-1)
        ).sum(dim=1)

        reference = torch.zeros_like(gathered)
        reference.index_add_(0, constraints.pairs[:, 0], gradient * step.unsqueeze(-1))
        reference.index_add_(0, constraints.pairs[:, 1], -gradient * step.unsqueeze(-1))
        torch.testing.assert_close(gathered, reference, rtol=1e-6, atol=1e-6)

    def test_incident_counts_sum_to_two_per_constraint(self) -> None:
        scene = two_triangle_scene()
        constraints = build_constraints(scene, scene.cloth_rest)
        self.assertEqual(int(constraints.incident.sum().item()), 2 * constraints.count)


class ProjectionTests(unittest.TestCase):
    def _fixture(self, **overrides):
        scene = two_triangle_scene()
        constraints = build_constraints(scene, scene.cloth_rest)
        state = {
            "position": scene.cloth_rest.clone(),
            "inertial": scene.cloth_rest.clone(),
            "pin_mask": scene.cloth_pins,
            "pin_target": scene.cloth_rest.clone(),
            "timestep": 1.0 / 30.0,
            "contacts": None,
        }
        state.update(overrides)
        return scene, constraints, state

    def test_satisfied_state_is_a_fixed_point(self) -> None:
        """No constraint violation and no network displacement must mean no motion, exactly."""
        _, constraints, state = self._fixture()
        for mode in ("standard", "warmstart", "nowarm"):
            result = project(constraints, SolverConfig(iterations=8, mode=mode), **state)
            self.assertTrue(torch.equal(result, state["position"]), f"{mode} moved a satisfied state")

    def test_pinned_vertices_never_move(self) -> None:
        scene = two_triangle_scene()
        pins = torch.tensor([True, False, False, False])
        scene.cloth_pins = pins
        constraints = build_constraints(scene, scene.cloth_rest)
        stretched = scene.cloth_rest.clone()
        stretched[1, 0] = 3.0  # drag vertex 1 far away so the solver has real work to do
        for mode in ("standard", "warmstart", "nowarm"):
            result = project(
                constraints,
                SolverConfig(iterations=16, mode=mode),
                position=stretched,
                inertial=scene.cloth_rest.clone(),
                pin_mask=pins,
                pin_target=scene.cloth_rest.clone(),
                timestep=1.0 / 30.0,
            )
            torch.testing.assert_close(result[0], scene.cloth_rest[0], rtol=0, atol=0)
            self.assertGreater(float((result[1] - stretched[1]).norm()), 0.0, f"{mode} did nothing")

    def test_stretch_residual_decreases_monotonically(self) -> None:
        scene = two_triangle_scene()
        constraints = build_constraints(scene, scene.cloth_rest)
        stretched = scene.cloth_rest * 1.4
        previous = float("inf")
        for iterations in (0, 1, 2, 4, 8, 16):
            result = project(
                constraints,
                SolverConfig(iterations=iterations, stretch_compliance=0.0, bend_compliance=0.0),
                position=stretched,
                inertial=stretched,
                pin_mask=scene.cloth_pins,
                pin_target=scene.cloth_rest,
                timestep=1.0 / 30.0,
            )
            residual = float(stretch_residual(constraints, result))
            self.assertLessEqual(residual, previous + 1.0e-9, f"residual rose at {iterations} iterations")
            previous = residual
        self.assertLess(previous, 0.05, "16 iterations should have made real progress")

    def test_warmstart_discards_a_null_space_displacement(self) -> None:
        """The direct test of what the plan's scheme C actually does to a position prediction.

        A rigid translation satisfies every distance constraint, so it lies in the null space of J
        and no multiplier can express it. `standard` must keep it in full; `warmstart` must project
        it to zero and land on x_tilde, exactly as `nowarm` does. This is the mechanism behind the
        prediction that a warm start can silently throw the network's contribution away.
        """
        scene = two_triangle_scene()
        constraints = build_constraints(scene, scene.cloth_rest)
        inertial = scene.cloth_rest.clone()
        displaced = inertial + torch.tensor([0.0, 0.0, 0.1])

        arguments = {
            "position": displaced,
            "inertial": inertial,
            "pin_mask": scene.cloth_pins,
            "pin_target": scene.cloth_rest,
            "timestep": 1.0 / 30.0,
        }
        before = float(primal_residual(constraints, displaced, inertial, scene.cloth_pins))
        # primal_residual is an RMS over all three components, so a 0.1 m shift along one axis
        # reads as 0.1 / sqrt(3).
        self.assertAlmostEqual(before, 0.1 / 3.0**0.5, places=6)

        standard = project(constraints, SolverConfig(iterations=4, mode="standard"), **arguments)
        warmstart = project(constraints, SolverConfig(iterations=4, mode="warmstart"), **arguments)
        nowarm = project(constraints, SolverConfig(iterations=4, mode="nowarm"), **arguments)

        self.assertAlmostEqual(
            float(primal_residual(constraints, standard, inertial, scene.cloth_pins)), before, places=6,
            msg="standard XPBD should leave a constraint-satisfying displacement alone",
        )
        torch.testing.assert_close(warmstart, nowarm, rtol=1e-6, atol=1e-7)
        self.assertLess(
            float(primal_residual(constraints, warmstart, inertial, scene.cloth_pins)), 1.0e-6,
            "warmstart should have projected a null-space displacement to nothing",
        )

    def test_warmstart_keeps_a_displacement_a_multiplier_can_express(self) -> None:
        """The complement: a stretch along one edge is in the range of J^T, so it must survive."""
        scene = two_triangle_scene()
        constraints = build_constraints(scene, scene.cloth_rest)
        inertial = scene.cloth_rest.clone()
        displaced = inertial.clone()
        displaced[1] = displaced[1] + torch.tensor([0.15, 0.0, 0.0])

        arguments = {
            "position": displaced,
            "inertial": inertial,
            "pin_mask": scene.cloth_pins,
            "pin_target": scene.cloth_rest,
            "timestep": 1.0 / 30.0,
        }
        warmstart = project(constraints, SolverConfig(iterations=0, mode="warmstart"), **arguments)
        # iterations=0 short-circuits, so compare the initialiser via one sweep against nowarm.
        started = project(constraints, SolverConfig(iterations=1, mode="warmstart"), **arguments)
        nowarm = project(constraints, SolverConfig(iterations=1, mode="nowarm"), **arguments)
        self.assertTrue(torch.equal(warmstart, displaced), "iterations=0 must be a pass-through")
        self.assertGreater(
            float((started - nowarm).norm()), 1.0e-4,
            "a stretch is expressible as a multiplier, so warmstart must differ from nowarm",
        )

    def test_coloured_and_jacobi_agree_on_the_fixed_point(self) -> None:
        """Different schedules, same solution: both must drive the residual down to the same place."""
        scene = two_triangle_scene()
        constraints = build_constraints(scene, scene.cloth_rest)
        stretched = scene.cloth_rest * 1.3
        arguments = {
            "position": stretched, "inertial": stretched, "pin_mask": scene.cloth_pins,
            "pin_target": scene.cloth_rest, "timestep": 1.0 / 30.0,
        }
        coloured = project(constraints, SolverConfig(iterations=400, sweep="coloured"), **arguments)
        jacobi = project(constraints, SolverConfig(iterations=400, sweep="jacobi"), **arguments)
        self.assertLess(float(stretch_residual(constraints, coloured)), 1.0e-4)
        self.assertLess(float(stretch_residual(constraints, jacobi)), 1.0e-3)

    def test_colouring_is_a_valid_vertex_disjoint_partition(self) -> None:
        scene = two_triangle_scene()
        constraints = build_constraints(scene, scene.cloth_rest)
        for group in constraints.colour_groups():
            touched = constraints.pairs[group].reshape(-1).tolist()
            self.assertEqual(len(touched), len(set(touched)), "a colour reused a vertex")

    def test_one_sided_ignores_compression(self) -> None:
        scene = two_triangle_scene()
        constraints = build_constraints(scene, scene.cloth_rest)
        compressed = scene.cloth_rest * 0.6
        arguments = {
            "inertial": compressed,
            "pin_mask": scene.cloth_pins,
            "pin_target": scene.cloth_rest,
            "timestep": 1.0 / 30.0,
        }
        one_sided = project(
            constraints, SolverConfig(iterations=8, one_sided=True), position=compressed, **arguments
        )
        two_sided = project(
            constraints, SolverConfig(iterations=8, one_sided=False), position=compressed, **arguments
        )
        self.assertTrue(torch.equal(one_sided, compressed), "one-sided mode resisted compression")
        self.assertGreater(float((two_sided - compressed).norm()), 1.0e-4, "two-sided mode did nothing")

    def test_repeated_projection_is_bit_identical(self) -> None:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for name in devices:
            device = torch.device(name)
            scene = two_triangle_scene()
            for field, value in vars(scene).items():
                setattr(scene, field, value.to(device))
            constraints = build_constraints(scene, scene.cloth_rest)
            stretched = (scene.cloth_rest * 1.3).contiguous()
            config = SolverConfig(iterations=12, mode="warmstart")
            first = project(
                constraints, config, position=stretched, inertial=scene.cloth_rest,
                pin_mask=scene.cloth_pins, pin_target=scene.cloth_rest, timestep=1.0 / 30.0,
            )
            for _ in range(4):
                again = project(
                    constraints, config, position=stretched, inertial=scene.cloth_rest,
                    pin_mask=scene.cloth_pins, pin_target=scene.cloth_rest, timestep=1.0 / 30.0,
                )
                self.assertTrue(torch.equal(first, again), f"projection not reproducible on {name}")


@unittest.skipUnless((SCENE_ROOT / "hml_001962").is_dir(), "baked scenes not present")
class FusedPortTests(unittest.TestCase):
    """The Vulkan kernel is one thread per vertex. These pin down that it computes the same thing.

    `_apply_jacobi` needs two dispatches per iteration on a GPU (constraints, then vertices);
    `_apply_fused` needs one, at the cost of every vertex recomputing its neighbours' multiplier
    updates and storing lambda per (vertex, slot) instead of per constraint. That restructuring is
    the whole reason the hybrid fits in 0.36 ms instead of 0.72 ms, so it has to be proved here
    rather than discovered after the HLSL is written.
    """

    def _real_fixture(self):
        from real_scene.runtime_scene import RuntimeScene

        scene = RuntimeScene.load(SCENE_ROOT / "hml_001962", "hml_001962", device="cpu", asset_stem="ch10032")
        start = scene.cloth_target(0)
        constraints = build_constraints(scene, start)
        disturbed = start + 0.02 * torch.randn(start.shape, generator=torch.Generator().manual_seed(11))
        return scene, constraints, start, disturbed

    def test_fused_sweep_matches_jacobi(self) -> None:
        """Bit-identical on the real mesh, for both compliance regimes and both residual signs."""
        scene, constraints, start, disturbed = self._real_fixture()
        for one_sided in (False, True):
            for compliance in (0.0, 1.0e-2):
                shared = dict(
                    iterations=16, mode="standard", one_sided=one_sided,
                    stretch_compliance=compliance, bend_compliance=compliance, collision=False,
                )
                state = dict(
                    position=disturbed, inertial=disturbed, pin_mask=scene.cloth_pins,
                    pin_target=start, timestep=1.0 / 30.0,
                )
                jacobi = project(constraints, SolverConfig(sweep="jacobi", **shared), **state)
                fused = project(constraints, SolverConfig(sweep="fused", **shared), **state)
                label = f"one_sided={one_sided} compliance={compliance}"
                self.assertTrue(torch.equal(jacobi, fused), f"fused diverged from jacobi ({label})")
                # Guard against the comparison passing because neither did anything.
                self.assertGreater(float((fused - disturbed).abs().max()), 1.0e-4, label)

    def test_fused_lambda_copies_stay_equal_across_endpoints(self) -> None:
        """The two per-slot copies of one constraint's lambda must not drift apart.

        This is the property that makes the redundant layout safe. It holds because both endpoints
        form `x[pairs[c, 0]] - x[pairs[c, 1]]` in the stored order and read `weight_sum` from the
        baked table instead of adding `w_a + w_b` in whichever order is local to them.
        """
        from real_scene.xpbd import _apply_fused

        scene, constraints, start, disturbed = self._real_fixture()
        config = SolverConfig(iterations=1, sweep="fused", one_sided=True, collision=False)
        tables = bake_tables(constraints, config, 1.0 / 30.0)
        current = disturbed.clone()
        multiplier = torch.zeros_like(constraints.slots, dtype=current.dtype)
        pinned = scene.cloth_pins.reshape(-1, 1)
        averaging = constraints.incident.clamp_min(1.0)

        real = constraints.slots < constraints.count
        for _ in range(8):
            current, multiplier = _apply_fused(
                current, multiplier, constraints, config, tables.denominator,
                tables.alpha, tables.alive, averaging, pinned,
            )
            # Group the live slots by the constraint they name and check every constraint's copies
            # agree exactly. Each constraint owns exactly two live slots.
            index = constraints.slots[real]
            value = multiplier[real]
            order = torch.argsort(index, stable=True)
            paired = value[order].reshape(-1, 2)
            self.assertEqual(paired.shape[0], constraints.count)
            self.assertTrue(torch.equal(paired[:, 0], paired[:, 1]), "lambda copies drifted apart")

    def test_baked_tables_are_what_the_asset_must_carry(self) -> None:
        scene, constraints, _, _ = self._real_fixture()
        config = SolverConfig(stretch_compliance=1.0e-2, bend_compliance=1.0e-2)
        tables = bake_tables(constraints, config, 1.0 / 30.0)
        self.assertEqual(tables.weight_sum.shape, (constraints.count,))
        torch.testing.assert_close(tables.denominator, (tables.weight_sum + tables.alpha).clamp_min(1.0e-20))
        # alpha = compliance / dt^2 = 1e-2 * 900 = 9.0, which is the magnitude gate G0 found the
        # 0..1e-6 sweep had missed by seven orders.
        torch.testing.assert_close(tables.alpha.max(), torch.tensor(9.0))
        # Both endpoints pinned is the only way a constraint can be dead, and the waistband has
        # internal edges, so this must not be empty or the `alive` flag is untested in practice.
        self.assertGreater(int((~tables.alive).sum()), 0, "expected some fully pinned constraints")

    def test_zero_iterations_returns_the_input_unchanged(self) -> None:
        """The Vulkan side needs the same short circuit, so k=0 is a no-op both places."""
        scene, constraints, start, disturbed = self._real_fixture()
        for sweep in ("jacobi", "fused", "coloured"):
            result = project(
                constraints, SolverConfig(iterations=0, sweep=sweep),
                position=disturbed, inertial=disturbed, pin_mask=scene.cloth_pins,
                pin_target=start, timestep=1.0 / 30.0,
            )
            self.assertTrue(result is disturbed or torch.equal(result, disturbed), sweep)

    def test_baked_asset_round_trips_into_an_equivalent_solve(self) -> None:
        """Read the `.vxpbd` back the way the kernel will and re-run the sweep from it.

        This is the layout guard. A wrong stride, a signs column read as int, or slots flattened in
        the other order would all still load, still produce plausible cloth, and be very hard to
        find later from a Vulkan-versus-Python mismatch of unknown origin. Reconstructing the
        constraint set from the bytes and demanding a bit-identical solve localises it here.
        """
        import subprocess
        import tempfile

        scene, constraints, start, disturbed = self._real_fixture()
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "hml_001962.vxpbd"
            subprocess.run(
                [sys.executable, "-B", str(POC_ROOT / "tools/bake_xpbd_constraints.py"),
                 "--scene", "hml_001962", "--calibration", "bind", "--device", "cpu",
                 "--output", str(asset)],
                check=True, capture_output=True, cwd=POC_ROOT,
            )
            rebuilt = load_vxpbd(asset)

        self.assertEqual(rebuilt.count, constraints.count)
        self.assertEqual(rebuilt.vertex_count, constraints.vertex_count)
        for field in ("pairs", "target_length", "kind", "slots", "signs", "inverse_mass"):
            self.assertTrue(
                torch.equal(getattr(rebuilt, field), getattr(constraints, field)),
                f"{field} did not survive the round trip",
            )
        # incident is clamped on the way out, so compare against the clamped form.
        self.assertTrue(torch.equal(rebuilt.incident, constraints.incident.clamp_min(1.0)))

        config = SolverConfig(iterations=8, sweep="fused", one_sided=True, collision=False)
        state = dict(
            position=disturbed, inertial=disturbed, pin_mask=scene.cloth_pins,
            pin_target=start, timestep=1.0 / 30.0,
        )
        self.assertTrue(torch.equal(
            project(rebuilt, config, **state), project(constraints, config, **state)
        ), "a solve driven by the baked asset diverged from one driven by build_constraints")

    def test_baked_min_edge_is_per_vertex_not_global(self) -> None:
        """gnn-xpbd-v2.md section 7.1: a global minimum would let one short edge clamp everything."""
        from tools.bake_xpbd_constraints import per_vertex_min_edge

        _, constraints, _, _ = self._real_fixture()
        minimum = per_vertex_min_edge(constraints.pairs, constraints.target_length, constraints.vertex_count)
        self.assertEqual(minimum.shape, (constraints.vertex_count,))
        self.assertGreater(float(minimum.max() / minimum.min()), 5.0, "edge scale is not uniform")
        for index in (0, constraints.vertex_count // 2, constraints.vertex_count - 1):
            touching = (constraints.pairs == index).any(dim=1)
            expected = float(constraints.target_length[touching].min())
            self.assertAlmostEqual(float(minimum[index]), expected, places=6)


class RealSceneTests(unittest.TestCase):
    """Numbers here are the measurements plans/gnn/gnn-xpbd-v2.md relies on."""

    @classmethod
    def setUpClass(cls) -> None:
        from real_scene.runtime_scene import RuntimeScene

        cls.scene = RuntimeScene.load(SCENE_ROOT / "hml_001962", "hml_001962", device="cpu", asset_stem="ch10032")

    def test_measured_constraint_counts(self) -> None:
        constraints = build_constraints(self.scene, self.scene.cloth_rest)
        stretch = int((constraints.kind == STRETCH).sum())
        bend = int((constraints.kind == BEND).sum())
        self.assertEqual(stretch, 3947, "undirected mesh edge count changed")
        # 2570 triangles give 3947 unique edges, 3763 of them interior. Those yield 3763 raw
        # opposite-corner pairs but only 3735 distinct ones: 28 pairs of interior edges happen to
        # share both opposite corners, and a duplicated constraint would simply be applied twice
        # with double weight, so bend_pairs deduplicates.
        self.assertEqual(bend, 3735, "interior-edge or bend dedup count changed")
        self.assertEqual(constraints.vertex_count, 1377)
        self.assertEqual(int(self.scene.cloth_pins.sum()), 72)

    def test_stretch_and_bend_constrain_disjoint_vertex_pairs(self) -> None:
        """No vertex pair may carry both a stretch and a bend target, or they would fight."""
        constraints = build_constraints(self.scene, self.scene.cloth_rest)
        as_set = lambda mask: {tuple(pair) for pair in constraints.pairs[mask].tolist()}  # noqa: E731
        self.assertEqual(as_set(constraints.kind == STRETCH) & as_set(constraints.kind == BEND), set())

    def test_reference_choice_changes_the_targets(self) -> None:
        """`cloth_rest` and the skinned frame 0 are genuinely different configurations."""
        rest = build_constraints(self.scene, self.scene.cloth_rest)
        bind = build_constraints(self.scene, self.scene.cloth_target(0))
        ratio = bind.target_length / rest.target_length
        self.assertGreater(float(ratio.max()), 2.0, "expected the skinning to stretch some edges")
        self.assertEqual(int(rest.suspect.sum()), 0, "rest against itself cannot be suspect")
        self.assertGreater(int(bind.suspect.sum()), 0, "expected some edges outside the trusted band")

    def test_inverse_mass_is_non_uniform_and_zero_on_pins(self) -> None:
        constraints = build_constraints(self.scene, self.scene.cloth_target(0))
        free = constraints.inverse_mass[~self.scene.cloth_pins]
        self.assertGreater(float(free.max() / free.min()), 10.0, "mass spread collapsed to uniform")
        self.assertEqual(float(constraints.inverse_mass[self.scene.cloth_pins].abs().max()), 0.0)

    def test_solver_reduces_stretch_on_the_real_mesh(self) -> None:
        constraints = build_constraints(self.scene, self.scene.cloth_target(0))
        start = self.scene.cloth_target(0)
        disturbed = start + 0.02 * torch.randn(start.shape, generator=torch.Generator().manual_seed(3))
        before = float(stretch_residual(constraints, disturbed))
        after = float(stretch_residual(constraints, project(
            constraints, SolverConfig(iterations=8, stretch_compliance=0.0),
            position=disturbed, inertial=disturbed,
            pin_mask=self.scene.cloth_pins, pin_target=start, timestep=1.0 / 30.0,
        )))
        self.assertLess(after, before * 0.8, f"8 iterations barely helped: {before:.5f} -> {after:.5f}")


class SubstepTests(unittest.TestCase):
    """The substep loop and the soft guide, on synthetic geometry."""

    def _fixture(self, *, pins=None):
        scene = two_triangle_scene()
        if pins is not None:
            scene.cloth_pins = pins
        constraints = build_constraints(scene, scene.cloth_rest)
        position = scene.cloth_rest.clone()
        previous = scene.cloth_rest.clone()
        graph = FakeGraph(position, previous, scene.cloth_pins, scene.cloth_rest.clone())
        # A guide that stretches the square, so the guide and the constraints genuinely disagree.
        guide = scene.cloth_rest * 1.30
        return scene, constraints, graph, guide

    def test_one_substep_reproduces_project_bit_for_bit(self) -> None:
        """The regression anchor for the whole change: substeps=1 must change nothing at all.

        Every result in results/ was measured through `project` called once per step. If this fails,
        no comparison against those numbers means anything, so it is worth pinning at the bit level
        rather than with a tolerance -- see `real_scene/xpbd.py::_blend` for the one place where the
        obvious implementation would have lost the last few bits.
        """
        scene, constraints, graph, guide = self._fixture()
        config = SolverConfig(iterations=8, mode="standard", sweep="fused", one_sided=True,
                              collision=False)
        expected = project(
            constraints, config,
            position=guide, inertial=2.0 * graph.effective_position - graph.effective_previous,
            pin_mask=graph.pin_mask, pin_target=graph.pin_target, timestep=1.0 / 30.0,
        )
        actual = step_substepped(
            constraints, config, scene=scene, graph=graph, guide=guide,
            timestep=1.0 / 30.0, frame=0.0, frame_advance=1.0, substeps=1,
        )
        self.assertTrue(torch.equal(expected, actual))

    def test_zero_confidence_guide_is_exactly_pure_xpbd(self) -> None:
        """The guide's safe end of the dial has to be *exactly* the constraints-only solve.

        This is the property the whole soft-guide scheme rests on: a vertex the trust region rejects
        must fall back to physics, not to physics-plus-a-little-network. Note what is deliberately
        NOT asserted here -- there is no compliance that makes the guide reproduce `mode="standard"`.
        `standard` applies the network once with infinite stiffness and never again; the guide applies
        it every iteration with finite stiffness. The dial is continuous towards pure XPBD only.
        """
        scene, constraints, graph, guide = self._fixture()
        common = dict(sweep="fused", iterations=8, one_sided=True, collision=False)
        nowarm = step_substepped(
            constraints, SolverConfig(mode="nowarm", **common), scene=scene, graph=graph,
            guide=None, timestep=1.0 / 30.0, frame=0.0, frame_advance=1.0, substeps=1,
        )
        # A trust radius of zero makes `guide_confidence` reject every vertex whose guide has moved
        # at all, which for a guide that stretches the mesh is all of them.
        guided = step_substepped(
            constraints,
            SolverConfig(mode="guide", guide_compliance=1.0, guide_trust_ratio=1.0e-9, **common),
            scene=scene, graph=graph, guide=guide, timestep=1.0 / 30.0, frame=0.0,
            frame_advance=1.0, substeps=1, min_edge=torch.ones(constraints.vertex_count),
        )
        self.assertTrue(torch.equal(nowarm, guided))

    def test_guide_authority_is_monotone_in_compliance(self) -> None:
        """Lower compliance must land closer to the network's prediction. That is the dial."""
        scene, constraints, graph, guide = self._fixture()
        distances = []
        for compliance in (100.0, 10.0, 1.0, 0.1):
            result = step_substepped(
                constraints,
                SolverConfig(mode="guide", guide_compliance=compliance, sweep="fused",
                             iterations=8, one_sided=True, collision=False),
                scene=scene, graph=graph, guide=guide, timestep=1.0 / 30.0,
                frame=0.0, frame_advance=1.0, substeps=1,
            )
            distances.append(float((result - guide).square().mean().sqrt()))
        for tighter, looser in zip(distances[1:], distances[:-1]):
            self.assertLess(tighter, looser, f"guide authority not monotone: {distances}")

    def test_compliant_guide_does_not_converge_onto_the_guide(self) -> None:
        """Iterating must reach the compliant equilibrium, not silently stiffen into a hard guide.

        This is what accumulating the guide multiplier buys. Recomputing it from scratch each
        iteration would let a repeated soft pull walk the vertex all the way onto the target, which
        would make `guide_compliance` a lie at high iteration counts -- and 128 is the shipping count.
        """
        scene, constraints, graph, guide = self._fixture()
        gap = []
        for iterations in (8, 32, 128):
            result = step_substepped(
                constraints,
                SolverConfig(mode="guide", guide_compliance=10.0, sweep="fused",
                             iterations=iterations, one_sided=True, collision=False),
                scene=scene, graph=graph, guide=guide, timestep=1.0 / 30.0,
                frame=0.0, frame_advance=1.0, substeps=1,
            )
            gap.append(float((result - guide).square().mean().sqrt()))
        self.assertGreater(gap[-1], 0.1 * gap[0],
                           f"the guide collapsed onto its target as iterations grew: {gap}")

    def test_substeps_do_not_change_the_total_guide_travel(self) -> None:
        """The guide arrives in instalments, so the last substep still aims at the full prediction."""
        scene, constraints, graph, guide = self._fixture()
        config = SolverConfig(mode="guide", guide_compliance=1.0e-6, sweep="fused",
                              iterations=32, one_sided=True, collision=False)
        for substeps in (1, 2, 4):
            result = step_substepped(
                constraints, config, scene=scene, graph=graph, guide=guide,
                timestep=1.0 / 30.0, frame=0.0, frame_advance=1.0, substeps=substeps,
            )
            # A near-zero compliance is a near-hard guide, so whatever the schedule, the final state
            # has to sit near the guide rather than a fraction of the way there.
            self.assertLess(float((result - guide).abs().max()), 0.05,
                            f"substeps={substeps} did not deliver the whole guide")

    def test_substeps_must_be_positive(self) -> None:
        scene, constraints, graph, guide = self._fixture()
        with self.assertRaises(ValueError):
            step_substepped(constraints, SolverConfig(iterations=1), scene=scene, graph=graph,
                            guide=guide, timestep=1.0 / 30.0, frame=0.0, frame_advance=1.0,
                            substeps=0)

    def test_search_radius_tracks_the_graph_builder(self) -> None:
        """`contacts_from_search` must accept exactly the proxies the network's own search accepted."""
        from real_scene.fine15 import COLLISION_RADIUS

        self.assertEqual(DEFAULT_SEARCH_RADIUS, COLLISION_RADIUS)


class AreaConstraintTests(unittest.TestCase):
    def _fixture(self, *, pins=None):
        scene = two_triangle_scene()
        if pins is not None:
            scene.cloth_pins = pins
        constraints = build_constraints(scene, scene.cloth_rest)
        area = build_area_constraints(scene, reference_position=scene.cloth_rest)
        return scene, constraints, area

    def test_tables_cover_every_corner_once(self) -> None:
        scene, _, area = self._fixture()
        self.assertEqual(int(area.incident.sum().item()), 3 * area.count)
        live = area.slots < area.count
        for vertex in range(area.vertex_count):
            owned = {(int(area.slots[vertex, k]), int(area.corner[vertex, k]))
                     for k in range(area.slots.shape[1]) if live[vertex, k]}
            for triangle, corner in owned:
                self.assertEqual(int(scene.cloth_triangles[triangle, corner]), vertex)

    def test_target_area_matches_the_geometry(self) -> None:
        scene, _, area = self._fixture()
        # Two right triangles of legs 1, so each has area 1/2.
        torch.testing.assert_close(area.target_area, torch.full((2,), 0.5), rtol=0, atol=1e-7)

    def test_floor_is_inert_when_every_triangle_is_large_enough(self) -> None:
        """A one-sided floor must be a no-op on a satisfied state, exactly."""
        scene, constraints, area = self._fixture()
        state = {
            "position": scene.cloth_rest.clone(), "inertial": scene.cloth_rest.clone(),
            "pin_mask": scene.cloth_pins, "pin_target": scene.cloth_rest.clone(),
            "timestep": 1.0 / 30.0,
        }
        result = project(constraints, SolverConfig(iterations=8, sweep="fused", area_floor=0.5),
                         area=area, **state)
        self.assertTrue(torch.equal(result, state["position"]))

    def test_floor_inflates_a_squashed_triangle(self) -> None:
        """The gap the existing constraint set leaves: an edge too short is legal, an area of zero
        is legal, and a one-sided stretch constraint has no opinion about either."""
        scene, constraints, area = self._fixture()
        squashed = scene.cloth_rest.clone()
        squashed[:, 1] *= 0.05          # flatten the square towards a sliver
        before = triangle_areas(squashed, scene.cloth_triangles) / area.target_area
        config = SolverConfig(iterations=64, sweep="fused", one_sided=True, area_floor=0.4)
        result = project(constraints, config, position=squashed, inertial=squashed,
                         pin_mask=scene.cloth_pins, pin_target=scene.cloth_rest.clone(),
                         timestep=1.0 / 30.0, area=area)
        after = triangle_areas(result, scene.cloth_triangles) / area.target_area
        self.assertLess(float(before.min()), 0.1)
        self.assertGreater(float(after.min()), float(before.min()) * 2.0,
                           f"area floor did not inflate: {before.tolist()} -> {after.tolist()}")

    def test_floor_never_moves_a_pinned_vertex(self) -> None:
        pins = torch.tensor([True, True, False, False])
        scene, constraints, area = self._fixture(pins=pins)
        squashed = scene.cloth_rest.clone()
        squashed[:, 1] *= 0.05
        target = scene.cloth_rest.clone()
        result = project(constraints, SolverConfig(iterations=32, sweep="fused", area_floor=0.5),
                         position=squashed, inertial=squashed, pin_mask=pins, pin_target=target,
                         timestep=1.0 / 30.0, area=area)
        self.assertTrue(torch.equal(result[pins], target[pins]))

    def test_area_gradients_match_finite_differences(self) -> None:
        """The three corner gradients in `_apply_area`, checked against the area they differentiate.

        A sign error here would inflate one corner and deflate another, which on a real mesh reads as
        noise rather than as a broken constraint -- so it is checked directly.
        """
        generator = torch.Generator().manual_seed(19)
        corner = [torch.randn(3, dtype=torch.float64, generator=generator) for _ in range(3)]

        def area_of(points):
            return 0.5 * torch.linalg.cross(points[1] - points[0], points[2] - points[0]).norm()

        normal = torch.linalg.cross(corner[1] - corner[0], corner[2] - corner[0])
        unit = normal / normal.norm()
        analytic = [
            0.5 * torch.linalg.cross(corner[1] - corner[2], unit),
            0.5 * torch.linalg.cross(corner[2] - corner[0], unit),
            0.5 * torch.linalg.cross(corner[0] - corner[1], unit),
        ]
        # A constraint that cannot move the centre of mass must have gradients summing to zero.
        self.assertLess(float(sum(analytic).abs().max()), 1e-12)
        epsilon = 1e-6
        for index in range(3):
            for axis in range(3):
                shifted = [point.clone() for point in corner]
                shifted[index][axis] += epsilon
                plus = area_of(shifted)
                shifted[index][axis] -= 2.0 * epsilon
                numeric = float((plus - area_of(shifted)) / (2.0 * epsilon))
                self.assertAlmostEqual(float(analytic[index][axis]), numeric, places=6)

    def test_calibrated_area_is_the_median_not_the_rest(self) -> None:
        scene = two_triangle_scene()
        frames = [scene.cloth_rest * scale for scale in (1.0, 2.0, 3.0, 10.0)]
        median = calibrate_area_from_trajectory(scene.cloth_triangles, frames)
        # Areas scale with the square of the frame scale: 0.5, 2, 4.5, 50. `torch.median` takes the
        # lower of the two middle elements rather than averaging them, so this is 2 and not 3.25 --
        # the same convention `calibrate_from_trajectory` already relies on for the lengths.
        torch.testing.assert_close(median, torch.full((2,), 2.0), rtol=0, atol=1e-6)
        # `skip` drops the settling transient, exactly as the length calibration does: the remaining
        # areas are 4.5 and 50, whose lower median is 4.5.
        skipped = calibrate_area_from_trajectory(scene.cloth_triangles, frames, skip=2)
        torch.testing.assert_close(skipped, torch.full((2,), 4.5), rtol=0, atol=1e-5)


@unittest.skipUnless((SCENE_ROOT / "hml_001962").is_dir(), "baked scenes not present")
class RuntimeSceneInterpolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from real_scene.runtime_scene import RuntimeScene

        cls.scene = RuntimeScene.load(SCENE_ROOT / "hml_001962", "hml_001962", device="cpu",
                                      asset_stem="ch10032")

    def test_integral_time_is_bit_identical_to_the_frame_lookup(self) -> None:
        """Substepping must not perturb the un-substepped path, and the frames are where it would.

        `matrices_at` short-circuits an integral time to the stored matrices rather than blending with
        weight 0, because `lerp(a, b, 0)` is only equal to `a` up to rounding.
        """
        for frame in (0, 7, self.scene.frame_count - 1):
            position, normal = self.scene.proxy(frame)
            interpolated_position, interpolated_normal = self.scene.proxy_at(float(frame))
            self.assertTrue(torch.equal(position, interpolated_position), f"proxy pos at {frame}")
            self.assertTrue(torch.equal(normal, interpolated_normal), f"proxy nrm at {frame}")
            self.assertTrue(torch.equal(self.scene.cloth_target(frame),
                                        self.scene.cloth_target_at(float(frame))))

    def test_fractional_time_lands_between_the_two_frames(self) -> None:
        before = self.scene.proxy_at(3.0)[0]
        after = self.scene.proxy_at(4.0)[0]
        middle = self.scene.proxy_at(3.5)[0]
        torch.testing.assert_close(middle, 0.5 * (before + after), rtol=1e-6, atol=1e-7)

    def test_time_past_the_last_frame_clamps(self) -> None:
        last = self.scene.frame_count - 1
        self.assertTrue(torch.equal(self.scene.proxy(last)[0], self.scene.proxy_at(last + 9.0)[0]))
        self.assertTrue(torch.equal(self.scene.proxy(0)[0], self.scene.proxy_at(-3.0)[0]))

    def test_search_reproduces_the_graph_pairing(self) -> None:
        """The substep contact search has to agree with the network's own on the frame they share."""
        from real_scene.fine15 import Fine15, Fine15Weights

        vhood = POC_ROOT / ".work/hood_data/fine15.vhood"
        if not vhood.is_file():
            self.skipTest("fine15.vhood not present")
        builder = Fine15(Fine15Weights.from_vhood(vhood, device=torch.device("cpu")))
        position = self.scene.cloth_target(0)
        proxy_position, normals = self.scene.proxy(0)
        proxy_target, _ = self.scene.proxy(1)
        graph = builder.prepare_graph(
            position=position, previous=position.clone(), rest_position=self.scene.cloth_rest,
            triangles=self.scene.cloth_triangles, mesh_senders=self.scene.cloth_senders,
            mesh_receivers=self.scene.cloth_receivers, mass=self.scene.cloth_mass,
            pin_mask=self.scene.cloth_pins, pin_target=self.scene.cloth_target(1),
            obstacle_position=proxy_position, obstacle_target=proxy_target,
            obstacle_normals=normals, timestep=1.0 / 30.0,
        )
        expected = contacts_from_graph(graph, proxy_target, normals)
        actual = contacts_from_search(graph.effective_position, proxy_position, proxy_target, normals)
        self.assertTrue(torch.equal(expected.vertex, actual.vertex))
        torch.testing.assert_close(expected.point, actual.point, rtol=0, atol=0)
        torch.testing.assert_close(expected.normal, actual.normal, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
