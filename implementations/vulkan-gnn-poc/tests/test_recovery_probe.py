"""Unit tests for the corruptions tools/recovery_probe.py injects and the speed axis it sweeps.

The recovery experiment's whole argument is that `fold` and `penetrate` put their damage in the null
space of J^T -- every intra-patch distance preserved, so no Lagrange multiplier can express the
displacement and no number of sweeps can see it. That is a property of the corruption, not of the
solver, so it has to be asserted here rather than hoped for while reading recovery curves. If these
tests fail, a flat forgetting curve for branch B means the corruption was a scramble the constraints
legitimately could not undo, and proves nothing about the architecture.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))
sys.path.insert(0, str(POC_ROOT / "tools"))

from recovery_probe import (  # noqa: E402
    corrupt_fold,
    corrupt_penetrate,
    corrupt_stretch,
    geodesic_patch,
    isometry_error,
)
from train_student import frame_of  # noqa: E402

SCENE_ROOT = POC_ROOT / ".work/real_scene"


class Args:
    """The handful of fields the corruptions read."""

    patch = 200
    stretch_amount = 0.30
    penetrate_depth = 0.12


class FrameScaleTests(unittest.TestCase):
    def test_unit_scale_returns_the_step_itself(self) -> None:
        # Every existing golden is pinned to the old `min(step, count-1)` expression, so the default
        # path has to be the identity and not merely close to it.
        for step in range(0, 400, 7):
            self.assertIs(type(frame_of(step, 1.0)), int)
            self.assertEqual(frame_of(step, 1.0), step)
            self.assertEqual(frame_of(step), step)

    def test_scaled_playback_advances_the_clip_faster(self) -> None:
        self.assertEqual([frame_of(s, 2.0) for s in range(5)], [0, 2, 4, 6, 8])
        self.assertEqual([frame_of(s, 3.0) for s in range(4)], [0, 3, 6, 9])
        # A fractional scale must stay monotone, or the body would walk backwards mid-rollout.
        walked = [frame_of(step, 1.5) for step in range(12)]
        self.assertEqual(walked, sorted(walked))

    def test_integer_scale_target_frame_leads_by_one_step_not_one_frame(self) -> None:
        # make_graph pairs frame_of(step) with frame_of(step + 1); at 2x that is a two-frame gap,
        # which is what keeps the pin target on the pose the body will actually reach.
        for scale in (1.0, 2.0, 3.0):
            for step in range(6):
                self.assertEqual(frame_of(step + 1, scale) - frame_of(step, scale), int(scale))


@unittest.skipUnless((SCENE_ROOT / "ch10032_tpose").is_dir(), "baked scenes not present")
class CorruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from real_scene.runtime_scene import RuntimeScene

        cls.scene = RuntimeScene.load(SCENE_ROOT / "ch10032_tpose", "ch10032_tpose",
                                      device=torch.device("cpu"), asset_stem="ch10032")
        cls.position = cls.scene.cloth_target(0)
        cls.proxy = cls.scene.proxy(0)
        cls.args = Args()

    def edge_lengths(self, position: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(
            position[self.scene.cloth_senders] - position[self.scene.cloth_receivers], dim=-1)

    def signed_to_body(self, position: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        """Signed distance of each patch vertex to its nearest proxy plane; negative is inside."""
        points, normals = self.proxy
        nearest = torch.cdist(position[patch], points).argmin(dim=-1)
        outward = torch.nn.functional.normalize(normals[nearest], dim=-1)
        return ((position[patch] - points[nearest]) * outward).sum(dim=-1)

    def test_patch_is_deterministic_free_and_connected_on_the_mesh(self) -> None:
        first = geodesic_patch(self.scene, self.position, self.args.patch)
        second = geodesic_patch(self.scene, self.position, self.args.patch)
        self.assertTrue(torch.equal(first, second), "a rerun must damage the same vertices")
        self.assertEqual(first.numel(), self.args.patch)
        self.assertEqual(int(first.unique().numel()), self.args.patch)
        self.assertFalse(bool(self.scene.cloth_pins[first].any()), "pins would be snapped back")
        # Connected over mesh edges: every vertex but the seed has a neighbour already in the patch.
        inside = torch.zeros(self.position.shape[0], dtype=torch.bool)
        inside[first] = True
        senders, receivers = self.scene.cloth_senders, self.scene.cloth_receivers
        neighbour = torch.zeros_like(inside)
        neighbour[receivers[inside[senders]]] = True
        neighbour[senders[inside[receivers]]] = True
        self.assertGreaterEqual(int((inside & neighbour).sum()), self.args.patch - 1)

    def test_fold_preserves_every_intra_patch_distance(self) -> None:
        patch = geodesic_patch(self.scene, self.position, self.args.patch)
        folded = corrupt_fold(self.scene, self.position, self.proxy, self.args)
        # A half-turn is orthogonal, so this is float round-off and nothing else.
        self.assertLess(isometry_error(self.position, folded, patch), 1.0e-6)

    def test_penetrate_preserves_every_intra_patch_distance(self) -> None:
        patch = geodesic_patch(self.scene, self.position, self.args.patch)
        pushed = corrupt_penetrate(self.scene, self.position, self.proxy, self.args)
        self.assertLess(isometry_error(self.position, pushed, patch), 1.0e-6)

    def test_fold_actually_moves_the_patch_and_only_the_patch(self) -> None:
        patch = geodesic_patch(self.scene, self.position, self.args.patch)
        folded = corrupt_fold(self.scene, self.position, self.proxy, self.args)
        moved = torch.linalg.vector_norm(folded - self.position, dim=-1)
        outside = torch.ones(moved.shape[0], dtype=torch.bool)
        outside[patch] = False
        self.assertEqual(float(moved[outside].max()), 0.0, "damage must stay inside the patch")
        self.assertGreater(float(moved[patch].max()), 0.1, "a near-no-op fold would prove nothing")

    def test_penetrate_puts_the_patch_behind_the_body_surface(self) -> None:
        """The default depth has to land in the window where penetration is actually detectable.

        Measured fraction of the patch behind its nearest proxy plane, by translation depth:
        0.05 m -> 4.5%, 0.08 -> 20.0%, 0.12 -> 28.5%, 0.16 -> 11.0%, 0.25 -> 1.0%. It is not
        monotone, because past ~0.16 m the patch is out the far side of the leg and a nearest-proxy
        half-plane calls it legal again. So this test guards a window, not a floor: a larger depth is
        a *worse* corruption to measure, not a stronger one.
        """
        patch = geodesic_patch(self.scene, self.position, self.args.patch)
        pushed = corrupt_penetrate(self.scene, self.position, self.proxy, self.args)
        before = float((self.signed_to_body(self.position, patch) < 0.0).float().mean())
        after = float((self.signed_to_body(pushed, patch) < 0.0).float().mean())
        self.assertLess(before, 0.02, "the undamaged hem should barely touch the body")
        self.assertGreater(after, 0.15, "this has to be a penetration, not a lift or a tunnel")

    def test_the_isometric_damage_is_confined_to_the_patch_boundary(self) -> None:
        """The quantitative form of the null-space claim: how much of the mesh reports a violation.

        `stretch` puts a violation on essentially every edge, so it is entirely inside what a
        distance solver can represent. `fold` is the corruption a viewer would call catastrophic, yet
        it violates only the edges crossing the patch boundary -- a few percent of the mesh. That
        gap, not the magnitude of any single residual, is what the constraints are blind to.
        """
        rest = self.edge_lengths(self.position)
        broken = lambda after: float(((self.edge_lengths(after) - rest).abs() > 1.0e-4).float().mean())
        stretched = broken(corrupt_stretch(self.scene, self.position, self.proxy, self.args))
        folded = broken(corrupt_fold(self.scene, self.position, self.proxy, self.args))
        pushed = broken(corrupt_penetrate(self.scene, self.position, self.proxy, self.args))
        self.assertGreater(stretched, 0.9, "stretch is the positive control and must be visible")
        self.assertLess(folded, 0.15, f"fold should touch a boundary ring, not {folded:.1%} of edges")
        self.assertLess(pushed, 0.15)


if __name__ == "__main__":
    unittest.main()
