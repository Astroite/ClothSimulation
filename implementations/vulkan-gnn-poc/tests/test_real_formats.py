from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

POC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC_ROOT))

from real_scene.formats import (  # noqa: E402
    FormatError,
    Section,
    load_sectioned,
    load_tensor_asset,
    write_sectioned,
    write_tensor_asset,
)
from real_scene.fine15 import Fine15Graph  # noqa: E402
from real_scene.tinyhood import TinyHood  # noqa: E402
from tools.bake_hood_grid_scene import make_grid, make_uv_sphere  # noqa: E402
from tools.validate_real_assets import check_root_motion  # noqa: E402


class RootMotionGuardTests(unittest.TestCase):
    """The guard has to separate a fast gait from a jump cut, which speed alone cannot do."""

    @staticmethod
    def _line(steps: list[float]) -> tuple[float, ...]:
        position, roots = 0.0, [0.0, 0.0, 0.0]
        for step in steps:
            position += step
            roots += [position, 0.0, 0.0]
        return tuple(roots)

    def test_a_smooth_sprint_launch_is_accepted(self) -> None:
        # sprint_start's measured ramp, which the earlier step-thresholded guard rejected.
        ramp = [0.0064, 0.0347, 0.0792, 0.1488, 0.2749, 0.3247, 0.3130, 0.3146, 0.3092]
        step, jerk = check_root_motion(self._line(ramp))
        self.assertAlmostEqual(step, 0.3247, places=4)
        self.assertAlmostEqual(jerk, 0.1261, places=4)  # 0.2749 -> 0.3247, the ramp's steepest rung

    def test_a_teleport_is_still_rejected(self) -> None:
        # A single-frame jump cut inside an otherwise stationary clip: the step guard's real target.
        with self.assertRaisesRegex(FormatError, "discontinuity"):
            check_root_motion(self._line([0.01, 0.01, 2.0, 0.01, 0.01]))

    def test_an_impossible_gait_is_rejected_even_when_smooth(self) -> None:
        # Ramped gently enough to keep the jerk legal, so only the absolute ceiling can catch it.
        with self.assertRaisesRegex(FormatError, "plausible gait"):
            check_root_motion(self._line([0.2 * index for index in range(12)]))

    def test_a_static_pose_has_no_motion_to_judge(self) -> None:
        self.assertEqual(check_root_motion((0.0, 0.0, 0.0)), (0.0, 0.0))


class RealFormatTests(unittest.TestCase):
    def test_tinyhood_architecture_and_world_edge_contract(self) -> None:
        model = TinyHood().eval()
        self.assertEqual(model.parameter_count, 286_275)
        graph = Fine15Graph(
            effective_position=torch.zeros((2, 3)),
            effective_previous=torch.zeros((2, 3)),
            pin_mask=torch.zeros(2, dtype=torch.bool),
            pin_target=torch.zeros((2, 3)),
            cloth_nodes=torch.zeros((2, 20)),
            obstacle_nodes=torch.zeros((1, 20)),
            mesh_edges=torch.zeros((2, 12)),
            direct_world=torch.zeros((1, 9)),
            inverse_world=torch.zeros((1, 9)),
            mesh_senders=torch.tensor([1, 0]),
            mesh_receivers=torch.tensor([0, 1]),
            world_cloth=torch.tensor([0]),
            world_obstacle=torch.tensor([0]),
            active_obstacle=torch.tensor([0]),
        )
        with torch.no_grad():
            output = model(graph)
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertTrue(torch.isfinite(output).all())

        graph.obstacle_nodes = torch.zeros((0, 20))
        graph.direct_world = torch.zeros((0, 9))
        graph.inverse_world = torch.zeros((0, 9))
        graph.world_cloth = torch.zeros(0, dtype=torch.long)
        graph.world_obstacle = torch.zeros(0, dtype=torch.long)
        graph.active_obstacle = torch.zeros(0, dtype=torch.long)
        with torch.no_grad():
            no_contact_output = model(graph)
        self.assertEqual(tuple(no_contact_output.shape), (2, 3))
        self.assertTrue(torch.isfinite(no_contact_output).all())

    def test_hood_grid64_geometry_contract(self) -> None:
        positions, uvs, triangles, offsets, neighbors, masses = make_grid(64, 1.2, 1.2, 0.2)
        self.assertEqual(len(positions), 4096)
        self.assertEqual(len(uvs), 4096)
        self.assertEqual(len(triangles), 7938)
        self.assertEqual(len(offsets), 4097)
        self.assertEqual(len(neighbors), 32004)
        self.assertEqual(offsets[-1], len(neighbors))
        self.assertTrue(all(mass > 0.0 for mass in masses))
        for vertex in range(4096):
            self.assertNotIn(vertex, neighbors[offsets[vertex] : offsets[vertex + 1]])

        sphere_positions, sphere_normals, sphere_uvs, sphere_triangles = make_uv_sphere(
            (0.0, -0.15, 0.28), 0.3, 32, 64
        )
        self.assertEqual(len(sphere_positions), 1986)
        self.assertEqual(len(sphere_normals), len(sphere_positions))
        self.assertEqual(len(sphere_uvs), len(sphere_positions))
        self.assertEqual(len(sphere_triangles), 3968)

    def test_sectioned_roundtrip_and_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scene.vchar"
            write_sectioned(path, b"VCHAR001", 1, [Section("info", 2, 4, bytes(8)), Section("points", 1, 12, bytes(12))])
            asset = load_sectioned(path, expected_magic=b"VCHAR001", expected_version=1, required_sections=("info", "points"))
            self.assertEqual(asset.require("points", count=1, stride=12).offset % 16, 0)
            with self.assertRaisesRegex(FormatError, "version"):
                load_sectioned(path, expected_version=2)
            with self.assertRaisesRegex(FormatError, "count"):
                asset.require("points", count=2)

            broken = bytearray(path.read_bytes())
            broken[-1] ^= 1
            checksum = root / "checksum.vchar"
            checksum.write_bytes(broken)
            with self.assertRaisesRegex(FormatError, "SHA-256"):
                load_sectioned(checksum)

            broken = bytearray(path.read_bytes())
            broken[0] = ord("X")
            magic = root / "magic.vchar"
            magic.write_bytes(broken)
            with self.assertRaisesRegex(FormatError, "magic"):
                load_sectioned(magic, expected_magic=b"VCHAR001")

            truncated = root / "truncated.vchar"
            truncated.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(FormatError, "file size"):
                load_sectioned(truncated)

    def test_tensor_roundtrip_and_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "fine15.vhood"
            write_tensor_asset(
                path,
                {"model.linear.weight": ((2, 3), bytes(2 * 3 * 4)), "model.linear.bias": ((2,), bytes(2 * 4))},
                checkpoint_sha256="11" * 32,
            )
            model = load_tensor_asset(path)
            self.assertEqual(model.checkpoint_sha256.hex(), "11" * 32)
            self.assertEqual(model.require("model.linear.weight", (2, 3)).offset % 16, 0)
            with self.assertRaisesRegex(FormatError, "shape"):
                model.require("model.linear.weight", (3, 2))
            with self.assertRaisesRegex(FormatError, "magic or version"):
                load_tensor_asset(path, expected_version=2)

            broken = bytearray(path.read_bytes())
            broken[-1] ^= 1
            checksum = root / "checksum.vhood"
            checksum.write_bytes(broken)
            with self.assertRaisesRegex(FormatError, "SHA-256"):
                load_tensor_asset(checksum)

            truncated = root / "truncated.vhood"
            truncated.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(FormatError, "directory declaration"):
                load_tensor_asset(truncated)


if __name__ == "__main__":
    unittest.main()
