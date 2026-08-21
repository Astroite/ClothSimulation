"""Load and skin the generated CH10032 runtime assets with PyTorch."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .formats import SectionView, load_sectioned


def _tensor(section: SectionView, dtype: torch.dtype, components: int) -> torch.Tensor:
    # Own writable storage so PyTorch does not warn about the read-only mmap.
    result = torch.frombuffer(bytearray(section.data), dtype=dtype).clone()
    return result.reshape(section.count, components) if components > 1 else result


@dataclass
class RuntimeScene:
    render_positions: torch.Tensor
    render_normals: torch.Tensor
    render_bones: torch.Tensor
    render_weights: torch.Tensor
    proxy_positions: torch.Tensor
    proxy_normals: torch.Tensor
    proxy_bones: torch.Tensor
    proxy_weights: torch.Tensor
    cloth_rest: torch.Tensor
    cloth_triangles: torch.Tensor
    cloth_senders: torch.Tensor
    cloth_receivers: torch.Tensor
    cloth_mass: torch.Tensor
    cloth_pins: torch.Tensor
    cloth_bones: torch.Tensor
    cloth_weights: torch.Tensor
    skin_matrices: torch.Tensor
    root_positions: torch.Tensor
    fps: int

    @property
    def frame_count(self) -> int:
        return int(self.skin_matrices.shape[0])

    @classmethod
    def load(
        cls,
        root: Path | str,
        motion: str,
        device: torch.device | str = "cpu",
        asset_stem: str = "ch10032",
    ) -> "RuntimeScene":
        root = Path(root)
        character = load_sectioned(root / f"{asset_stem}.vchar", expected_magic=b"VCHAR001", expected_version=1)
        cloth_name = "ch10032_lower.vcloth2" if asset_stem == "ch10032" else f"{asset_stem}.vcloth2"
        cloth = load_sectioned(root / cloth_name, expected_magic=b"VCLTH002", expected_version=2)
        animation = load_sectioned(root / f"{motion}.vanim", expected_magic=b"VANIM001", expected_version=1)
        render_vertices, _, bone_count, proxy_vertices, _, _ = struct.unpack("<6I", character.require("info", count=6, stride=4).data)
        frame_count, animation_bones, fps, _ = struct.unpack("<4I", animation.require("info", count=4, stride=4).data)
        if animation_bones != bone_count:
            raise ValueError("VCHAR and VANIM bone counts differ")
        cloth_vertices = cloth.require("positions", stride=12).count
        offsets = _tensor(cloth.require("csr_offsets", count=cloth_vertices + 1, stride=4), torch.uint32, 1).to(torch.long)
        neighbors = _tensor(cloth.require("csr_neighbors", stride=4), torch.uint32, 1).to(torch.long)
        receivers = torch.repeat_interleave(torch.arange(cloth_vertices), offsets[1:] - offsets[:-1])
        scene = cls(
            render_positions=_tensor(character.require("render_pos", count=render_vertices, stride=12), torch.float32, 3),
            render_normals=_tensor(character.require("render_nrm", count=render_vertices, stride=12), torch.float32, 3),
            render_bones=_tensor(character.require("bone_idx", count=render_vertices, stride=48), torch.uint32, 12).to(torch.long),
            render_weights=_tensor(character.require("bone_weight", count=render_vertices, stride=48), torch.float32, 12),
            proxy_positions=_tensor(character.require("proxy_pos", count=proxy_vertices, stride=12), torch.float32, 3),
            proxy_normals=_tensor(character.require("proxy_nrm", count=proxy_vertices, stride=12), torch.float32, 3),
            proxy_bones=_tensor(character.require("proxy_bone_idx", count=proxy_vertices, stride=48), torch.uint32, 12).to(torch.long),
            proxy_weights=_tensor(character.require("proxy_weight", count=proxy_vertices, stride=48), torch.float32, 12),
            cloth_rest=_tensor(cloth.require("positions", count=cloth_vertices, stride=12), torch.float32, 3),
            cloth_triangles=_tensor(cloth.require("triangles", stride=12), torch.uint32, 3).to(torch.long),
            cloth_senders=neighbors,
            cloth_receivers=receivers,
            cloth_mass=_tensor(cloth.require("mass", count=cloth_vertices, stride=4), torch.float32, 1),
            cloth_pins=_tensor(cloth.require("pin_mask", count=cloth_vertices, stride=4), torch.uint32, 1).to(torch.bool),
            cloth_bones=_tensor(cloth.require("bone_idx", count=cloth_vertices, stride=48), torch.uint32, 12).to(torch.long),
            cloth_weights=_tensor(cloth.require("bone_weight", count=cloth_vertices, stride=48), torch.float32, 12),
            skin_matrices=_tensor(animation.require("skin_matrices", count=frame_count * bone_count, stride=48), torch.float32, 12).reshape(frame_count, bone_count, 3, 4),
            root_positions=_tensor(animation.require("root_pos", count=frame_count, stride=12), torch.float32, 3),
            fps=fps,
        )
        for field, value in vars(scene).items():
            if isinstance(value, torch.Tensor):
                setattr(scene, field, value.to(device))
        return scene

    def _skin_positions_with(
        self, matrices: torch.Tensor, positions: torch.Tensor, bones: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        picked = matrices[bones]
        homogeneous = torch.cat((positions, torch.ones_like(positions[:, :1])), dim=-1)
        transformed = torch.einsum("vbij,vbj->vbi", picked, homogeneous[:, None, :].expand(-1, bones.shape[1], -1))
        return (transformed * weights[..., None]).sum(dim=1)

    def _skin_normals_with(
        self, matrices: torch.Tensor, normals: torch.Tensor, bones: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        rotations = matrices[bones][..., :3]
        transformed = torch.einsum("vbij,vbj->vbi", rotations, normals[:, None, :].expand(-1, bones.shape[1], -1))
        return F.normalize((transformed * weights[..., None]).sum(dim=1), dim=-1)

    def skin_positions(self, positions: torch.Tensor, bones: torch.Tensor, weights: torch.Tensor, frame: int) -> torch.Tensor:
        return self._skin_positions_with(self.skin_matrices[frame], positions, bones, weights)

    def skin_normals(self, normals: torch.Tensor, bones: torch.Tensor, weights: torch.Tensor, frame: int) -> torch.Tensor:
        return self._skin_normals_with(self.skin_matrices[frame], normals, bones, weights)

    def matrices_at(self, time: float) -> torch.Tensor:
        """Skin matrices at a fractional frame, as a linear blend of the two bracketing frames.

        Substepping the solver needs the body somewhere between two animation frames, and this is the
        only interpolation the asset supports: `.vanim` stores skin matrices that are already composed
        (bone times bind-inverse), not per-bone translate/rotate/scale, so there is nothing to slerp.
        Blending the composed matrices is the standard linear approximation and it slightly shrinks a
        fast-rotating limb mid-blend -- worth knowing on exactly the sprint clips this is for, where
        the root covers 0.32 m per frame.

        What matters more than the approximation is that hood_skin.comp does the *same* one. The
        Python reference and the Vulkan runtime are compared bit-for-bit, so a different sub-frame
        pose on either side would show up as a solver mismatch. Integral `time` returns the stored
        matrices untouched, which keeps every existing golden reproducing.
        """
        last = self.frame_count - 1
        clamped = min(max(float(time), 0.0), float(last))
        lower = int(math.floor(clamped))
        if lower >= last:
            return self.skin_matrices[last]
        blend = clamped - float(lower)
        if blend == 0.0:
            return self.skin_matrices[lower]
        return torch.lerp(self.skin_matrices[lower], self.skin_matrices[lower + 1], blend)

    def cloth_target(self, frame: int) -> torch.Tensor:
        return self.skin_positions(self.cloth_rest, self.cloth_bones, self.cloth_weights, frame)

    def proxy(self, frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.skin_positions(self.proxy_positions, self.proxy_bones, self.proxy_weights, frame),
            self.skin_normals(self.proxy_normals, self.proxy_bones, self.proxy_weights, frame),
        )

    def cloth_target_at(self, time: float) -> torch.Tensor:
        """`cloth_target` at a fractional frame. See `matrices_at`."""
        return self._skin_positions_with(
            self.matrices_at(time), self.cloth_rest, self.cloth_bones, self.cloth_weights
        )

    def proxy_at(self, time: float) -> tuple[torch.Tensor, torch.Tensor]:
        """`proxy` at a fractional frame. See `matrices_at`."""
        matrices = self.matrices_at(time)
        return (
            self._skin_positions_with(matrices, self.proxy_positions, self.proxy_bones, self.proxy_weights),
            self._skin_normals_with(matrices, self.proxy_normals, self.proxy_bones, self.proxy_weights),
        )
