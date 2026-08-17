"""Strict little-endian containers used by the real-character PoC.

The runtime formats deliberately contain no pickle, JSON, FBX, or USD parser.
Human-readable JSON files are sidecars only; all information needed by the
runtime is validated from these binary containers.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ALIGNMENT = 16
SECTION_HEADER = struct.Struct("<8sIIQQ32s32s")
SECTION_ENTRY = struct.Struct("<16sQII")
VCLOTH1_HEADER = struct.Struct("<8sIIQ32s8s")
TENSOR_HEADER = struct.Struct("<8sIIQQ32s32s")
TENSOR_ENTRY = struct.Struct("<160sQII8I")


class FormatError(ValueError):
    """Raised when a runtime asset fails strict structural validation."""


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Section:
    name: str
    count: int
    stride: int
    data: bytes

    def __post_init__(self) -> None:
        try:
            encoded = self.name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise FormatError(f"section name is not ASCII: {self.name!r}") from exc
        if not encoded or len(encoded) >= 16:
            raise FormatError(f"section name must contain 1..15 ASCII bytes: {self.name!r}")
        if self.count < 0 or self.stride <= 0:
            raise FormatError(f"invalid section shape for {self.name}: {self.count} x {self.stride}")
        if len(self.data) != self.count * self.stride:
            raise FormatError(
                f"section {self.name} has {len(self.data)} bytes, expected {self.count * self.stride}"
            )


@dataclass(frozen=True)
class SectionView:
    name: str
    count: int
    stride: int
    offset: int
    data: memoryview


@dataclass(frozen=True)
class SectionedAsset:
    magic: bytes
    version: int
    source_sha256: bytes
    payload_sha256: bytes
    bytes: bytes
    sections: Mapping[str, SectionView]

    def require(self, name: str, *, count: int | None = None, stride: int | None = None) -> SectionView:
        if name not in self.sections:
            raise FormatError(f"missing required section {name!r}")
        section = self.sections[name]
        if count is not None and section.count != count:
            raise FormatError(f"section {name} count is {section.count}, expected {count}")
        if stride is not None and section.stride != stride:
            raise FormatError(f"section {name} stride is {section.stride}, expected {stride}")
        return section


def write_sectioned(
    path: os.PathLike[str] | str,
    magic: bytes,
    version: int,
    sections: Sequence[Section],
    *,
    source_sha256: bytes | str | None = None,
) -> dict:
    if len(magic) != 8:
        raise FormatError("sectioned magic must be exactly 8 bytes")
    if version <= 0 or not sections:
        raise FormatError("version and section list must be non-zero")
    names = [section.name for section in sections]
    if len(names) != len(set(names)):
        raise FormatError("section names must be unique")

    if source_sha256 is None:
        source_digest = bytes(32)
    elif isinstance(source_sha256, str):
        source_digest = bytes.fromhex(source_sha256)
    else:
        source_digest = source_sha256
    if len(source_digest) != 32:
        raise FormatError("source SHA-256 must be 32 bytes")

    payload_offset = _align_up(SECTION_HEADER.size + SECTION_ENTRY.size * len(sections))
    output = bytearray(payload_offset)
    entries: list[tuple[Section, int]] = []
    for section in sections:
        offset = _align_up(len(output))
        output.extend(bytes(offset - len(output)))
        entries.append((section, offset))
        output.extend(section.data)

    payload_digest = hashlib.sha256(output[payload_offset:]).digest()
    output[: SECTION_HEADER.size] = SECTION_HEADER.pack(
        magic,
        version,
        len(sections),
        len(output),
        payload_offset,
        payload_digest,
        source_digest,
    )
    cursor = SECTION_HEADER.size
    for section, offset in entries:
        name = section.name.encode("ascii")
        output[cursor : cursor + SECTION_ENTRY.size] = SECTION_ENTRY.pack(
            name + bytes(16 - len(name)), offset, section.count, section.stride
        )
        cursor += SECTION_ENTRY.size

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    load_sectioned(output_path, expected_magic=magic, expected_version=version)
    return {
        "magic": magic.decode("ascii"),
        "version": version,
        "file_bytes": len(output),
        "file_sha256": hashlib.sha256(output).hexdigest(),
        "payload_sha256": payload_digest.hex(),
        "source_sha256": source_digest.hex(),
        "sections": [
            {"name": section.name, "offset": offset, "count": section.count, "stride": section.stride}
            for section, offset in entries
        ],
    }


def load_sectioned(
    path: os.PathLike[str] | str,
    *,
    expected_magic: bytes | None = None,
    expected_version: int | None = None,
    required_sections: Iterable[str] = (),
) -> SectionedAsset:
    blob = Path(path).read_bytes()
    if len(blob) < SECTION_HEADER.size:
        raise FormatError("sectioned file is shorter than its header")
    magic, version, section_count, file_size, payload_offset, payload_hash, source_hash = SECTION_HEADER.unpack_from(blob)
    if expected_magic is not None and magic != expected_magic:
        raise FormatError(f"invalid magic {magic!r}; expected {expected_magic!r}")
    if expected_version is not None and version != expected_version:
        raise FormatError(f"unsupported version {version}; expected {expected_version}")
    directory_end = SECTION_HEADER.size + section_count * SECTION_ENTRY.size
    if section_count == 0 or file_size != len(blob):
        raise FormatError("invalid section count or declared file size")
    if payload_offset != _align_up(directory_end) or payload_offset > len(blob):
        raise FormatError("invalid payload offset")
    if hashlib.sha256(blob[payload_offset:]).digest() != payload_hash:
        raise FormatError("payload SHA-256 mismatch")

    sections: dict[str, SectionView] = {}
    ranges: list[tuple[int, int, str]] = []
    for index in range(section_count):
        raw_name, offset, count, stride = SECTION_ENTRY.unpack_from(blob, SECTION_HEADER.size + index * SECTION_ENTRY.size)
        if b"\0" in raw_name:
            raw_name = raw_name[: raw_name.index(b"\0")]
        try:
            name = raw_name.decode("ascii")
        except UnicodeDecodeError as exc:
            raise FormatError("section name is not ASCII") from exc
        if not name or name in sections or stride == 0:
            raise FormatError("empty/duplicate section name or zero stride")
        end = offset + count * stride
        if offset < payload_offset or offset % ALIGNMENT or end > len(blob):
            raise FormatError(f"invalid byte range for section {name}")
        sections[name] = SectionView(name, count, stride, offset, memoryview(blob)[offset:end])
        ranges.append((offset, end, name))
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if previous[1] > current[0]:
            raise FormatError(f"overlapping sections {previous[2]} and {current[2]}")
    missing = sorted(set(required_sections) - set(sections))
    if missing:
        raise FormatError(f"missing required sections: {', '.join(missing)}")
    return SectionedAsset(magic, version, source_hash, payload_hash, blob, sections)


def load_vcloth_v1(path: os.PathLike[str] | str) -> SectionedAsset:
    """Read the already-generated VCLTH v1 container without weakening checks."""

    blob = Path(path).read_bytes()
    if len(blob) < VCLOTH1_HEADER.size:
        raise FormatError("VCLTH v1 is shorter than its header")
    magic, version, section_count, file_size, payload_hash, reserved = VCLOTH1_HEADER.unpack_from(blob)
    if magic != b"VCLTH001" or version != 1 or file_size != len(blob) or any(reserved):
        raise FormatError("invalid VCLTH v1 header")
    payload_offset = _align_up(VCLOTH1_HEADER.size + section_count * SECTION_ENTRY.size)
    if hashlib.sha256(blob[payload_offset:]).digest() != payload_hash:
        raise FormatError("VCLTH v1 payload SHA-256 mismatch")
    sections: dict[str, SectionView] = {}
    ranges: list[tuple[int, int, str]] = []
    for index in range(section_count):
        raw_name, offset, count, stride = SECTION_ENTRY.unpack_from(blob, VCLOTH1_HEADER.size + index * SECTION_ENTRY.size)
        name = raw_name.split(b"\0", 1)[0].decode("ascii")
        end = offset + count * stride
        if not name or name in sections or stride == 0 or offset % ALIGNMENT or offset < payload_offset or end > len(blob):
            raise FormatError("invalid VCLTH v1 section directory")
        sections[name] = SectionView(name, count, stride, offset, memoryview(blob)[offset:end])
        ranges.append((offset, end, name))
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if previous[1] > current[0]:
            raise FormatError("overlapping VCLTH v1 sections")
    return SectionedAsset(magic, version, bytes(32), payload_hash, blob, sections)


@dataclass(frozen=True)
class TensorView:
    name: str
    shape: tuple[int, ...]
    offset: int
    data: memoryview


@dataclass(frozen=True)
class TensorAsset:
    version: int
    checkpoint_sha256: bytes
    payload_sha256: bytes
    bytes: bytes
    tensors: Mapping[str, TensorView]

    def require(self, name: str, shape: Sequence[int] | None = None) -> TensorView:
        if name not in self.tensors:
            raise FormatError(f"missing tensor {name!r}")
        tensor = self.tensors[name]
        if shape is not None and tuple(shape) != tensor.shape:
            raise FormatError(f"tensor {name} shape is {tensor.shape}, expected {tuple(shape)}")
        return tensor


def write_tensor_asset(
    path: os.PathLike[str] | str,
    tensors: Mapping[str, tuple[Sequence[int], bytes]],
    *,
    checkpoint_sha256: bytes | str,
    version: int = 1,
) -> dict:
    if not tensors:
        raise FormatError("tensor asset cannot be empty")
    checkpoint_digest = bytes.fromhex(checkpoint_sha256) if isinstance(checkpoint_sha256, str) else checkpoint_sha256
    if len(checkpoint_digest) != 32:
        raise FormatError("checkpoint SHA-256 must be 32 bytes")
    payload_offset = _align_up(TENSOR_HEADER.size + TENSOR_ENTRY.size * len(tensors))
    output = bytearray(payload_offset)
    entries: list[tuple[str, tuple[int, ...], int, int]] = []
    for name in sorted(tensors):
        try:
            encoded_name = name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise FormatError(f"tensor name is not ASCII: {name!r}") from exc
        if not encoded_name or len(encoded_name) >= 160:
            raise FormatError(f"tensor name exceeds 159 ASCII bytes: {name!r}")
        shape, data = tensors[name]
        shape_tuple = tuple(int(value) for value in shape)
        if not shape_tuple or len(shape_tuple) > 8 or any(value <= 0 for value in shape_tuple):
            raise FormatError(f"invalid tensor shape for {name}: {shape_tuple}")
        count = 1
        for dimension in shape_tuple:
            count *= dimension
        if len(data) != count * 4:
            raise FormatError(f"tensor {name} is not packed FP32")
        offset = _align_up(len(output))
        output.extend(bytes(offset - len(output)))
        output.extend(data)
        entries.append((name, shape_tuple, offset, count))

    payload_hash = hashlib.sha256(output[payload_offset:]).digest()
    output[: TENSOR_HEADER.size] = TENSOR_HEADER.pack(
        b"VHOOD001", version, len(entries), len(output), payload_offset, payload_hash, checkpoint_digest
    )
    cursor = TENSOR_HEADER.size
    for name, shape, offset, count in entries:
        encoded = name.encode("ascii")
        dims = (*shape, *(0 for _ in range(8 - len(shape))))
        output[cursor : cursor + TENSOR_ENTRY.size] = TENSOR_ENTRY.pack(
            encoded + bytes(160 - len(encoded)), offset, count, len(shape), *dims
        )
        cursor += TENSOR_ENTRY.size
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    load_tensor_asset(output_path)
    return {
        "magic": "VHOOD001",
        "version": version,
        "tensor_count": len(entries),
        "file_bytes": len(output),
        "file_sha256": hashlib.sha256(output).hexdigest(),
        "payload_sha256": payload_hash.hex(),
        "checkpoint_sha256": checkpoint_digest.hex(),
        "tensors": [{"name": name, "shape": list(shape), "offset": offset} for name, shape, offset, _ in entries],
    }


def load_tensor_asset(path: os.PathLike[str] | str, *, expected_version: int = 1) -> TensorAsset:
    blob = Path(path).read_bytes()
    if len(blob) < TENSOR_HEADER.size:
        raise FormatError("VHOOD file is shorter than its header")
    magic, version, tensor_count, file_size, payload_offset, payload_hash, checkpoint_hash = TENSOR_HEADER.unpack_from(blob)
    if magic != b"VHOOD001" or version != expected_version:
        raise FormatError("invalid VHOOD magic or version")
    directory_end = TENSOR_HEADER.size + tensor_count * TENSOR_ENTRY.size
    if tensor_count == 0 or file_size != len(blob) or payload_offset != _align_up(directory_end):
        raise FormatError("invalid VHOOD directory declaration")
    if hashlib.sha256(blob[payload_offset:]).digest() != payload_hash:
        raise FormatError("VHOOD payload SHA-256 mismatch")
    tensors: dict[str, TensorView] = {}
    ranges: list[tuple[int, int, str]] = []
    for index in range(tensor_count):
        values = TENSOR_ENTRY.unpack_from(blob, TENSOR_HEADER.size + index * TENSOR_ENTRY.size)
        raw_name, offset, count, rank, *dims = values
        name = raw_name.split(b"\0", 1)[0].decode("ascii")
        if not name or name in tensors or not 1 <= rank <= 8:
            raise FormatError("invalid VHOOD tensor name or rank")
        shape = tuple(dims[:rank])
        expected_count = 1
        for dimension in shape:
            expected_count *= dimension
        end = offset + count * 4
        if count != expected_count or offset < payload_offset or offset % ALIGNMENT or end > len(blob):
            raise FormatError(f"invalid VHOOD tensor range for {name}")
        tensors[name] = TensorView(name, shape, offset, memoryview(blob)[offset:end])
        ranges.append((offset, end, name))
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if previous[1] > current[0]:
            raise FormatError(f"overlapping VHOOD tensors {previous[2]} and {current[2]}")
    return TensorAsset(version, checkpoint_hash, payload_hash, blob, tensors)


def pack_f32(rows: Iterable[Iterable[float]] | Iterable[float]) -> bytes:
    flat: list[float] = []
    for value in rows:
        if isinstance(value, (list, tuple)):
            flat.extend(float(component) for component in value)
        else:
            try:
                flat.extend(float(component) for component in value)  # type: ignore[arg-type]
            except TypeError:
                flat.append(float(value))  # type: ignore[arg-type]
    return struct.pack(f"<{len(flat)}f", *flat)


def pack_u32(rows: Iterable[Iterable[int]] | Iterable[int]) -> bytes:
    flat: list[int] = []
    for value in rows:
        if isinstance(value, (list, tuple)):
            flat.extend(int(component) for component in value)
        else:
            try:
                flat.extend(int(component) for component in value)  # type: ignore[arg-type]
            except TypeError:
                flat.append(int(value))  # type: ignore[arg-type]
    return struct.pack(f"<{len(flat)}I", *flat)
