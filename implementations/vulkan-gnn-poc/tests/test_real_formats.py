from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

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


class RealFormatTests(unittest.TestCase):
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
