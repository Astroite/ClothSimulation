"""Unit tests for temporal preprocessing.

Run: python -m unittest tests.test_temporal_data -v
"""
from __future__ import annotations
import json
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

from temporal_preprocess import (
    build_graph, sample_body_surface, compute_capsule_samples,
    validate_mesh, make_split_records, read_dmpack, fingerprint_sources,
    CLIPS, TRAIN_CLIPS, VAL_CLIPS, TEST_CLIPS, SPEEDS, main as preprocess_main,
)

DEMO_DIR = ROOT / '.work' / 'demo'
MANIFEST = DEMO_DIR / 'demo.json'


def make_grid_mesh(rows=4, cols=5, spacing=0.01):
    rest = []
    for r in range(rows):
        for c in range(cols):
            rest.append([c * spacing, r * spacing, 0.0])
    rest = np.array(rest, dtype=np.float32)
    triangles = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = r * cols + c
            b = a + 1
            cc = a + cols
            d = cc + 1
            triangles.extend([a, b, d, a, d, cc])
    triangles = np.array(triangles, dtype=np.uint32)
    n = len(rest)
    mass = np.full(n, 0.02, dtype=np.float32)
    pinned = np.zeros(n, dtype=np.uint32)
    pinned[:cols] = 1
    return rest, triangles, mass, pinned


class TestGraphConstruction(unittest.TestCase):
    def test_directed_edge_count(self):
        rest, tri, mass, pinned = make_grid_mesh(3, 4)
        n = len(rest)
        senders, receivers, offsets = build_graph(tri, n)
        edge_set = set()
        for a, b, c in tri.reshape(-1, 3):
            for u, v in [(a, b), (b, c), (c, a)]:
                edge_set.add((min(int(u), int(v)), max(int(u), int(v))))
        self.assertEqual(len(senders), len(edge_set) * 2)

    def test_csr_consistency(self):
        rest, tri, mass, pinned = make_grid_mesh(3, 4)
        n = len(rest)
        senders, receivers, offsets = build_graph(tri, n)
        self.assertEqual(len(offsets), n + 1)
        for r in range(n):
            start, end = int(offsets[r]), int(offsets[r + 1])
            for idx in range(start, end):
                self.assertEqual(int(receivers[idx]), r)
        self.assertEqual(int(offsets[-1]), len(senders))

    def test_sorted_by_receiver(self):
        rest, tri, mass, pinned = make_grid_mesh(4, 5)
        n = len(rest)
        senders, receivers, offsets = build_graph(tri, n)
        for r in range(n):
            start, end = int(offsets[r]), int(offsets[r + 1])
            if end - start > 1:
                for i in range(start + 1, end):
                    self.assertEqual(int(receivers[i]), r)

    def test_no_self_loops(self):
        rest, tri, mass, pinned = make_grid_mesh(3, 4)
        senders, receivers, _ = build_graph(tri, len(rest))
        for s, r in zip(senders, receivers):
            self.assertNotEqual(int(s), int(r))


class TestBodySurfaceSampling(unittest.TestCase):
    def test_one_hot_barycentric(self):
        n = 10
        positions = np.random.randn(n, 3).astype(np.float32)
        triangles = np.array([[i, (i + 1) % n, (i + 2) % n] for i in range(n)], dtype=np.uint32)
        tri_idx, bary, stable_id = sample_body_surface(positions, triangles)
        for v in range(n):
            row = bary[v]
            ones = np.sum(np.isclose(row, 1.0))
            zeros = np.sum(np.isclose(row, 0.0))
            self.assertEqual(ones, 1, 'Vertex {} bary {}'.format(v, row))
            self.assertEqual(zeros, 2, 'Vertex {} bary {}'.format(v, row))
            self.assertAlmostEqual(float(row.sum()), 1.0, places=5)

    def test_stable_id_is_vertex_index(self):
        n = 8
        positions = np.random.randn(n, 3).astype(np.float32)
        triangles = np.array([[i, (i + 1) % n, (i + 2) % n] for i in range(n)], dtype=np.uint32)
        _, _, stable_id = sample_body_surface(positions, triangles)
        np.testing.assert_array_equal(stable_id, np.arange(n, dtype=np.uint32))

    def test_isolated_vertex_rejected(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [99, 99, 99]], dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype=np.uint32)
        with self.assertRaises(ValueError) as ctx:
            sample_body_surface(positions, triangles)
        self.assertIn('Isolated', str(ctx.exception))

    def test_triangle_indices_in_range(self):
        n = 12
        positions = np.random.randn(n, 3).astype(np.float32)
        triangles = np.array([[i, (i + 1) % n, (i + 2) % n] for i in range(n)], dtype=np.uint32)
        tri_idx, _, _ = sample_body_surface(positions, triangles)
        self.assertTrue((tri_idx < n).all())


class TestCapsuleSamples(unittest.TestCase):
    def test_sample_count(self):
        cap = {
            'bone': np.array([0, 1], dtype=np.uint32),
            'a': np.array([[0, 0, 0], [0.1, 0, 0]], dtype=np.float32),
            'b': np.array([[0, 0, 0.1], [0.1, 0, 0.1]], dtype=np.float32),
            'radius': np.array([0.02, 0.03], dtype=np.float32),
        }
        idx, t, ang, u, v, sid = compute_capsule_samples(cap)
        self.assertEqual(len(idx), 80)

    def test_stable_id_highbit(self):
        cap = {
            'bone': np.array([0], dtype=np.uint32),
            'a': np.array([[0, 0, 0]], dtype=np.float32),
            'b': np.array([[0, 0, 0.1]], dtype=np.float32),
            'radius': np.array([0.02], dtype=np.float32),
        }
        _, _, _, _, _, sid = compute_capsule_samples(cap)
        for s in sid:
            self.assertTrue(int(s) & (1 << 63), 'High bit not set for {}'.format(s))
            raw = int(s) & ((1 << 63) - 1)
            self.assertLess(raw, 40)

    def test_param_t_range(self):
        cap = {
            'bone': np.array([0], dtype=np.uint32),
            'a': np.array([[0, 0, 0]], dtype=np.float32),
            'b': np.array([[0, 0, 0.1]], dtype=np.float32),
            'radius': np.array([0.02], dtype=np.float32),
        }
        _, t, ang, _, _, _ = compute_capsule_samples(cap)
        self.assertTrue((t >= 0).all())
        self.assertTrue((t <= 1).all())
        np.testing.assert_allclose(np.sort(np.unique(t)), [0, 0.25, 0.5, 0.75, 1.0])

    def test_u_v_orthogonal(self):
        cap = {
            'bone': np.array([0], dtype=np.uint32),
            'a': np.array([[0, 0, 0]], dtype=np.float32),
            'b': np.array([[0, 0, 0.1]], dtype=np.float32),
            'radius': np.array([0.02], dtype=np.float32),
        }
        _, _, _, u, v, _ = compute_capsule_samples(cap)
        for i in range(len(u)):
            dot = float(np.dot(u[i], v[i]))
            self.assertAlmostEqual(dot, 0.0, places=5)
            self.assertAlmostEqual(float(np.linalg.norm(u[i])), 1.0, places=5)
            self.assertAlmostEqual(float(np.linalg.norm(v[i])), 1.0, places=5)

    def test_unique_ids(self):
        cap = {
            'bone': np.array([0, 1, 2], dtype=np.uint32),
            'a': np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], dtype=np.float32),
            'b': np.array([[0, 0, 0.1], [0.1, 0, 0.1], [0.2, 0, 0.1]], dtype=np.float32),
            'radius': np.array([0.02, 0.02, 0.02], dtype=np.float32),
        }
        _, _, _, _, _, sid = compute_capsule_samples(cap)
        self.assertEqual(len(set(sid.tolist())), len(sid))


class TestMeshValidation(unittest.TestCase):
    def test_valid_mesh_passes(self):
        rest, tri, mass, pinned = make_grid_mesh(3, 4)
        validate_mesh(tri, rest, mass, pinned)

    def test_degenerate_triangle_fails(self):
        rest, tri, mass, pinned = make_grid_mesh(3, 4)
        bad_tri = tri.copy()
        bad_tri[0] = bad_tri[1]
        bad_tri[2] = bad_tri[1]
        with self.assertRaises(ValueError):
            validate_mesh(bad_tri, rest, mass, pinned)

    def test_out_of_range_index_fails(self):
        rest, tri, mass, pinned = make_grid_mesh(3, 4)
        bad_tri = tri.copy()
        bad_tri[0] = len(rest) + 10
        with self.assertRaises(ValueError):
            validate_mesh(bad_tri, rest, mass, pinned)

    def test_nonfinite_position_fails(self):
        rest, tri, mass, pinned = make_grid_mesh(3, 4)
        bad_rest = rest.copy()
        bad_rest[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            validate_mesh(tri, bad_rest, mass, pinned)

    def test_nonpositive_mass_fails(self):
        rest, tri, mass, pinned = make_grid_mesh(3, 4)
        bad_mass = mass.copy()
        bad_mass[0] = -1.0
        with self.assertRaises(ValueError):
            validate_mesh(tri, rest, bad_mass, pinned)

    def test_bad_pinned_fails(self):
        rest, tri, mass, pinned = make_grid_mesh(3, 4)
        bad_pinned = pinned.copy()
        bad_pinned[0] = 2
        with self.assertRaises(ValueError):
            validate_mesh(tri, rest, mass, bad_pinned)


class TestSplits(unittest.TestCase):
    def test_split_counts(self):
        manifest = {
            'clips': [{'id': c, 'name': c.title(), 'duration': 1.0, 'animation': '{}.dma'.format(c)} for c in CLIPS]
        }
        splits = make_split_records(manifest)
        self.assertEqual(len(splits['train']), len(TRAIN_CLIPS) * len(SPEEDS) * 2)
        self.assertEqual(len(splits['validation']), len(VAL_CLIPS) * len(SPEEDS) * 2)
        self.assertEqual(len(splits['test']), len(TEST_CLIPS) * len(SPEEDS) * 2)

    def test_no_clip_overlap(self):
        manifest = {
            'clips': [{'id': c, 'name': c.title(), 'duration': 1.0, 'animation': '{}.dma'.format(c)} for c in CLIPS]
        }
        splits = make_split_records(manifest)
        train_clips = {r['clip'] for r in splits['train']}
        val_clips = {r['clip'] for r in splits['validation']}
        test_clips = {r['clip'] for r in splits['test']}
        self.assertEqual(len(train_clips & val_clips), 0)
        self.assertEqual(len(train_clips & test_clips), 0)
        self.assertEqual(len(val_clips & test_clips), 0)

    def test_all_speeds_present(self):
        manifest = {
            'clips': [{'id': c, 'name': c.title(), 'duration': 1.0, 'animation': '{}.dma'.format(c)} for c in CLIPS]
        }
        splits = make_split_records(manifest)
        for group in [splits['train'], splits['validation'], splits['test']]:
            speeds = {r['speed'] for r in group}
            self.assertEqual(speeds, set(SPEEDS))

    def test_in_place_note(self):
        manifest = {
            'clips': [{'id': c, 'name': c.title(), 'duration': 1.0, 'animation': '{}.dma'.format(c)} for c in CLIPS]
        }
        splits = make_split_records(manifest)
        for group_name, group in splits.items():
            for r in group:
                self.assertIn('note', r)
                self.assertIn('display offset', r['note'])


class TestDmpackReader(unittest.TestCase):
    def test_read_valid_dmpack(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'test.dmp'
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            payload = arr.tobytes()
            meta = {'version': 1, 'arrays': {'data': {'offset': 0, 'shape': [3], 'dtype': '<f4', 'bytes': 12}}}
            meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')
            while len(payload) % 16:
                payload += b'\0'
            blob = b'DMPACK01' + struct.pack('<Q', len(meta_bytes)) + meta_bytes + payload
            path.write_bytes(blob)
            m, arrays = read_dmpack(path)
            np.testing.assert_allclose(arrays['data'], arr)

    def test_invalid_magic_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.dmp'
            path.write_bytes(b'BADMAGIC' + b'\0' * 100)
            with self.assertRaises(ValueError):
                read_dmpack(path)

    def test_negative_offset_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.dmp'
            arr = np.array([1.0], dtype=np.float32)
            payload = arr.tobytes()
            meta = {'version': 1, 'arrays': {'data': {'offset': -1, 'shape': [1], 'dtype': '<f4', 'bytes': 4}}}
            meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')
            blob = b'DMPACK01' + struct.pack('<Q', len(meta_bytes)) + meta_bytes + payload
            path.write_bytes(blob)
            with self.assertRaises(ValueError):
                read_dmpack(path)

    def test_unsupported_dtype_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.dmp'
            arr = np.array([1.0], dtype=np.float64)
            payload = arr.tobytes()
            meta = {'version': 1, 'arrays': {'data': {'offset': 0, 'shape': [1], 'dtype': '<f8', 'bytes': 8}}}
            meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')
            blob = b'DMPACK01' + struct.pack('<Q', len(meta_bytes)) + meta_bytes + payload
            path.write_bytes(blob)
            with self.assertRaises(ValueError):
                read_dmpack(path)

    def test_shape_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.dmp'
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            payload = arr.tobytes()
            meta = {'version': 1, 'arrays': {'data': {'offset': 0, 'shape': [2, 2], 'dtype': '<f4', 'bytes': 12}}}
            meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')
            blob = b'DMPACK01' + struct.pack('<Q', len(meta_bytes)) + meta_bytes + payload
            path.write_bytes(blob)
            with self.assertRaises(ValueError):
                read_dmpack(path)

    def test_nonfinite_float_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.dmp'
            arr = np.array([1.0, float('nan'), 3.0], dtype=np.float32)
            payload = arr.tobytes()
            meta = {'version': 1, 'arrays': {'data': {'offset': 0, 'shape': [3], 'dtype': '<f4', 'bytes': 12}}}
            meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')
            blob = b'DMPACK01' + struct.pack('<Q', len(meta_bytes)) + meta_bytes + payload
            path.write_bytes(blob)
            with self.assertRaises(ValueError) as ctx:
                read_dmpack(path)
            self.assertIn('Non-finite', str(ctx.exception))


class TestFullPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MANIFEST.exists():
            raise unittest.SkipTest('Demo manifest not found: {}'.format(MANIFEST))
        cls.manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        cls.directory = MANIFEST.parent

    def test_fingerprint_sources(self):
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        hashes = fingerprint_sources(manifest, self.directory)
        self.assertIn('cloth', hashes)
        self.assertIn('collision', hashes)
        self.assertIn('body', hashes)
        self.assertIn('capsules', hashes)
        for clip in self.manifest['clips']:
            key = 'animation_{}'.format(clip['id'])
            self.assertIn(key, hashes, 'Missing hash for {}'.format(key))
        for v in hashes.values():
            self.assertEqual(len(v), 64)

    def test_actual_preprocess_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'preprocessed'
            sys.argv = ['temporal_preprocess', '--manifest', str(MANIFEST), '--output', str(out)]
            rc = preprocess_main()
            self.assertEqual(rc, 0)
            npz_path = out / 'asset.npz'
            meta_path = out / 'metadata.json'
            self.assertTrue(npz_path.exists())
            self.assertTrue(meta_path.exists())
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            self.assertEqual(meta['version'], 2)
            self.assertEqual(meta['cloth_vertex_count'], 5294)
            self.assertEqual(meta['stm_vertex_count'], 1719)
            self.assertEqual(meta['capsule_count'], 10)
            self.assertEqual(meta['capsule_samples_per_capsule'], 40)
            self.assertIn('source_hashes', meta)
            self.assertIn('artifact_sha256', meta)
            self.assertIn('root_motion_note', meta)
            self.assertIn('display offset', meta['root_motion_note'])

    def test_asset_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'preprocessed'
            sys.argv = ['temporal_preprocess', '--manifest', str(MANIFEST), '--output', str(out)]
            preprocess_main()
            data = np.load(str(out / 'asset.npz'))
            try:
                self.assertEqual(data['cloth_rest'].shape, (5294, 3))
                self.assertEqual(data['cloth_mass'].shape, (5294,))
                self.assertEqual(data['cloth_pinned'].shape, (5294,))
                self.assertEqual(data['stm_positions'].shape, (1719, 3))
                self.assertEqual(data['stm_binding_triangle'].shape, (1719, 3))
                self.assertEqual(data['stm_binding_bary'].shape, (1719, 3))
                self.assertEqual(data['stm_stable_id'].shape, (1719,))
                self.assertEqual(data['capsule_sample_capsule_index'].shape, (400,))
                self.assertEqual(data['capsule_sample_t'].shape, (400,))
                self.assertEqual(data['capsule_sample_angle'].shape, (400,))
                self.assertEqual(data['capsule_sample_surface_u'].shape, (400, 3))
                self.assertEqual(data['capsule_sample_surface_v'].shape, (400, 3))
            finally:
                data.close()

    def test_graph_properties(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'preprocessed'
            sys.argv = ['temporal_preprocess', '--manifest', str(MANIFEST), '--output', str(out)]
            preprocess_main()
            data = np.load(str(out / 'asset.npz'))
            try:
                senders = data['graph_senders']
                receivers = data['graph_receivers']
                offsets = data['graph_offsets']
                self.assertEqual(len(offsets), 5295)
                self.assertEqual(int(offsets[-1]), len(senders))
                for r in range(5294):
                    s, e = int(offsets[r]), int(offsets[r + 1])
                    for idx in range(s, e):
                        self.assertEqual(int(receivers[idx]), r)
            finally:
                data.close()

    def test_barycentric_one_hot(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'preprocessed'
            sys.argv = ['temporal_preprocess', '--manifest', str(MANIFEST), '--output', str(out)]
            preprocess_main()
            data = np.load(str(out / 'asset.npz'))
            try:
                bary = data['stm_binding_bary']
                for i in range(len(bary)):
                    row = bary[i]
                    ones = np.sum(np.isclose(row, 1.0))
                    zeros = np.sum(np.isclose(row, 0.0))
                    self.assertEqual(ones, 1, 'STM vertex {} bary {}'.format(i, row))
                    self.assertEqual(zeros, 2)
            finally:
                data.close()

    def test_capsule_stable_ids(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'preprocessed'
            sys.argv = ['temporal_preprocess', '--manifest', str(MANIFEST), '--output', str(out)]
            preprocess_main()
            data = np.load(str(out / 'asset.npz'))
            try:
                sid = data['capsule_sample_stable_id']
                self.assertEqual(len(sid), 400)
                self.assertEqual(len(set(sid.tolist())), 400)
                for s in sid:
                    self.assertTrue(int(s) & (1 << 63))
            finally:
                data.close()

    def test_split_variants(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'preprocessed'
            sys.argv = ['temporal_preprocess', '--manifest', str(MANIFEST), '--output', str(out)]
            preprocess_main()
            meta = json.loads((out / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(len(meta['split']['train']), 30)
            self.assertEqual(len(meta['split']['validation']), 6)
            self.assertEqual(len(meta['split']['test']), 6)


if __name__ == '__main__':
    unittest.main()
