"""Bounded preprocessing for temporal v4 training data.

Reads existing demo assets, writes .work/temporal_v4/preprocessed/asset.npz + metadata.json:
static rest mesh graph (directed receiver-sorted edges/CSR), masses, pins,
fixed body surface sampling (STM vertex one-hot barycentrics) and capsule
40-sample bind parameter coordinates, SHA-256 fingerprints of all source assets.

Run: python tools/temporal_preprocess.py [--manifest PATH] [--output DIR]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / '.work' / 'temporal_v4' / 'preprocessed'

CLIPS = ['idle', 'walk', 'run', 'sprint', 'jump', 'turn', 'complex']
TRAIN_CLIPS = ['idle', 'walk', 'run', 'sprint', 'jump']
VAL_CLIPS = ['turn']
TEST_CLIPS = ['complex']
SPEEDS = [0.75, 1.0, 1.5]


def read_dmpack(path: Path):
    blob = path.read_bytes()
    if len(blob) < 16 or blob[:8] != b'DMPACK01':
        raise ValueError('Invalid DMPACK magic: {}'.format(path))
    length = struct.unpack_from('<Q', blob, 8)[0]
    if length < 0 or length > len(blob) - 16 or length > 16 * 1024 * 1024:
        raise ValueError('Invalid DMPACK header length: {}'.format(path))
    meta = json.loads(blob[16:16 + length])
    payload = blob[16 + length:]
    arrays = {}
    used_ranges = []
    for key, entry in meta.get('arrays', {}).items():
        dt_str = entry.get('dtype', '')
        if dt_str not in ('<f4', '<u4', '<i4'):
            raise ValueError('Unsupported dtype {} for array {}'.format(dt_str, key))
        dt = np.dtype(dt_str)
        offset = entry['offset']
        nbytes = entry['bytes']
        shape = entry['shape']
        if not isinstance(offset,int) or not isinstance(nbytes,int) or offset < 0 or nbytes < 0 or nbytes % dt.itemsize:
            raise ValueError('Negative offset/bytes for array {}'.format(key))
        if offset + nbytes > len(payload):
            raise ValueError('Array {} extends beyond payload (offset={}, bytes={}, payload={})'.format(
                key, offset, nbytes, len(payload)))
        expected = 1
        for s in shape:
            if not isinstance(s,int) or s < 0:
                raise ValueError('Non-positive dimension in array {}'.format(key))
            expected *= s
        count = nbytes // dt.itemsize
        if expected != count:
            raise ValueError('Shape {} does not match element count {} for array {}'.format(shape, count, key))
        arr = np.frombuffer(payload, dtype=dt, count=count, offset=offset).reshape(shape).copy()
        if dt.kind == 'f' and not np.isfinite(arr).all():
            raise ValueError('Non-finite values in float array {}'.format(key))
        used_ranges.append((offset, offset + nbytes))
        arrays[key] = arr
    used_ranges.sort()
    for i in range(len(used_ranges) - 1):
        if used_ranges[i][1] > used_ranges[i + 1][0]:
            raise ValueError('Overlapping arrays in DMPACK')
    return meta, arrays


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def fingerprint_sources(manifest, directory, manifest_path=None):
    hashes = {}
    hashes['manifest'] = file_sha256(manifest_path) if manifest_path else sha256_hex(json.dumps(manifest,sort_keys=True).encode())
    for key in ['cloth', 'collision', 'body']:
        fname = manifest.get(key)
        if fname:
            p = directory / fname
            hashes[key] = file_sha256(p)
    if manifest.get('capsules'):
        p = directory / manifest['capsules']
        hashes['capsules'] = file_sha256(p)
    for clip in manifest.get('clips', []):
        anim = clip.get('animation', '')
        p = directory / anim
        hashes['animation_{}'.format(clip['id'])] = file_sha256(p)
    return hashes


def build_graph(triangles: np.ndarray, vertex_count: int):
    tri = triangles.reshape(-1, 3)
    directed = set()
    for a, b, c in tri:
        directed.add((int(a), int(b)))
        directed.add((int(b), int(c)))
        directed.add((int(c), int(a)))
        directed.add((int(b), int(a)))
        directed.add((int(c), int(b)))
        directed.add((int(a), int(c)))
    directed = sorted(directed, key=lambda e: (e[1], e[0]))
    senders = np.array([e[0] for e in directed], dtype=np.uint32)
    receivers = np.array([e[1] for e in directed], dtype=np.uint32)
    offsets = np.zeros(vertex_count + 1, dtype=np.uint32)
    for r in receivers:
        offsets[r + 1] += 1
    offsets = np.cumsum(offsets, dtype=np.uint32)
    return senders, receivers, offsets


def sample_body_surface(stm_positions, stm_triangles):
    n = len(stm_positions)
    tris = stm_triangles.reshape(-1, 3)
    if (tris<0).any() or (tris>=n).any():raise ValueError('STM triangle index out of range')
    incident_tri = np.full(n, -1, dtype=np.int32)
    for ti in range(len(tris)):
        a, b, c = int(tris[ti, 0]), int(tris[ti, 1]), int(tris[ti, 2])
        for v in (a, b, c):
            if incident_tri[v] < 0:
                incident_tri[v] = ti
    isolated = np.where(incident_tri < 0)[0]
    if len(isolated) > 0:
        raise ValueError('Isolated STM vertices (no incident triangle): {}'.format(isolated.tolist()[:10]))
    tri_indices = tris[incident_tri].astype(np.uint32)
    bary = np.zeros((n, 3), dtype=np.float32)
    for v in range(n):
        ia, ib, ic = int(tris[incident_tri[v], 0]), int(tris[incident_tri[v], 1]), int(tris[incident_tri[v], 2])
        if v == ia:
            bary[v] = [1.0, 0.0, 0.0]
        elif v == ib:
            bary[v] = [0.0, 1.0, 0.0]
        else:
            bary[v] = [0.0, 0.0, 1.0]
    stable_id = np.arange(n, dtype=np.uint32)
    return tri_indices, bary, stable_id


def compute_capsule_samples(capsules_arrays):
    bones = capsules_arrays['bone']
    a_arr = capsules_arrays['a'].reshape(-1, 3)
    b_arr = capsules_arrays['b'].reshape(-1, 3)
    radii = capsules_arrays['radius']
    nc = len(bones)
    if a_arr.shape!=(nc,3) or b_arr.shape!=(nc,3) or radii.shape!=(nc,) or not np.isfinite(radii).all() or (radii<=0).any():raise ValueError('Invalid capsule binding shapes/radius')
    samples_per_capsule = 40
    total = nc * samples_per_capsule

    capsule_index = np.zeros(total, dtype=np.uint32)
    sample_t = np.zeros(total, dtype=np.float32)
    sample_angle = np.zeros(total, dtype=np.float32)
    surface_u = np.zeros((total, 3), dtype=np.float32)
    surface_v = np.zeros((total, 3), dtype=np.float32)
    stable_id = np.zeros(total, dtype=np.uint64)

    idx = 0
    for ci in range(nc):
        ca = a_arr[ci]
        cb = b_arr[ci]
        axis = cb - ca
        ax_len = float(np.linalg.norm(axis))
        if ax_len < 1e-10:
            raise ValueError('Zero-length capsule axis for capsule {}'.format(ci))
        axis_norm = axis / ax_len
        seed = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(axis_norm[0]) < 0.8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        u = np.cross(axis_norm, seed)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-10:
            seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            u = np.cross(axis_norm, seed)
            u_norm = np.linalg.norm(u)
        u /= u_norm
        v = np.cross(axis_norm, u)
        v /= max(np.linalg.norm(v), 1e-10)

        for ring in range(5):
            t = ring / 4.0
            for k in range(8):
                angle = k * 2.0 * math.pi / 8.0
                capsule_index[idx] = ci
                sample_t[idx] = t
                sample_angle[idx] = angle
                surface_u[idx] = u
                surface_v[idx] = v
                stable_id[idx] = (np.uint64(1) << np.uint64(63)) | np.uint64(ci * 40 + ring * 8 + k)
                idx += 1

    return capsule_index, sample_t, sample_angle, surface_u, surface_v, stable_id


def validate_mesh(triangles, rest, mass, pinned):
    n = len(rest)
    if rest.shape!=(n,3) or mass.shape!=(n,):raise ValueError('Rest/mass shape mismatch')
    tri = triangles.reshape(-1, 3)
    if len(tri) == 0:
        raise ValueError('Empty triangle array')
    if int(tri.max()) >= n:
        raise ValueError('Triangle index {} out of range (vertices={})'.format(int(tri.max()), n))
    if int(tri.min()) < 0:
        raise ValueError('Negative triangle index')
    areas = []
    for i in range(len(tri)):
        a, b, c = rest[tri[i]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        areas.append(area)
        if area < 1e-12:
            raise ValueError('Degenerate triangle {}: area={}'.format(i, area))
    if not np.isfinite(rest).all():
        raise ValueError('Non-finite rest positions')
    if not np.isfinite(mass).all():
        raise ValueError('Non-finite masses')
    if np.any(mass <= 0):
        raise ValueError('Non-positive mass')
    if pinned.ndim != 1 or len(pinned) != n:
        raise ValueError('Pinned array shape mismatch')
    if not np.all((pinned == 0) | (pinned == 1)):
        raise ValueError('Pinned values must be 0 or 1')
    print('  Validated: {} vertices, {} triangles'.format(n, len(tri)))


def make_split_records(manifest):
    records = []
    for clip_entry in manifest['clips']:
        clip_id = clip_entry['id']
        if clip_id not in CLIPS:
            continue
        for speed in SPEEDS:
            for in_place in [False, True]:
                records.append({
                    'clip': clip_id,
                    'speed': speed,
                    'in_place': in_place,
                    'duration': clip_entry['duration'] / speed,
                    'note': 'in_place is display offset only; physical trajectory unchanged',
                })
    train = [r for r in records if r['clip'] in TRAIN_CLIPS]
    val = [r for r in records if r['clip'] in VAL_CLIPS]
    test = [r for r in records if r['clip'] in TEST_CLIPS]
    return {'train': train, 'validation': val, 'test': test}


def main():
    parser = argparse.ArgumentParser(description='Preprocess demo assets for temporal v4')
    parser.add_argument('--manifest', type=Path, default=ROOT / '.work' / 'demo' / 'demo.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        print('Manifest not found: {}'.format(manifest_path), file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    directory = manifest_path.parent
    output_dir = args.output.resolve()

    print('Fingerprinting source assets...')
    source_hashes = fingerprint_sources(manifest, directory, manifest_path)
    for k, v in source_hashes.items():
        print('  {}: {}...'.format(k, v[:16]))

    print('Reading cloth asset...')
    cloth_meta, cloth = read_dmpack(directory / manifest['cloth'])
    rest = cloth['positions'].astype(np.float32)
    triangles = cloth['triangles'].astype(np.uint32).reshape(-1,3)
    mass = cloth['mass'].astype(np.float32)
    pinned = cloth['pinned'].astype(np.uint32)
    n = len(rest)
    print('  Cloth: {} vertices, {} triangles'.format(n, len(triangles.reshape(-1, 3))))

    print('Reading collision (STM) asset...')
    coll_meta, coll = read_dmpack(directory / manifest['collision'])
    stm_positions = coll['positions'].astype(np.float32)
    stm_triangles = coll['triangles'].astype(np.uint32).reshape(-1,3)
    n_stm = len(stm_positions)
    print('  STM: {} vertices, {} triangles'.format(n_stm, len(stm_triangles.reshape(-1, 3))))

    capsules_arrays = None
    if manifest.get('capsules'):
        print('Reading capsule asset...')
        _, capsules_arrays = read_dmpack(directory / manifest['capsules'])
        print('  Capsules: {}'.format(len(capsules_arrays['bone'])))

    print('Building graph topology...')
    senders, receivers, offsets = build_graph(triangles, n)
    edge_count = len(senders)
    print('  Graph: {} directed edges'.format(edge_count))

    print('Sampling body surface (STM one-hot barycentrics)...')
    stm_bind_tri, stm_bind_bary, stm_stable_id = sample_body_surface(stm_positions, stm_triangles)
    print('  STM bindings: {} samples'.format(len(stm_stable_id)))

    cap_arrays = None
    if capsules_arrays is not None:
        print('Computing capsule 40-sample bindings...')
        cap_arrays = compute_capsule_samples(capsules_arrays)
        cap_idx, cap_t, cap_ang, cap_u, cap_v, cap_sid = cap_arrays
        print('  Capsule samples: {} ({} capsules x 40)'.format(len(cap_sid), len(capsules_arrays['bone'])))

    print('Validating cloth mesh...')
    validate_mesh(triangles, rest, mass, pinned)

    print('Creating split records...')
    splits = make_split_records(manifest)
    print('  Train: {}, Validation: {}, Test: {}'.format(
        len(splits['train']), len(splits['validation']), len(splits['test'])))

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_dict = {
        'cloth_rest': rest,
        'cloth_triangles': triangles,
        'cloth_mass': mass,
        'cloth_pinned': pinned,
        'graph_senders': senders,
        'graph_receivers': receivers,
        'graph_offsets': offsets,
        'stm_positions': stm_positions,
        'stm_triangles': stm_triangles,
        'stm_binding_triangle': stm_bind_tri,
        'stm_binding_bary': stm_bind_bary,
        'stm_stable_id': stm_stable_id,
    }
    if cap_arrays is not None:
        npz_dict['capsule_sample_capsule_index'] = cap_idx
        npz_dict['capsule_sample_t'] = cap_t
        npz_dict['capsule_sample_angle'] = cap_ang
        npz_dict['capsule_sample_surface_u'] = cap_u
        npz_dict['capsule_sample_surface_v'] = cap_v
        npz_dict['capsule_sample_stable_id'] = cap_sid

    npz_path = output_dir / 'asset.npz'
    np.savez_compressed(str(npz_path), **npz_dict)
    artifact_sha = file_sha256(npz_path)
    print('  Written: {} (SHA-256: {}...)'.format(npz_path, artifact_sha[:16]))

    meta = {
        'version': 2,
        'cloth_vertex_count': int(n),
        'cloth_triangle_count': int(len(triangles.reshape(-1, 3))),
        'stm_vertex_count': int(n_stm),
        'stm_triangle_count': int(len(stm_triangles.reshape(-1, 3))),
        'graph_edge_count': int(edge_count),
        'capsule_count': int(len(capsules_arrays['bone'])) if capsules_arrays is not None else 0,
        'capsule_samples_per_capsule': 40,
        'source_hashes': source_hashes,
        'artifact_sha256': artifact_sha,
        'asset_file': 'asset.npz',
        'split': splits,
        'all_sequences': splits['train'] + splits['validation'] + splits['test'],
        'array_contract': {
            'cloth_rest': {'shape': [int(n), 3], 'dtype': 'float32'},
            'cloth_triangles': {'shape': [int(len(triangles.reshape(-1, 3))), 3], 'dtype': 'uint32'},
            'cloth_mass': {'shape': [int(n)], 'dtype': 'float32'},
            'cloth_pinned': {'shape': [int(n)], 'dtype': 'uint32'},
            'graph_senders': {'shape': [int(edge_count)], 'dtype': 'uint32'},
            'graph_receivers': {'shape': [int(edge_count)], 'dtype': 'uint32'},
            'graph_offsets': {'shape': [int(n) + 1], 'dtype': 'uint32'},
            'stm_positions': {'shape': [int(n_stm), 3], 'dtype': 'float32'},
            'stm_triangles': {'shape': [int(len(stm_triangles.reshape(-1, 3))), 3], 'dtype': 'uint32'},
            'stm_binding_triangle': {'shape': [int(n_stm), 3], 'dtype': 'uint32'},
            'stm_binding_bary': {'shape': [int(n_stm), 3], 'dtype': 'float32'},
            'stm_stable_id': {'shape': [int(n_stm)], 'dtype': 'uint32'},
        },
    }
    if cap_arrays is not None:
        nc = int(len(capsules_arrays['bone']))
        total_cap = nc * 40
        meta['array_contract'].update({
            'capsule_sample_capsule_index': {'shape': [total_cap], 'dtype': 'uint32'},
            'capsule_sample_t': {'shape': [total_cap], 'dtype': 'float32'},
            'capsule_sample_angle': {'shape': [total_cap], 'dtype': 'float32'},
            'capsule_sample_surface_u': {'shape': [total_cap, 3], 'dtype': 'float32'},
            'capsule_sample_surface_v': {'shape': [total_cap, 3], 'dtype': 'float32'},
            'capsule_sample_stable_id': {'shape': [total_cap], 'dtype': 'uint64'},
        })
    meta['root_motion_note'] = 'in_place is display offset only; physical trajectory is unchanged'

    (output_dir / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
