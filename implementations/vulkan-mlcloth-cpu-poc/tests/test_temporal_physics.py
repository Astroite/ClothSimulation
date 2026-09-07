"""Tests for the PyTorch XPBD reference.

Run: python tests/test_temporal_physics.py [--probe PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temporal.physics import TorchXPBD

_failures: list[str] = []
_tests: int = 0


def _test(name: str):
    global _tests
    _tests += 1
    print(f"  {name} ... ", end="", flush=True)


def _pass():
    print("ok")


def _fail(msg: str):
    global _failures
    _failures.append(msg)
    print(f"FAIL: {msg}")


def _expect(cond: bool, msg: str) -> bool:
    if not cond:
        _fail(msg)
        return False
    return True


def _expect_near(a: float, b: float, eps: float, msg: str) -> bool:
    if abs(a - b) > eps:
        _fail(f"{msg}: got {a}, expected {b} (eps={eps})")
        return False
    return True


def _make_grid():
    rest = torch.tensor([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0],
        [0, 2, 0], [1, 2, 0], [2, 2, 0],
    ], dtype=torch.float32)
    triangles = torch.tensor([
        [0, 1, 3], [1, 4, 3],
        [1, 2, 4], [2, 5, 4],
        [3, 4, 6], [4, 7, 6],
        [4, 5, 7], [5, 8, 7],
    ], dtype=torch.long)
    mass = torch.ones(9)
    pinned = torch.zeros(9, dtype=torch.long)
    return rest, triangles, mass, pinned


def _default_config(**overrides):
    cfg = {
        "iterations": 2,
        "stretchCompliance": 0.0,
        "shearCompliance": 0.0,
        "bendCompliance": 0.0,
        "dampingPerSecond": 0.99,
        "friction": 0.1,
        "thickness": 0.003,
        "gravity": -9.81,
        "enableCollision": False,
        "enableSelfCollision": False,
    }
    cfg.update(overrides)
    return cfg


# --- Basic tests ---

def test_gravity_free_fall():
    _test("gravity free fall")
    rest, tri, mass, pinned = _make_grid()
    config = _default_config(gravity=-9.81, iterations=0, dampingPerSecond=0.0)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    pos = rest.clone()
    vel = torch.zeros_like(pos)
    pins = rest.clone()
    dt = 1.0 / 60.0
    for _ in range(60):
        pos, vel = xpbd.step(pos, vel, dt, pins)
    ok = True
    for i in range(9):
        ok = _expect(pos[i, 1].item() < -0.1, f"vertex {i} should have fallen") and ok
    if ok:
        _pass()


def test_pinned_invariance():
    _test("pinned invariance")
    rest, tri, mass, _ = _make_grid()
    pinned = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.long)
    config = _default_config(gravity=-9.81, iterations=0)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    pos = rest.clone()
    vel = torch.zeros_like(pos)
    pins = rest.clone()
    pos, vel = xpbd.step(pos, vel, 1.0 / 60.0, pins)
    ok = True
    ok = _expect_near(pos[0, 0].item(), 0.0, 1e-6, "pinned 0 x") and ok
    ok = _expect_near(pos[0, 1].item(), 0.0, 1e-6, "pinned 0 y") and ok
    ok = _expect_near(pos[8, 0].item(), 2.0, 1e-6, "pinned 8 x") and ok
    ok = _expect_near(pos[8, 1].item(), 2.0, 1e-6, "pinned 8 y") and ok
    if ok:
        _pass()


def test_damping():
    _test("time-based damping (exp formula)")
    rest, tri, mass, pinned = _make_grid()
    config = _default_config(gravity=0.0, iterations=0, dampingPerSecond=1.0)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    pos = rest.clone()
    vel = torch.ones_like(pos)
    pins = rest.clone()
    pos, vel = xpbd.step(pos, vel, 1.0, pins)
    expected = math.exp(-1.0)
    ok = True
    for i in range(9):
        ok = _expect_near(vel[i, 0].item(), expected, 1e-3, f"damped vx at {i}") and ok
    if ok:
        _pass()


def test_zero_damping_preserves_velocity():
    _test("zero damping preserves velocity")
    rest, tri, mass, pinned = _make_grid()
    config = _default_config(gravity=0.0, iterations=0, dampingPerSecond=0.0)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    pos = rest.clone()
    vel = torch.full_like(pos, 1.0)
    vel[:, 1] = 2.0
    vel[:, 2] = 3.0
    pins = rest.clone()
    pos, vel = xpbd.step(pos, vel, 1.0 / 60.0, pins)
    ok = True
    for i in range(9):
        ok = _expect_near(vel[i, 0].item(), 1.0, 1e-4, f"preserved vx at {i}") and ok
        ok = _expect_near(vel[i, 1].item(), 2.0, 1e-4, f"preserved vy at {i}") and ok
        ok = _expect_near(vel[i, 2].item(), 3.0, 1e-4, f"preserved vz at {i}") and ok
    if ok:
        _pass()


def test_rest_pose_invariance():
    _test("rest pose invariance without gravity")
    rest, tri, mass, pinned = _make_grid()
    config = _default_config(gravity=0.0, iterations=1, dampingPerSecond=0.0)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    pos = rest.clone()
    vel = torch.zeros_like(pos)
    pins = rest.clone()
    pos, vel = xpbd.step(pos, vel, 1.0 / 60.0, pins)
    ok = True
    for i in range(9):
        for c in range(3):
            ok = _expect_near(pos[i, c].item(), rest[i, c].item(), 1e-5, f"rest at {i},{c}") and ok
    if ok:
        _pass()


def test_stretch_shear_bend_correction():
    _test("stretch/shear/bend constraint correction")
    rest, tri, mass, _ = _make_grid()
    pinned = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=10, dampingPerSecond=0.0)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    deformed = rest.clone()
    deformed[4] = torch.tensor([1.0, 0.5, 0.5])
    pos = deformed.clone()
    vel = torch.zeros_like(pos)
    pins = rest.clone()
    dist_before = float(torch.linalg.norm(deformed[4] - rest[4]))
    pos, vel = xpbd.step(pos, vel, 1.0 / 60.0, pins)
    dist_after = float(torch.linalg.norm(pos[4] - rest[4]))
    ok = _expect(dist_after < dist_before, "constraint should reduce deformation")
    if ok:
        _pass()


# --- Gradient FD checks ---

def test_shear_gradient_fd():
    _test("shear constraint gradient (finite difference)")
    rest = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mass = torch.ones(3)
    pinned = torch.zeros(3, dtype=torch.long)
    config = _default_config()
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    pos = torch.tensor([[0.1, 0.05, 0], [1.1, 0.1, 0.05], [-0.05, 1.05, 0.1]], dtype=torch.float32)
    sc = xpbd.shear_records[0]
    eps = 1e-5

    def compute_c(pa, pb, pc):
        e1 = pb - pa
        e2 = pc - pa
        return torch.dot(e1, e2) - sc.rest_dot

    ok = True
    for vi in range(3):
        idx = [sc.a, sc.b, sc.c][vi]
        for comp in range(3):
            p_plus = pos.clone()
            p_minus = pos.clone()
            p_plus[idx, comp] += eps
            p_minus[idx, comp] -= eps
            c_plus = compute_c(p_plus[sc.a], p_plus[sc.b], p_plus[sc.c])
            c_minus = compute_c(p_minus[sc.a], p_minus[sc.b], p_minus[sc.c])
            fd_grad = (c_plus - c_minus) / (2 * eps)
            e1 = pos[sc.b] - pos[sc.a]
            e2 = pos[sc.c] - pos[sc.a]
            if vi == 0:
                grad = -(e1 + e2)
            elif vi == 1:
                grad = e2
            else:
                grad = e1
            analytic = grad[comp].item()
            ok = _expect_near(fd_grad.item(), analytic, 1e-3, f"shear grad v{vi} c{comp}") and ok
    if ok:
        _pass()


def test_bend_gradient_fd():
    _test("bend constraint gradient (finite difference)")
    rest = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0, 1]
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long)
    mass = torch.ones(4)
    pinned = torch.zeros(4, dtype=torch.long)
    config = _default_config()
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    if not xpbd.bend_records:
        _fail("no bend constraints")
        return
    bc = xpbd.bend_records[0]
    eps = 1e-5
    init_pos = rest.clone()
    init_pos[2, 1] += 0.1
    init_pos[3, 2] += 0.1
    pos = init_pos

    def compute_c(pa, pb, pc, pd):
        e = pb - pa
        e_len = torch.linalg.norm(e)
        if e_len < 1e-12:
            return torch.tensor(0.0)
        n1 = torch.linalg.cross(e, pc - pa)
        n2 = torch.linalg.cross(e, pd - pa)
        n1_len = torch.linalg.norm(n1)
        n2_len = torch.linalg.norm(n2)
        if n1_len < 1e-12 or n2_len < 1e-12:
            return torch.tensor(0.0)
        n1h = n1 / n1_len
        n2h = n2 / n2_len
        eh = e / e_len
        cos_a = torch.clamp(torch.dot(n1h, n2h), -1.0, 1.0)
        sin_a = torch.dot(torch.linalg.cross(n1h, n2h), eh)
        return torch.atan2(sin_a, cos_a) - bc.rest_angle

    ok = True
    for vi in range(4):
        idx = [bc.a, bc.b, bc.c, bc.d][vi]
        for comp in range(3):
            p_plus = pos.clone()
            p_minus = pos.clone()
            p_plus[idx, comp] += eps
            p_minus[idx, comp] -= eps
            c_plus = compute_c(p_plus[bc.a], p_plus[bc.b], p_plus[bc.c], p_plus[bc.d])
            c_minus = compute_c(p_minus[bc.a], p_minus[bc.b], p_minus[bc.c], p_minus[bc.d])
            fd_grad = (c_plus - c_minus) / (2 * eps)
            e = pos[bc.b] - pos[bc.a]
            e_sq = torch.dot(e, e)
            e_len = torch.sqrt(e_sq)
            n1 = torch.linalg.cross(e, pos[bc.c] - pos[bc.a])
            n2 = torch.linalg.cross(e, pos[bc.d] - pos[bc.a])
            n1_len = torch.linalg.norm(n1)
            n2_len = torch.linalg.norm(n2)
            hc = n1_len / e_len
            hd = n2_len / e_len
            n1h = n1 / n1_len
            n2h = n2 / n2_len
            tc = torch.dot(pos[bc.c] - pos[bc.a], e) / e_sq
            td = torch.dot(pos[bc.d] - pos[bc.a], e) / e_sq
            grad_c = -n1h / hc
            grad_d = n2h / hd
            grad_a = grad_c * (-(1.0 - tc)) + grad_d * (-(1.0 - td))
            grad_b = grad_c * (-tc) + grad_d * (-td)
            grads = [grad_a, grad_b, grad_c, grad_d]
            analytic = grads[vi][comp].item()
            ok = _expect_near(fd_grad.item(), analytic, 1e-2, f"bend grad v{vi} c{comp}") and ok
    if ok:
        _pass()


# --- Contact tests ---

def test_self_collision_vf():
    _test("self-collision vertex-face")
    rest = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 0.5]
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mass = torch.ones(4)
    pinned = torch.zeros(4, dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=0, enableSelfCollision=True, thickness=0.05)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    positions = rest.clone()
    positions[3] = torch.tensor([0.5, 0.3, 0.01])
    pos = positions.clone()
    vel = torch.zeros_like(pos)
    pins = rest.clone()
    pos, vel = xpbd.step(pos, vel, 1.0 / 60.0, pins)
    ok = True
    for i in range(4):
        for c in range(3):
            ok = _expect(torch.isfinite(pos[i, c]).item(), f"position at {i},{c} should be finite") and ok
    if ok:
        _pass()


def test_self_collision_ee():
    _test("self-collision edge-edge")
    rest = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0],
        [0, 0, 0.5], [1, 0, 0.5], [0.5, 1, 0.5],
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    mass = torch.ones(6)
    pinned = torch.zeros(6, dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=0, enableSelfCollision=True, thickness=0.3)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    positions = rest.clone()
    positions[0] = torch.tensor([0, 0, 0.2])
    positions[1] = torch.tensor([1, 0, 0.2])
    positions[3] = torch.tensor([0.5, -0.5, 0])
    positions[4] = torch.tensor([0.5, 0.5, 0.4])
    pos = positions.clone()
    vel = torch.zeros_like(pos)
    pins = rest.clone()
    pos, vel = xpbd.step(pos, vel, 1.0 / 60.0, pins)
    ok = True
    for i in range(6):
        for c in range(3):
            ok = _expect(torch.isfinite(pos[i, c]).item(), f"position at {i},{c} should be finite") and ok
    if ok:
        _pass()


def test_capsule_contact():
    _test("capsule contact (crossing)")
    rest = torch.tensor([[0, 1, 0.2], [1, 1, 0], [1, 1, 0.1]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mass = torch.ones(3)
    pinned = torch.tensor([0, 1, 1], dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=0, dampingPerSecond=0.0, enableCollision=True)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    capsule = {
        "previousA": torch.tensor([0, 0.8, 0]),
        "previousB": torch.tensor([0, 1.2, 0]),
        "currentA": torch.tensor([0, 0.8, 0]),
        "currentB": torch.tensor([0, 1.2, 0]),
        "radius": 0.05,
    }
    collider = {
        "previous": rest.clone(),
        "current": rest.clone(),
        "triangles": tri,
        "capsules": [capsule],
    }
    positions = rest.clone()
    vel = torch.zeros_like(rest)
    vel[0] = torch.tensor([0, 0, -96.0])
    pins = rest.clone()
    pos, vel = xpbd.step(positions, vel, 1.0 / 240.0, pins, collider=collider)
    ok = _expect(pos[0, 2].item() >= 0.053 - 1e-4, "fast particle crossed capsule")
    if ok:
        _pass()


def test_capsule_only_collision():
    _test("capsule-only collision (empty triangles)")
    rest = torch.tensor([[0, 1, 0.2], [1, 1, 0], [1, 1, 0.1]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mass = torch.ones(3)
    pinned = torch.tensor([0, 1, 1], dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=0, dampingPerSecond=0.0, enableCollision=True)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    capsule = {
        "previousA": torch.tensor([0, 0.8, 0]),
        "previousB": torch.tensor([0, 1.2, 0]),
        "currentA": torch.tensor([0, 0.8, 0]),
        "currentB": torch.tensor([0, 1.2, 0]),
        "radius": 0.05,
    }
    collider = {
        "previous": rest.clone(),
        "current": rest.clone(),
        "triangles": torch.zeros(0, 3, dtype=torch.long),
        "capsules": [capsule],
    }
    positions = rest.clone()
    vel = torch.zeros_like(rest)
    vel[0] = torch.tensor([0, 0, -96.0])
    pins = rest.clone()
    pos, vel = xpbd.step(positions, vel, 1.0 / 240.0, pins, collider=collider)
    ok = _expect(pos[0, 2].item() >= 0.053 - 1e-4, "capsule-only: fast particle should be stopped")
    if ok:
        _pass()


def test_acceleration_replaces_gravity():
    _test("acceleration replaces gravity")
    rest, tri, mass, pinned = _make_grid()
    config = _default_config(gravity=-9.81, iterations=0, dampingPerSecond=0.0)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    pos = rest.clone()
    vel = torch.zeros_like(pos)
    pins = rest.clone()
    accel = torch.zeros_like(pos)
    accel[:, 1] = 100.0
    pos_g, vel_g = xpbd.step(pos.clone(), vel.clone(), 1.0 / 60.0, pins)
    pos_a, vel_a = xpbd.step(pos.clone(), vel.clone(), 1.0 / 60.0, pins, acceleration=accel)
    ok = True
    ok = _expect(pos_a[0, 1].item() > pos_g[0, 1].item() + 0.001, "acceleration should override gravity") and ok
    if ok:
        _pass()


# --- Gradient flow tests ---

def test_gradient_flow_stretch():
    _test("gradient flow through stretch constraint")
    rest = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 0]], dtype=torch.long)
    mass = torch.ones(2)
    pinned = torch.zeros(2, dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=1, dampingPerSecond=0.0, stretchCompliance=0.0)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    pos = torch.tensor([[0, 0, 0], [1.5, 0, 0]], dtype=torch.float32, requires_grad=True)
    vel = torch.zeros_like(pos, requires_grad=True)
    pins = rest.clone()
    out_pos, out_vel = xpbd.step(pos, vel, 1.0 / 60.0, pins)
    loss = out_pos.sum()
    loss.backward()
    ok = True
    ok = _expect(pos.grad is not None, "gradient should flow through stretch") and ok
    if ok:
        ok = _expect(pos.grad.abs().sum().item() > 1e-6, "gradient should be non-trivial") and ok
    if ok:
        _pass()


def test_gradient_flow_bend():
    _test("gradient flow through bend constraint")
    rest = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0, 1]
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long)
    mass = torch.ones(4)
    pinned = torch.zeros(4, dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=1, dampingPerSecond=0.0, bendCompliance=0.0)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    init_pos = rest.clone()
    init_pos[2, 1] = 0.8
    pos = init_pos.requires_grad_(True)
    vel = torch.zeros_like(pos, requires_grad=True)
    pins = rest.clone()
    out_pos, out_vel = xpbd.step(pos, vel, 1.0 / 60.0, pins)
    loss = out_pos.sum()
    loss.backward()
    ok = True
    ok = _expect(pos.grad is not None, "gradient should flow through bend") and ok
    if ok:
        ok = _expect(pos.grad.abs().sum().item() > 1e-6, "bend gradient should be non-trivial") and ok
    if ok:
        _pass()


def test_gradient_fd_step():
    _test("finite difference gradient of step output")
    rest = torch.tensor([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mass = torch.ones(3)
    pinned = torch.zeros(3, dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=1, dampingPerSecond=0.0, stretchCompliance=0.0)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    eps = 1e-4
    pos = rest.clone().requires_grad_(True)
    vel = torch.zeros_like(pos)
    pins = rest.clone()
    out_pos, _ = xpbd.step(pos, vel, 1.0 / 60.0, pins)
    target_component = out_pos[1, 0]
    target_component.backward()
    analytic_grad = pos.grad.clone()

    fd_grad = torch.zeros_like(pos)
    for i in range(3):
        for c in range(3):
            p_plus = rest.clone()
            p_minus = rest.clone()
            p_plus[i, c] += eps
            p_minus[i, c] -= eps
            out_plus, _ = xpbd.step(p_plus, vel.clone(), 1.0 / 60.0, pins)
            out_minus, _ = xpbd.step(p_minus, vel.clone(), 1.0 / 60.0, pins)
            fd_grad[i, c] = (out_plus[1, 0] - out_minus[1, 0]) / (2 * eps)

    ok = True
    for i in range(3):
        for c in range(3):
            ok = _expect_near(analytic_grad[i, c].item(), fd_grad[i, c].item(), 1e-3, f"FD grad at {i},{c}") and ok
    if ok:
        _pass()


def test_gradient_fd_vf_contact():
    _test("FD gradient with active VF self-collision contact")
    rest = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 0.5]
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mass = torch.ones(4)
    pinned = torch.zeros(4, dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=0, dampingPerSecond=0.0, enableSelfCollision=True, thickness=0.1)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    eps = 1e-4

    def run_step(pos_vals):
        p = pos_vals.clone().detach()
        v = torch.zeros_like(p)
        pins = rest.clone()
        out, _ = xpbd.step(p, v, 1.0 / 60.0, pins)
        return out[3, 1]

    base_pos = rest.clone()
    base_pos[3] = torch.tensor([0.5, 0.15, 0.01])
    base_val = run_step(base_pos)

    ok = True
    for i in range(4):
        for c in range(3):
            p_plus = base_pos.clone()
            p_minus = base_pos.clone()
            p_plus[i, c] += eps
            p_minus[i, c] -= eps
            fd = (run_step(p_plus) - run_step(p_minus)) / (2 * eps)
            ok = _expect(
                torch.isfinite(fd).item(),
                f"VF contact FD grad at {i},{c} should be finite",
            ) and ok
    if ok:
        _pass()


def test_gradient_fd_ee_contact():
    _test("FD gradient with active EE self-collision contact")
    rest = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0],
        [0, 0, 0.5], [1, 0, 0.5], [0.5, 1, 0.5],
    ], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    mass = torch.ones(6)
    pinned = torch.zeros(6, dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=0, dampingPerSecond=0.0, enableSelfCollision=True, thickness=0.3)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    eps = 1e-4

    def run_step(pos_vals):
        p = pos_vals.clone().detach()
        v = torch.zeros_like(p)
        pins = rest.clone()
        out, _ = xpbd.step(p, v, 1.0 / 60.0, pins)
        return out[0, 2]

    base_pos = rest.clone()
    base_pos[0] = torch.tensor([0, 0, 0.2])
    base_pos[1] = torch.tensor([1, 0, 0.2])
    base_pos[3] = torch.tensor([0.5, -0.5, 0])
    base_pos[4] = torch.tensor([0.5, 0.5, 0.4])
    base_val = run_step(base_pos)

    ok = True
    for i in range(6):
        for c in range(3):
            p_plus = base_pos.clone()
            p_minus = base_pos.clone()
            p_plus[i, c] += eps
            p_minus[i, c] -= eps
            fd = (run_step(p_plus) - run_step(p_minus)) / (2 * eps)
            ok = _expect(
                torch.isfinite(fd).item(),
                f"EE contact FD grad at {i},{c} should be finite",
            ) and ok
    if ok:
        _pass()


def test_gradient_fd_body_contact():
    _test("FD gradient with active body (STM) contact")
    rest = torch.tensor([[0, 1, 0.2], [1, 1, 0], [1, 1, 0.1]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mass = torch.ones(3)
    pinned = torch.tensor([0, 1, 1], dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=0, dampingPerSecond=0.0, enableCollision=True, thickness=0.05)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    collider = {
        "previous": rest.clone(),
        "current": rest.clone(),
        "triangles": tri,
        "capsules": [],
    }
    eps = 1e-4

    def run_step(pos_vals):
        p = pos_vals.clone().detach()
        v = torch.zeros_like(p)
        pins = rest.clone()
        out, _ = xpbd.step(p, v, 1.0 / 240.0, pins, collider=collider)
        return out[0, 2]

    base_pos = rest.clone()
    base_pos[0] = torch.tensor([0, 1, 0.01])
    base_val = run_step(base_pos)

    ok = True
    for i in range(3):
        for c in range(3):
            p_plus = base_pos.clone()
            p_minus = base_pos.clone()
            p_plus[i, c] += eps
            p_minus[i, c] -= eps
            fd = (run_step(p_plus) - run_step(p_minus)) / (2 * eps)
            ok = _expect(
                torch.isfinite(fd).item(),
                f"body contact FD grad at {i},{c} should be finite",
            ) and ok
    if ok:
        _pass()


def test_gradient_fd_capsule_contact():
    _test("FD gradient with active capsule contact")
    rest = torch.tensor([[0, 1, 0.2], [1, 1, 0], [1, 1, 0.1]], dtype=torch.float32)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mass = torch.ones(3)
    pinned = torch.tensor([0, 1, 1], dtype=torch.long)
    config = _default_config(gravity=0.0, iterations=0, dampingPerSecond=0.0, enableCollision=True, thickness=0.003)
    xpbd = TorchXPBD(rest, tri, mass, pinned, config)
    capsule = {
        "previousA": torch.tensor([0, 0.8, 0]),
        "previousB": torch.tensor([0, 1.2, 0]),
        "currentA": torch.tensor([0, 0.8, 0]),
        "currentB": torch.tensor([0, 1.2, 0]),
        "radius": 0.05,
    }
    collider = {
        "previous": rest.clone(),
        "current": rest.clone(),
        "triangles": tri,
        "capsules": [capsule],
    }
    eps = 1e-4

    def run_step(pos_vals):
        p = pos_vals.clone().detach()
        v = torch.zeros_like(p)
        pins = rest.clone()
        out, _ = xpbd.step(p, v, 1.0 / 240.0, pins, collider=collider)
        return out[0, 2]

    base_pos = rest.clone()
    base_pos[0] = torch.tensor([0, 1, 0.01])
    base_val = run_step(base_pos)

    ok = True
    for i in range(3):
        for c in range(3):
            p_plus = base_pos.clone()
            p_minus = base_pos.clone()
            p_plus[i, c] += eps
            p_minus[i, c] -= eps
            fd = (run_step(p_plus) - run_step(p_minus)) / (2 * eps)
            ok = _expect(
                torch.isfinite(fd).item(),
                f"capsule contact FD grad at {i},{c} should be finite",
            ) and ok
    if ok:
        _pass()


# --- Probe comparison ---

def test_probe_comparison(probe_path: str):
    _test(f"CPU probe comparison ({Path(probe_path).name})")
    with open(probe_path) as f:
        data = json.load(f)

    rest = torch.tensor(data["rest"], dtype=torch.float32)
    triangles = torch.tensor(data["triangles"], dtype=torch.long).reshape(-1, 3)
    mass = torch.tensor(data["mass"], dtype=torch.float32)
    pinned = torch.tensor(data["pinned"], dtype=torch.long)
    config = data["config"]

    xpbd = TorchXPBD(rest, triangles, mass, pinned, config)

    positions = torch.tensor(data["positions"], dtype=torch.float32)
    velocities = torch.tensor(data["velocities"], dtype=torch.float32)
    pin_targets = torch.tensor(data["pin_targets"], dtype=torch.float32)
    dt = data["dt"]

    collider_data = data.get("collider")
    collider = None
    if collider_data is not None:
        collider = {
            "previous": torch.tensor(collider_data["previous"], dtype=torch.float32),
            "current": torch.tensor(collider_data["current"], dtype=torch.float32),
            "triangles": torch.tensor(collider_data["triangles"], dtype=torch.long).reshape(-1, 3),
            "capsules": [],
        }
        for cap in collider_data.get("capsules", []):
            collider["capsules"].append({
                "previousA": torch.tensor(cap["previousA"], dtype=torch.float32),
                "previousB": torch.tensor(cap["previousB"], dtype=torch.float32),
                "currentA": torch.tensor(cap["currentA"], dtype=torch.float32),
                "currentB": torch.tensor(cap["currentB"], dtype=torch.float32),
                "radius": cap["radius"],
            })

    out_pos, out_vel = xpbd.step(positions, velocities, dt, pin_targets, collider=collider)

    expected_pos = torch.tensor(data["expected_positions"], dtype=torch.float32)
    expected_vel = torch.tensor(data["expected_velocities"], dtype=torch.float32)

    pos_err = float(torch.linalg.norm(out_pos - expected_pos).max())
    vel_err = float(torch.linalg.norm(out_vel - expected_vel).max())

    tol = data.get("tolerance", 1e-4)
    ok = True
    ok = _expect(pos_err < tol, f"position error {pos_err} > {tol}") and ok
    ok = _expect(vel_err < tol, f"velocity error {vel_err} > {tol}") and ok
    if ok:
        _pass()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=str, default=None, help="Path to probe fixture JSON")
    args = parser.parse_args()

    print("Temporal XPBD reference tests:")

    # Basic physics
    test_gravity_free_fall()
    test_pinned_invariance()
    test_damping()
    test_zero_damping_preserves_velocity()
    test_rest_pose_invariance()
    test_stretch_shear_bend_correction()

    # Gradient FD checks (structural)
    test_shear_gradient_fd()
    test_bend_gradient_fd()

    # Contact tests
    test_self_collision_vf()
    test_self_collision_ee()
    test_capsule_contact()
    test_capsule_only_collision()
    test_acceleration_replaces_gravity()

    # Gradient flow
    test_gradient_flow_stretch()
    test_gradient_flow_bend()
    test_gradient_fd_step()

    # Active-contact FD gradient checks
    test_gradient_fd_vf_contact()
    test_gradient_fd_ee_contact()
    test_gradient_fd_body_contact()
    test_gradient_fd_capsule_contact()

    if args.probe:
        test_probe_comparison(args.probe)

    print(f"\n{_tests} tests, {len(_failures)} failures")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
