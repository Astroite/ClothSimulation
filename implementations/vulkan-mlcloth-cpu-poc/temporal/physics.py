"""PyTorch XPBD reference matching the Demo CPU/GPU solver.

Public interface:
    TorchXPBD(rest, triangles, mass, pinned, config)
    .step(position, velocity, dt, pin_targets, collider=None, acceleration=None)
        -> (new_position, new_velocity)

Constraint tables and self-collision records are exposed for debugging.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from .structure import build_structure,project
from .capsule import project_capsules

EPS = 1e-12
DENOM_FLOOR = 1e-20


def _vec3_len(v: Tensor) -> Tensor:
    return torch.linalg.vector_norm(v)


def _normalize(v: Tensor) -> Tensor:
    n = _vec3_len(v)
    return v / n if n > 1e-12 else v.new_zeros(3)


def _point_triangle_bary(p: Tensor, a: Tensor, b: Tensor, c: Tensor):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = torch.dot(ab, ap)
    d2 = torch.dot(ac, ap)

    if d1 <= 0 and d2 <= 0:
        return a.clone(), p.new_tensor(0.0), p.new_tensor(0.0), p.new_tensor(1.0)

    bp = p - b
    d3 = torch.dot(ab, bp)
    d4 = torch.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b.clone(), p.new_tensor(1.0), p.new_tensor(0.0), p.new_tensor(0.0)

    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3 + 1e-30)
        closest = a + ab * v
        return closest, v, p.new_tensor(0.0), 1.0 - v

    cp = p - c
    d5 = torch.dot(ab, cp)
    d6 = torch.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c.clone(), p.new_tensor(0.0), p.new_tensor(1.0), p.new_tensor(0.0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6 + 1e-30)
        closest = a + ac * w
        return closest, p.new_tensor(0.0), w, 1.0 - w

    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4-d3)>=0 and (d5-d6)>=0:
        w=(d4-d3)/(d4-d3+d5-d6)
        return b+(c-b)*w,1-w,w,p.new_zeros(())
    denom = va + vb + vc
    if abs(float(denom)) < 1e-20:
        return a.clone(), p.new_tensor(0.0), p.new_tensor(0.0), p.new_tensor(1.0)

    inv = 1.0 / denom
    u = torch.clamp(vb * inv, 0.0, 1.0)
    vv = torch.clamp(vc * inv, 0.0, 1.0 - u)
    closest = a + ab * u + ac * vv
    return closest, u, vv, torch.clamp(1.0 - u - vv, 0.0)


def _edge_edge_sq(a0: Tensor, a1: Tensor, b0: Tensor, b1: Tensor):
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    aa = torch.dot(d1, d1)
    ee = torch.dot(d2, d2)
    ff = torch.dot(d2, r)
    s = a0.new_tensor(0.0)
    t = a0.new_tensor(0.0)

    if aa <= 1e-16 and ee <= 1e-16:
        d = a0 - b0
        return torch.dot(d, d), s, t
    if aa <= 1e-16:
        t = torch.clamp(ff / ee, 0.0, 1.0)
    else:
        cc = torch.dot(d1, r)
        if ee <= 1e-16:
            s = torch.clamp(-cc / aa, 0.0, 1.0)
        else:
            bb_val = torch.dot(d1, d2)
            denom = aa * ee - bb_val * bb_val
            if abs(float(denom)) > 1e-16:
                s = torch.clamp((bb_val * ff - cc * ee) / denom, 0.0, 1.0)
            else:
                s = a0.new_tensor(0.0)
            t = (bb_val * s + ff) / ee
            if t < 0:
                t = a0.new_tensor(0.0)
                s = torch.clamp(-cc / aa, 0.0, 1.0)
            elif t > 1:
                t = a0.new_tensor(1.0)
                s = torch.clamp((bb_val - cc) / aa, 0.0, 1.0)
    pa = a0 + d1 * s
    pb = b0 + d2 * t
    diff = pa - pb
    return torch.dot(diff, diff), s, t


def _signed_dihedral_angle(pos: Tensor, a: int, b: int, c: int, d: int) -> Tensor:
    e = pos[b] - pos[a]
    n1 = torch.linalg.cross(e, pos[c] - pos[a])
    n2 = torch.linalg.cross(e, pos[d] - pos[a])
    n1_norm = torch.linalg.norm(n1)
    n2_norm = torch.linalg.norm(n2)
    e_norm = torch.linalg.norm(e)
    if n1_norm < EPS or n2_norm < EPS or e_norm < EPS:
        return pos.new_tensor(0.0)
    n1h = n1 / n1_norm
    n2h = n2 / n2_norm
    eh = e / e_norm
    cos_a = torch.clamp(torch.dot(n1h, n2h), -1.0, 1.0)
    sin_a = torch.dot(torch.linalg.cross(n1h, n2h), eh)
    return torch.atan2(sin_a, cos_a)


@dataclass
class ShearRecord:
    a: int
    b: int
    c: int
    rest_dot: float


@dataclass
class BendRecord:
    a: int
    b: int
    c: int
    d: int
    rest_angle: float


@dataclass
class CollisionRecord:
    vertex: int
    ids: Tuple[int, ...]
    weights: Tuple[float, ...]
    delta: Tuple[float, float, float]
    distance: float


class TorchXPBD:
    def __init__(
        self,
        rest: Tensor,
        triangles: Tensor,
        mass: Tensor,
        pinned: Tensor,
        config: Dict[str, Any],
    ):
        device = rest.device
        dtype = rest.dtype
        self.device = device
        self.dtype = dtype
        self.config = dict(config)

        self.N = rest.shape[0]
        self.rest = rest.clone()
        self.triangles = triangles.clone()
        self.mass = mass.clone()
        self.pinned = pinned.clone()
        self.inv_mass = torch.where(
            pinned > 0, torch.zeros_like(mass), 1.0 / mass
        )

        self.stretch_pairs: List[Tuple[int, int]] = []
        self.stretch_rest: List[float] = []
        self.shear_records: List[ShearRecord] = []
        self.bend_records: List[BendRecord] = []

        self.collision_records: List[CollisionRecord] = []

        self._build_constraints(rest, triangles)
        build_structure(self)

    def _build_constraints(self, rest: Tensor, triangles: Tensor) -> None:
        tri_np = triangles.cpu().numpy()
        rest_np = rest.cpu().numpy()

        edge_set = set()
        for t in range(tri_np.shape[0]):
            for k in range(3):
                a = int(tri_np[t, k])
                b = int(tri_np[t, (k + 1) % 3])
                edge_set.add((min(a, b), max(a, b)))

        for a, b in sorted(edge_set):
            self.stretch_pairs.append((a, b))
            d = rest_np[a] - rest_np[b]
            self.stretch_rest.append(float(math.sqrt(float((d * d).sum()))))

        for t in range(tri_np.shape[0]):
            i0, i1, i2 = int(tri_np[t, 0]), int(tri_np[t, 1]), int(tri_np[t, 2])
            e1 = rest_np[i1] - rest_np[i0]
            e2 = rest_np[i2] - rest_np[i0]
            self.shear_records.append(ShearRecord(i0, i1, i2, float((e1 * e2).sum())))

        edge_tris: Dict[Tuple[int, int], List[int]] = {}
        for t in range(tri_np.shape[0]):
            for k in range(3):
                a = int(tri_np[t, k])
                b = int(tri_np[t, (k + 1) % 3])
                key = (min(a, b), max(a, b))
                edge_tris.setdefault(key, []).append(t)

        seen_bend = set()
        for (ea, eb), tris in sorted(edge_tris.items()):
            if len(tris) != 2:
                continue
            opposites = []
            for side in range(2):
                for k in range(3):
                    corner = int(tri_np[tris[side], k])
                    if corner != ea and corner != eb:
                        opposites.append(corner)
                        break
            if len(opposites) < 2 or opposites[0] == opposites[1]:
                continue
            a, b, c, d = ea, eb, opposites[0], opposites[1]
            pair = (min(c, d), max(c, d))
            seen_bend.add(pair)
            rest_angle = _signed_dihedral_angle(rest, a, b, c, d)
            self.bend_records.append(BendRecord(a, b, c, d, float(rest_angle.item())))

    def _alpha_tilde(self, compliance: float, dt: float) -> float:
        return compliance / max(dt * dt, EPS)

    def step(
        self,
        position: Tensor,
        velocity: Tensor,
        dt: float,
        pin_targets: Tensor,
        collider: Optional[Dict[str, Any]] = None,
        acceleration: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        cfg = self.config
        iterations = cfg.get("iterations", 2)
        stretch_comp = cfg.get("stretchCompliance", 0.0)
        shear_comp = cfg.get("shearCompliance", 0.0)
        bend_comp = cfg.get("bendCompliance", 0.0)
        damping = cfg.get("dampingPerSecond", 0.99)
        gravity_val = cfg.get("gravity", -9.81)
        enable_collision = cfg.get("enableCollision", False)
        enable_self_collision = cfg.get("enableSelfCollision", False)
        thickness = cfg.get("thickness", 0.003)
        friction = cfg.get("friction", 0.1)

        pos = position.clone()
        vel = velocity.clone()
        pin_mask = self.pinned > 0

        prev_pos = pos.clone()

        # 1. Pin assignment before structural passes
        pos = torch.where(pin_mask.unsqueeze(1), pin_targets, pos)
        vel = torch.where(pin_mask.unsqueeze(1), pos.new_zeros(vel.shape), vel)

        # 2. Damping: exp(-dampingPerSecond * dt)
        damp_factor = math.exp(-damping * dt)
        vel = vel * damp_factor

        # 3. Gravity / acceleration (acceleration REPLACES gravity)
        if acceleration is not None:
            vel = vel + acceleration * dt
        else:
            gravity_vec = pos.new_tensor([0.0, gravity_val, 0.0])
            vel = vel + gravity_vec * dt

        # 4. Semi-implicit Euler
        pos = pos + vel * dt
        pos = torch.where(pin_mask.unsqueeze(1), pin_targets, pos)

        # 5. XPBD iterations
        stretch_lambdas = pos.new_zeros(len(self.stretch_pairs))
        shear_lambdas = pos.new_zeros(len(self.shear_records))
        bend_lambdas = pos.new_zeros(len(self.bend_records))

        for _ in range(iterations):
            pos,stretch_lambdas=project(self,pos,0,stretch_comp,dt,stretch_lambdas)
            pos,shear_lambdas=project(self,pos,1,shear_comp,dt,shear_lambdas)
            pos,bend_lambdas=project(self,pos,2,bend_comp,dt,bend_lambdas)

        # 6. Self-collision
        if enable_self_collision:
            pos = self._apply_self_collision(pos, prev_pos, thickness, pin_mask)

        # 7-9. Body collision pipeline (exact C++ order)
        body_contacts: List[Dict[str, Any]] = []
        ground_impulse = pos.new_zeros(self.N)
        if collider is not None and enable_collision:
            # 7a. Triangle body collision
            if collider.get("triangles") is not None and collider["triangles"].shape[0] > 0:
                pos, body_contacts = self._apply_body_collision(
                    pos, prev_pos, collider, thickness, dt, pin_mask
                )
            # 7b. Capsule collision (works with or without triangles)
            capsules = collider.get("capsules", [])
            if capsules:
                pos,body_contacts=project_capsules(self,pos,prev_pos,capsules,thickness,dt,pin_mask,body_contacts)
            # 7c. Recover body contacts (re-check embedded vertices)
            if collider.get("triangles") is not None and collider["triangles"].shape[0] > 0 and body_contacts:
                pos, body_contacts = self._recover_body_contacts(
                    pos, prev_pos, collider, thickness, dt, pin_mask, body_contacts
                )

        # Ground also applies without a body collider. Keep normal impulse live
        # for the subsequent differentiable Coulomb friction calculation.
        if enable_collision:
            ground=(~pin_mask)&(pos[:,1]<thickness)
            ground_impulse=torch.where(ground,(thickness-pos[:,1])/dt,torch.zeros_like(pos[:,1]))
            pos=torch.stack((pos[:,0],torch.where(ground,pos.new_full((self.N,),thickness),pos[:,1]),pos[:,2]),-1)

        # 9. Velocity reconstruction from position change
        vel = torch.where(
            pin_mask.unsqueeze(1),
            pos.new_zeros(vel.shape),
            (pos - prev_pos) / dt,
        )

        # 10. Body-relative friction AFTER velocity reconstruction
        if collider is not None and enable_collision and body_contacts:
            for ct in body_contacts:
                v = ct["vertex"]
                if pin_mask[v] or float(ground_impulse[v]) > 0:
                    continue
                body_vel = torch.as_tensor(ct["body_vel"],device=pos.device,dtype=pos.dtype)
                contact_normal = torch.as_tensor(ct["normal"],device=pos.device,dtype=pos.dtype)
                rel_vel = vel[v] - body_vel
                vn = torch.dot(rel_vel, contact_normal)
                tangential = rel_vel - contact_normal * vn
                tang_len = torch.linalg.norm(tangential)
                if float(tang_len) > EPS:
                    max_friction = ct["impulse"] * friction
                    reduction = torch.minimum(tang_len,torch.as_tensor(max_friction,device=pos.device,dtype=pos.dtype))
                    vel[v] = vel[v] - _normalize(tangential) * reduction
                if float(vn) < 0 or ct.get("recovery", False):
                    vel[v] = vel[v] - contact_normal * vn

        # 11. Ground friction
        tangent=vel*vel.new_tensor([1,0,1]);speed=torch.linalg.vector_norm(tangent,dim=-1)
        factor=torch.where((ground_impulse>0)&(speed>1e-9),(friction*ground_impulse/speed.clamp_min(1e-30)).clamp_max(1),torch.zeros_like(speed))
        vel=vel-tangent*factor[:,None]
        vel=torch.stack((vel[:,0],torch.where(ground_impulse>0,vel[:,1].clamp_min(0),vel[:,1]),vel[:,2]),-1)

        # 12. Hard pin overwrite
        pos = torch.where(pin_mask.unsqueeze(1), pin_targets, pos)
        vel = torch.where(pin_mask.unsqueeze(1), pos.new_zeros(vel.shape), vel)

        return pos, vel

    def _apply_stretch(
        self, pos: Tensor, compliance: float, dt: float, lambdas: Tensor, pin_mask: Tensor
    ) -> Tensor:
        if not self.stretch_pairs:
            return pos
        alpha = self._alpha_tilde(compliance, dt)
        N = self.N
        acc = pos.new_zeros(N, 3)
        weight_acc = pos.new_zeros(N)

        for ci, (a, b) in enumerate(self.stretch_pairs):
            delta = pos[a] - pos[b]
            dist = _vec3_len(delta)
            if dist < EPS:
                continue
            residual = dist - self.stretch_rest[ci]
            w_sum = self.inv_mass[a] + self.inv_mass[b]
            denom = max(w_sum + alpha, DENOM_FLOOR)
            step = (-residual - alpha * lambdas[ci]) / denom
            lambdas[ci] = lambdas[ci] + step
            grad = delta / dist
            acc[a] = acc[a] + grad * step
            acc[b] = acc[b] - grad * step
            weight_acc[a] = weight_acc[a] + self.inv_mass[a]
            weight_acc[b] = weight_acc[b] + self.inv_mass[b]

        free = (~pin_mask).float()
        scale = torch.where(weight_acc > 0, self.inv_mass / weight_acc, pos.new_zeros(self.N))
        return pos + acc * scale.unsqueeze(1) * free.unsqueeze(1)

    def _apply_shear(
        self, pos: Tensor, compliance: float, dt: float, lambdas: Tensor, pin_mask: Tensor
    ) -> Tensor:
        if not self.shear_records:
            return pos
        alpha = self._alpha_tilde(compliance, dt)

        for ci, sc in enumerate(self.shear_records):
            e1 = pos[sc.b] - pos[sc.a]
            e2 = pos[sc.c] - pos[sc.a]
            ga = -e1 - e2
            gb = e2
            gc = e1
            w_a = self.inv_mass[sc.a]
            w_b = self.inv_mass[sc.b]
            w_c = self.inv_mass[sc.c]
            denom = w_a * torch.dot(ga, ga) + w_b * torch.dot(gb, gb) + w_c * torch.dot(gc, gc) + alpha
            if denom < 1e-12:
                continue
            current_dot = torch.dot(e1, e2)
            residual = -(current_dot - sc.rest_dot)
            dl = (residual - alpha * lambdas[ci]) / denom
            lambdas[ci] = lambdas[ci] + dl

            if not pin_mask[sc.a]:
                pos[sc.a] = pos[sc.a] + ga * (w_a * dl)
            if not pin_mask[sc.b]:
                pos[sc.b] = pos[sc.b] + gb * (w_b * dl)
            if not pin_mask[sc.c]:
                pos[sc.c] = pos[sc.c] + gc * (w_c * dl)

        return pos

    def _apply_bend(
        self, pos: Tensor, compliance: float, dt: float, lambdas: Tensor, pin_mask: Tensor
    ) -> Tensor:
        if not self.bend_records:
            return pos
        alpha = self._alpha_tilde(compliance, dt)

        for ci, bc in enumerate(self.bend_records):
            a, b, c, d = bc.a, bc.b, bc.c, bc.d
            e = pos[b] - pos[a]
            e_sq = torch.dot(e, e)
            if e_sq < EPS * EPS:
                continue
            n1 = torch.linalg.cross(e, pos[c] - pos[a])
            n2 = torch.linalg.cross(e, pos[d] - pos[a])
            n1_len = torch.linalg.norm(n1)
            n2_len = torch.linalg.norm(n2)
            if n1_len < EPS or n2_len < EPS:
                continue
            e_len = torch.sqrt(e_sq)
            n1h = n1 / n1_len
            n2h = n2 / n2_len
            eh = e / e_len
            cos_a = torch.clamp(torch.dot(n1h, n2h), -1.0, 1.0)
            sin_a = torch.dot(torch.linalg.cross(n1h, n2h), eh)
            angle = torch.atan2(sin_a, cos_a)
            C = torch.atan2(torch.sin(angle - bc.rest_angle), torch.cos(angle - bc.rest_angle))
            hc = n1_len / e_len
            hd = n2_len / e_len
            tc = torch.dot(pos[c] - pos[a], e) / e_sq
            td = torch.dot(pos[d] - pos[a], e) / e_sq
            grad_c = -n1h / hc
            grad_d = n2h / hd
            grad_a = grad_c * (-(1.0 - tc)) + grad_d * (-(1.0 - td))
            grad_b = grad_c * (-tc) + grad_d * (-td)
            w_sum = (
                self.inv_mass[a] * torch.dot(grad_a, grad_a)
                + self.inv_mass[b] * torch.dot(grad_b, grad_b)
                + self.inv_mass[c] * torch.dot(grad_c, grad_c)
                + self.inv_mass[d] * torch.dot(grad_d, grad_d)
            )
            denom = max(w_sum + alpha, DENOM_FLOOR)
            d_lambda = (-C - alpha * lambdas[ci]) / denom
            lambdas[ci] = lambdas[ci] + d_lambda

            if not pin_mask[a]:
                pos[a] = pos[a] + grad_a * (self.inv_mass[a] * d_lambda)
            if not pin_mask[b]:
                pos[b] = pos[b] + grad_b * (self.inv_mass[b] * d_lambda)
            if not pin_mask[c]:
                pos[c] = pos[c] + grad_c * (self.inv_mass[c] * d_lambda)
            if not pin_mask[d]:
                pos[d] = pos[d] + grad_d * (self.inv_mass[d] * d_lambda)

        return pos

    def _apply_self_collision(
        self, pos: Tensor, prev_pos: Tensor, thickness: float, pin_mask: Tensor
    ) -> Tensor:
        N = self.N
        if N == 0:
            return pos

        tri = self.triangles
        inv_mass = self.inv_mass

        rest_np = self.rest.detach().cpu().numpy()
        adj_offsets = [0]
        adj_verts: List[int] = []
        adj_dists: List[float] = []
        for v in range(N):
            neighbors: Dict[int, float] = {}
            for a, b in self.stretch_pairs:
                other = -1
                if a == v:
                    other = b
                elif b == v:
                    other = a
                if other >= 0:
                    d_val = float(math.sqrt(float(((rest_np[v] - rest_np[other]) ** 2).sum())))
                    if other not in neighbors or d_val < neighbors[other]:
                        neighbors[other] = d_val
            for nb, d_val in sorted(neighbors.items()):
                adj_verts.append(nb)
                adj_dists.append(d_val)
            adj_offsets.append(len(adj_verts))

        def is_neighbor(va: int, vb: int) -> bool:
            first = adj_offsets[va]
            last = adj_offsets[va + 1]
            for idx in range(first, last):
                if adj_verts[idx] == vb and adj_dists[idx] <= 2 * thickness:
                    return True
            return False

        self.collision_records = []

        # Phase 1: DETACHED candidate selection using snapshot positions
        with torch.no_grad():
            snapshot = pos.detach().clone()
            prev_snapshot = prev_pos.detach().clone()

            # Store selected contact info for phase 2
            selected_contacts: List[Dict[str, Any]] = []

            for kind in range(2):
                prim_count = tri.shape[0] if kind == 0 else len(self.stretch_pairs)
                if prim_count == 0:
                    continue
                query_count = N if kind == 0 else prim_count

                for qid in range(query_count):
                    if kind == 0:
                        v_id = qid
                        va, vb = v_id, v_id
                    else:
                        va = self.stretch_pairs[qid][0]
                        vb = self.stretch_pairs[qid][1]

                    aa = snapshot[va]
                    bb = snapshot[vb]
                    oa = prev_snapshot[va]
                    ob = prev_snapshot[vb]

                    best_ids = None
                    best_weights = None
                    best_normal_snap = None
                    best_s = None
                    best_t = None
                    best_key = (float("inf"), float("inf"))

                    for pi in range(prim_count):
                        if kind == 0:
                            ia = int(tri[pi, 0])
                            ib = int(tri[pi, 1])
                            ic = int(tri[pi, 2])
                            if is_neighbor(v_id, ia) or is_neighbor(v_id, ib) or is_neighbor(v_id, ic):
                                continue
                            pa, pb, pc = snapshot[ia], snapshot[ib], snapshot[ic]
                            opa, opb, opc = prev_snapshot[ia], prev_snapshot[ib], prev_snapshot[ic]
                            closest_q, u_f, v_f, w0_f = _point_triangle_bary(aa, pa, pb, pc)
                            distance = float(_vec3_len(aa - closest_q))
                            normal = torch.linalg.cross(pb - pa, pc - pa)
                            nl = float(torch.linalg.norm(normal))
                            if nl < 1e-12:
                                continue
                            normal = normal / nl
                            old_normal = torch.linalg.cross(opb - opa, opc - opa)
                            if float(torch.dot(oa - opa, old_normal)) < 0:
                                normal = -normal

                            contact = distance < thickness
                            if not contact:
                                motion = aa - oa
                                bound = max(
                                    float(torch.linalg.norm(pa - opa - motion)),
                                    float(torch.linalg.norm(pb - opb - motion)),
                                    float(torch.linalg.norm(pc - opc - motion)),
                                )
                                if distance > bound + thickness:
                                    continue
                                time = 0.0
                                for _ in range(24):
                                    if bound < 1e-9:
                                        break
                                    at = opa + (pa - opa) * time
                                    bt = opb + (pb - opb) * time
                                    ct = opc + (pc - opc) * time
                                    vt = oa + (aa - oa) * time
                                    gap_closest, bu, bv, bw0 = _point_triangle_bary(vt, at, bt, ct)
                                    gap = float(_vec3_len(vt - gap_closest))
                                    if gap < thickness + 1e-5:
                                        u_f, v_f, w0_f = bu, bv, bw0
                                        contact = True
                                        break
                                    time += (gap - thickness) / bound
                                    if time > 1:
                                        break
                            if not contact:
                                continue

                            residual_snap = float(torch.dot(aa - closest_q, normal)) - thickness
                            if residual_snap >= 0:
                                continue
                            w0, u_val, vv_val = float(w0_f), float(u_f), float(v_f)
                            denom_val = (
                                float(inv_mass[v_id])
                                + float(inv_mass[ia]) * w0 * w0
                                + float(inv_mass[ib]) * u_val * u_val
                                + float(inv_mass[ic]) * vv_val * vv_val
                            )
                            key = (distance, float(pi))
                            if denom_val < 1e-12 or key >= best_key:
                                continue
                            mobile = max(
                                float(inv_mass[v_id]),
                                float(inv_mass[ia]) * w0,
                                float(inv_mass[ib]) * u_val,
                                float(inv_mass[ic]) * vv_val,
                            )
                            best_key = key
                            best_ids = (v_id, ia, ib, ic)
                            best_weights = (1.0, -w0, -u_val, -vv_val)
                            best_normal_snap = normal.clone()
                            best_s = None
                            best_t = None

                        else:
                            if pi <= qid:
                                continue
                            c_idx = self.stretch_pairs[pi][0]
                            d_idx = self.stretch_pairs[pi][1]
                            if is_neighbor(va, c_idx) or is_neighbor(va, d_idx) or is_neighbor(vb, c_idx) or is_neighbor(vb, d_idx):
                                continue
                            cc, dd = snapshot[c_idx], snapshot[d_idx]
                            oc, od = prev_snapshot[c_idx], prev_snapshot[d_idx]
                            dist_sq, s_f, t_f = _edge_edge_sq(aa, bb, cc, dd)
                            distance = math.sqrt(max(0, float(dist_sq)))
                            contact = distance < thickness
                            if not contact:
                                common = ((aa - oa) + (bb - ob) + (cc - oc) + (dd - od)) * 0.25
                                bound = max(
                                    float(torch.linalg.norm(aa - oa - common)),
                                    float(torch.linalg.norm(bb - ob - common)),
                                ) + max(
                                    float(torch.linalg.norm(cc - oc - common)),
                                    float(torch.linalg.norm(dd - od - common)),
                                )
                                if distance > bound + thickness:
                                    continue
                                time = 0.0
                                for _ in range(24):
                                    if bound < 1e-9:
                                        break
                                    at = oa + (aa - oa) * time
                                    bt = ob + (bb - ob) * time
                                    ct = oc + (cc - oc) * time
                                    dt_v = od + (dd - od) * time
                                    gap_sq, gs, gt = _edge_edge_sq(at, bt, ct, dt_v)
                                    gap = math.sqrt(max(0, float(gap_sq)))
                                    if gap < thickness + 1e-5:
                                        s_f, t_f = gs, gt
                                        contact = True
                                        break
                                    time += (gap - thickness) / bound
                                    if time > 1:
                                        break
                            if not contact:
                                continue

                            s_val, t_val = float(s_f), float(t_f)
                            old_delta = (oa + (ob - oa) * s_val) - (oc + (od - oc) * t_val)
                            if float(torch.dot(old_delta, old_delta)) > 1e-12:
                                normal = _normalize(old_delta)
                            else:
                                normal = torch.linalg.cross(bb - aa, dd - cc)
                                if float(torch.linalg.norm(normal)) < 1e-12:
                                    continue
                                normal = normal / torch.linalg.norm(normal)
                            pa_pt = aa + (bb - aa) * s_val
                            pb_pt = cc + (dd - cc) * t_val
                            residual_snap = float(torch.dot(pa_pt - pb_pt, normal)) - thickness
                            if residual_snap >= 0:
                                continue
                            w0, w1, w2, w3 = 1 - s_val, s_val, 1 - t_val, t_val
                            denom_val = (
                                float(inv_mass[va]) * w0 * w0
                                + float(inv_mass[vb]) * w1 * w1
                                + float(inv_mass[c_idx]) * w2 * w2
                                + float(inv_mass[d_idx]) * w3 * w3
                            )
                            key = (distance, float(pi))
                            if denom_val < 1e-12 or key >= best_key:
                                continue
                            mobile = max(
                                float(inv_mass[va]) * w0,
                                float(inv_mass[vb]) * w1,
                                float(inv_mass[c_idx]) * w2,
                                float(inv_mass[d_idx]) * w3,
                            )
                            best_key = key
                            best_ids = (va, vb, c_idx, d_idx)
                            best_weights = (w0, w1, -w2, -w3)
                            best_normal_snap = normal.clone()
                            best_s = s_val
                            best_t = t_val

                    if best_ids is not None and best_normal_snap is not None:
                        selected_contacts.append({
                            "ids": best_ids,
                            "weights": best_weights,
                            "normal_snap": best_normal_snap,
                            "s": best_s,
                            "t": best_t,
                            "distance": best_key[0],
                            "kind": kind,
                        })

        # Phase 2: DIFFERENTIABLE correction using live positions
        # Recompute residual with live positions, then apply correction.
        # Gradient flows through the residual (depends on pos[ids]).
        if not selected_contacts:
            return pos

        delta_table = pos.new_zeros(N, 3)
        count_table = torch.zeros(N, dtype=torch.long, device=self.device)

        for ct in selected_contacts:
            ids = ct["ids"]
            weights = ct["weights"]
            normal_snap = ct["normal_snap"]
            kind = ct["kind"]

            if kind == 0:
                # VF contact: recompute residual with live positions
                v_id, ia, ib, ic = ids
                q_live, _, _, _ = _point_triangle_bary(pos[v_id], pos[ia], pos[ib], pos[ic])
                residual_live = torch.dot(pos[v_id] - q_live, normal_snap) - thickness
                if float(residual_live) >= 0:
                    continue
                denom_val = (
                    float(inv_mass[v_id])
                    + float(inv_mass[ia]) * weights[1] * weights[1]
                    + float(inv_mass[ib]) * weights[2] * weights[2]
                    + float(inv_mass[ic]) * weights[3] * weights[3]
                )
                if denom_val < 1e-12:
                    continue
                w0, u_val, vv_val = abs(float(weights[1])), abs(float(weights[2])), abs(float(weights[3]))
                mobile = max(
                    float(inv_mass[v_id]),
                    float(inv_mass[ia]) * w0,
                    float(inv_mass[ib]) * u_val,
                    float(inv_mass[ic]) * vv_val,
                )
                delta = normal_snap * (-residual_live / max(denom_val, mobile * 0.25))
            else:
                # EE contact: recompute residual with live positions
                va, vb, c_idx, d_idx = ids
                s_val = ct["s"]
                t_val = ct["t"]
                pa_pt = pos[va] + (pos[vb] - pos[va]) * s_val
                pb_pt = pos[c_idx] + (pos[d_idx] - pos[c_idx]) * t_val
                residual_live = torch.dot(pa_pt - pb_pt, normal_snap) - thickness
                if float(residual_live) >= 0:
                    continue
                w0, w1, w2, w3 = abs(float(weights[0])), abs(float(weights[1])), abs(float(weights[2])), abs(float(weights[3]))
                denom_val = (
                    float(inv_mass[va]) * w0 * w0
                    + float(inv_mass[vb]) * w1 * w1
                    + float(inv_mass[c_idx]) * w2 * w2
                    + float(inv_mass[d_idx]) * w3 * w3
                )
                if denom_val < 1e-12:
                    continue
                mobile = max(
                    float(inv_mass[va]) * w0,
                    float(inv_mass[vb]) * w1,
                    float(inv_mass[c_idx]) * w2,
                    float(inv_mass[d_idx]) * w3,
                )
                delta = normal_snap * (-residual_live / max(denom_val, mobile * 0.25))

            self.collision_records.append(CollisionRecord(
                vertex=ids[0], ids=ids, weights=weights,
                delta=tuple(float(d) for d in delta), distance=ct["distance"],
            ))

            for k in range(4):
                vid = ids[k]
                w = weights[k]
                if abs(w) > 1e-6 and float(inv_mass[vid]) > 0:
                    delta_table[vid] = delta_table[vid] + delta * (w * float(inv_mass[vid]))
                    count_table[vid] = count_table[vid] + 1

        # Apply corrections (differentiable: delta depends on pos via residual_live)
        for v in range(N):
            if count_table[v] > 0 and not pin_mask[v]:
                pos[v] = pos[v] + delta_table[v] / float(count_table[v])

        return pos

    def _apply_body_collision(
        self,
        pos: Tensor,
        prev_pos: Tensor,
        collider: Dict[str, Any],
        thickness: float,
        dt: float,
        pin_mask: Tensor,
    ) -> Tuple[Tensor, List[Dict[str, Any]]]:
        body_prev = collider["previous"]
        body_curr = collider["current"]
        body_tris = collider["triangles"]

        N = self.N
        contacts: List[Dict[str, Any]] = []

        # Phase 1: DETACHED candidate selection
        with torch.no_grad():
            best_per_vertex: Dict[int, Dict[str, Any]] = {}

            for v in range(N):
                if pin_mask[v]:
                    continue
                predicted = pos[v].detach()
                start = prev_pos[v].detach()
                best_dist = 1e30
                best_time = 2.0
                best_tri = -1
                best_normal = pos.new_zeros(3)
                best_bary = (0.0, 0.0, 0.0)
                best_q = pos.new_zeros(3)
                best_found = False
                nearest_dist = 1e30
                nearest_signed = 0.0
                nearest_normal = pos.new_zeros(3)
                nearest_tri = -1

                motion = predicted - start

                T_body = body_tris.shape[0]
                for t_idx in range(T_body):
                    ia = int(body_tris[t_idx, 0])
                    ib = int(body_tris[t_idx, 1])
                    ic = int(body_tris[t_idx, 2])
                    a, b, c = body_curr[ia], body_curr[ib], body_curr[ic]
                    oa, ob, oc = body_prev[ia], body_prev[ib], body_prev[ic]
                    face_n = torch.linalg.cross(b - a, c - a)
                    fn_len = float(torch.linalg.norm(face_n))
                    if fn_len < 1e-12:
                        continue
                    face_n = face_n / fn_len
                    q, u, vv, w0 = _point_triangle_bary(predicted, a, b, c)
                    distance = float(_vec3_len(predicted - q))
                    normal = _distance_normal(predicted - q, face_n, distance)
                    signed_dist = float(torch.dot(predicted - q, normal))
                    if distance < nearest_dist - 1e-6 or (
                        abs(distance - nearest_dist) <= 1e-6 and t_idx < nearest_tri
                    ):
                        nearest_dist = distance
                        nearest_tri = t_idx
                        nearest_signed = signed_dist
                        nearest_normal = normal.clone()

                    contact = False
                    bound = max(
                        float(torch.linalg.norm(a - oa - motion)),
                        float(torch.linalg.norm(b - ob - motion)),
                        float(torch.linalg.norm(c - oc - motion)),
                    )
                    time = 0.0
                    if distance <= bound + thickness + 1e-5:
                        for _ in range(32):
                            if bound < 1e-9 or contact:
                                break
                            at = oa + (a - oa) * time
                            bt = ob + (b - ob) * time
                            ct = oc + (c - oc) * time
                            pt = start + (predicted - start) * time
                            cl, bu_t, bv_t, _ = _point_triangle_bary(pt, at, bt, ct)
                            gap = float(_vec3_len(pt - cl))
                            if gap <= thickness + 1e-5:
                                contact = True
                                u, vv = float(bu_t), float(bv_t)
                                q = a + (b - a) * u + (c - a) * vv
                                break
                            time += (gap - thickness) / bound
                            if time > 1:
                                break
                    if not contact and distance < thickness:
                        contact = True
                        time = 0.0 if bound <= 1e-9 else 1.0

                    sd = float(torch.dot(predicted - q, normal))
                    earlier = time < best_time - 1e-6
                    same_time = abs(time - best_time) <= 1e-6
                    if contact and sd < thickness and (
                        earlier or (same_time and (
                            distance < best_dist - 1e-6
                            or (abs(distance - best_dist) <= 1e-6 and t_idx < best_tri)
                        ))
                    ):
                        best_dist = distance
                        best_time = time
                        best_tri = t_idx
                        best_normal = normal.clone()
                        best_bary = (float(w0), float(u), float(vv))
                        best_q = q.clone()
                        best_found = True

                if not best_found and nearest_signed < 0:
                    best_found = True
                    best_normal = nearest_normal.clone()
                    best_tri = nearest_tri
                    best_q = predicted - nearest_normal * (thickness - nearest_signed)

                if best_found:
                    best_per_vertex[v] = {
                        "tri": best_tri,
                        "normal": best_normal,
                        "bary": best_bary,
                        "q": best_q,
                    }

        # Phase 2: DIFFERENTIABLE correction using live positions
        for v, info in best_per_vertex.items():
            t_idx = info["tri"]
            normal = info["normal"]
            bary = info["bary"]
            q_snap = info["q"]

            ia = int(body_tris[t_idx, 0])
            ib = int(body_tris[t_idx, 1])
            ic = int(body_tris[t_idx, 2])

            # Recompute q with live body positions (body is fixed, so same)
            # But residual uses live cloth position
            q_live = body_curr[ia] * bary[0] + body_curr[ib] * bary[1] + body_curr[ic] * bary[2]
            signed_live = torch.dot(pos[v] - q_live, normal)
            if float(signed_live) >= thickness:
                continue

            pos[v] = pos[v] + normal * (thickness - signed_live)
            impulse = max(0, float(torch.dot(pos[v] - pos[v].detach() + pos[v].detach() - pos[v], normal)) / dt)
            # Recompute impulse properly
            predicted = pos[v].detach()  # position before correction was applied
            impulse = max(0, float(torch.dot(pos[v] - (predicted + normal * (thickness - signed_live - float(torch.dot(predicted - q_live, normal)))), normal)) / dt)
            # Simpler: impulse = (thickness - signed_live) / dt
            impulse = max(0, (thickness - float(signed_live)) / dt)

            body_vel_at = (
                (body_curr[ia] - body_prev[ia]) * bary[0]
                + (body_curr[ib] - body_prev[ib]) * bary[1]
                + (body_curr[ic] - body_prev[ic]) * bary[2]
            ) / dt

            contacts.append({
                "vertex": v,
                "normal": tuple(float(n) for n in normal),
                "bary": bary,
                "impulse": impulse,
                "body_vel": tuple(float(bv) for bv in body_vel_at),
                "recovery": info.get("recovery", False),
            })

        return pos, contacts

    def _apply_capsule_collision(
        self,
        pos: Tensor,
        prev_pos: Tensor,
        capsules: List[Dict[str, Any]],
        thickness: float,
        dt: float,
        pin_mask: Tensor,
        existing_contacts: List[Dict[str, Any]],
    ) -> Tuple[Tensor, List[Dict[str, Any]]]:
        N = self.N
        contact_index: Dict[int, int] = {}
        for i, ct in enumerate(existing_contacts):
            contact_index[ct["vertex"]] = i

        for v in range(N):
            if pin_mask[v]:
                continue
            for capsule in capsules:
                ca = capsule["currentA"]
                cb = capsule["currentB"]
                pa = capsule["previousA"]
                pb = capsule["previousB"]
                radius = capsule["radius"]
                eff_radius = radius + thickness

                # DETACHED candidate check
                with torch.no_grad():
                    q_cap, along = _capsule_closest(pos[v].detach(), ca, cb)
                    old_q, _ = _capsule_closest(prev_pos[v].detach(), pa, pb)
                    dist_cap = float(torch.linalg.norm(pos[v].detach() - q_cap))
                    old_gap = float(torch.linalg.norm(prev_pos[v].detach() - old_q)) - eff_radius
                    motion = pos[v].detach() - prev_pos[v].detach()
                    bound = max(
                        float(torch.linalg.norm(ca - pa - motion)),
                        float(torch.linalg.norm(cb - pb - motion)),
                    )
                    hit = dist_cap < eff_radius
                    recovery = old_gap < 0

                    if bound > 1e-9 and dist_cap <= bound + eff_radius:
                        time = 0.0
                        for _ in range(32):
                            a_t = pa + (ca - pa) * time
                            b_t = pb + (cb - pb) * time
                            pt = prev_pos[v].detach() + (pos[v].detach() - prev_pos[v].detach()) * time
                            feat, t_along = _capsule_closest(pt, a_t, b_t)
                            gap = float(torch.linalg.norm(pt - feat)) - eff_radius
                            if gap <= 1e-5:
                                along = t_along
                                q_cap = ca + (cb - ca) * along
                                hit = True
                                break
                            time += gap / bound
                            if time > 1:
                                break

                if not hit:
                    continue

                # DIFFERENTIABLE correction: recompute with live positions
                # Normal points INWARD (from particle toward closest point on surface)
                q_live, along_live = _capsule_closest(pos[v], ca, cb)
                normal = q_live - pos[v]
                nl = float(torch.linalg.norm(normal))
                if nl < 1e-8:
                    normal = _capsule_closest(prev_pos[v], pa, pb)[0] - prev_pos[v]
                    nl = float(torch.linalg.norm(normal))
                if nl < 1e-8:
                    axis = cb - ca
                    normal = torch.linalg.cross(axis, pos.new_tensor([1.0, 0, 0]))
                    if float(torch.linalg.norm(normal)) < 1e-8:
                        normal = pos.new_tensor([0.0, 0, 1.0])
                normal = _normalize(normal)
                residual = torch.dot(pos[v] - q_live, normal) - eff_radius
                if float(residual) >= 0:
                    continue

                pos[v] = pos[v] - normal * residual
                impulse = max(0, float(-residual) / dt)
                body_vel = ((ca - pa) * (1 - float(along_live)) + (cb - pb) * float(along_live)) / dt

                ct_entry = {
                    "vertex": v,
                    "normal": tuple(float(n) for n in normal),
                    "impulse": impulse,
                    "body_vel": tuple(float(bv) for bv in body_vel),
                    "recovery": recovery,
                    "bary": (0.0, 0.0, 0.0),
                }
                if v in contact_index:
                    existing_contacts[contact_index[v]] = ct_entry
                else:
                    contact_index[v] = len(existing_contacts)
                    existing_contacts.append(ct_entry)

        return pos, existing_contacts

    def _recover_body_contacts(
        self,
        pos: Tensor,
        prev_pos: Tensor,
        collider: Dict[str, Any],
        thickness: float,
        dt: float,
        pin_mask: Tensor,
        body_contacts: List[Dict[str, Any]],
    ) -> Tuple[Tensor, List[Dict[str, Any]]]:
        body_prev = collider["previous"]
        body_curr = collider["current"]
        body_tris = collider["triangles"]

        for _pass in range(2):
            changed = set()
            for ct in list(body_contacts):
                v = ct["vertex"]
                if pin_mask[v]:
                    continue

                # DETACHED: find nearest body triangle
                with torch.no_grad():
                    point = pos[v].detach()
                    nearest = 1e30
                    signed_dist = 0.0
                    selected_tri = -1
                    selected_normal = pos.new_zeros(3)
                    selected_bary = (0.0, 0.0, 0.0)

                    T_body = body_tris.shape[0]
                    for t_idx in range(T_body):
                        ia = int(body_tris[t_idx, 0])
                        ib = int(body_tris[t_idx, 1])
                        ic = int(body_tris[t_idx, 2])
                        a, b, c = body_curr[ia], body_curr[ib], body_curr[ic]
                        face_n = torch.linalg.cross(b - a, c - a)
                        fn_len = float(torch.linalg.norm(face_n))
                        if fn_len < 1e-12:
                            continue
                        face_n = face_n / fn_len
                        q, u, vv, w0 = _point_triangle_bary(point, a, b, c)
                        distance = float(_vec3_len(point - q))
                        normal = _distance_normal(point - q, face_n, distance)
                        sd = float(torch.dot(point - q, normal))
                        if distance < nearest - 1e-6 or (
                            abs(distance - nearest) <= 1e-6 and t_idx < selected_tri
                        ):
                            nearest = distance
                            selected_tri = t_idx
                            signed_dist = sd
                            selected_normal = normal.clone()
                            selected_bary = (float(w0), float(u), float(vv))

                if signed_dist >= -1e-6:
                    continue

                # DIFFERENTIABLE correction
                ia = int(body_tris[selected_tri, 0])
                ib = int(body_tris[selected_tri, 1])
                ic = int(body_tris[selected_tri, 2])
                q_live = body_curr[ia] * selected_bary[0] + body_curr[ib] * selected_bary[1] + body_curr[ic] * selected_bary[2]
                signed_live = torch.dot(pos[v] - q_live, selected_normal)
                if float(signed_live) >= -1e-6:
                    continue

                pos[v] = pos[v] + selected_normal * (thickness - signed_live)
                impulse = max(0, (thickness - float(signed_live)) / dt)
                body_vel = (
                    (body_curr[ia] - body_prev[ia]) * selected_bary[0]
                    + (body_curr[ib] - body_prev[ib]) * selected_bary[1]
                    + (body_curr[ic] - body_prev[ic]) * selected_bary[2]
                ) / dt

                ct["normal"] = tuple(float(n) for n in selected_normal)
                ct["impulse"] = impulse
                ct["body_vel"] = tuple(float(bv) for bv in body_vel)
                ct["recovery"] = True
                changed.add(v)

            if not changed:
                break

            # Re-apply capsule collision for changed vertices
            capsules = collider.get("capsules", [])
            if capsules:
                pos, body_contacts = self._apply_capsule_collision(
                    pos, prev_pos, capsules, thickness, dt, pin_mask, body_contacts
                )

        return pos, body_contacts


def _distance_normal(delta: Tensor, pseudo: Tensor, distance: float) -> Tensor:
    if distance > 1e-8:
        sign = -1.0 if float(torch.dot(delta, pseudo)) < 0 else 1.0
        return delta * (sign / distance)
    return pseudo


def _capsule_closest(point: Tensor, a: Tensor, b: Tensor):
    axis = b - a
    norm = torch.dot(axis, axis)
    t = torch.clamp(torch.dot(point - a, axis) / max(float(norm), 1e-16), 0.0, 1.0)
    return a + axis * t, t
