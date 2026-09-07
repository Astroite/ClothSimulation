#include "demo_physics.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace mlcloth::demo;

namespace {

int g_failures = 0;
int g_tests = 0;

#define TEST(name) \
    do { \
        ++g_tests; \
        std::cout << "  " << name << " ... "; \
    } while (0)

#define PASS() std::cout << "ok\n"

#define FAIL(msg) do { \
    std::cout << "FAIL: " << msg << "\n"; \
    ++g_failures; \
    } while (0)

#define EXPECT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while (0)
#define EXPECT_NEAR(a, b, eps, msg) do { if (std::abs((a) - (b)) > (eps)) { FAIL(msg); return; } } while (0)

Mesh makeGridMesh() {
    Mesh mesh;
    mesh.rest = {
        {0, 0, 0}, {1, 0, 0}, {2, 0, 0},
        {0, 1, 0}, {1, 1, 0}, {2, 1, 0},
        {0, 2, 0}, {1, 2, 0}, {2, 2, 0}
    };
    mesh.triangles = {
        0, 1, 3,  1, 4, 3,
        1, 2, 4,  2, 5, 4,
        3, 4, 6,  4, 7, 6,
        4, 5, 7,  5, 8, 7
    };
    mesh.mass.assign(9, 1.0f);
    mesh.pinned.assign(9, 0);
    return mesh;
}

TriangleCollider makeFloorCollider() {
    TriangleCollider body;
    body.previous = { {-10, -1, -10}, {10, -1, -10}, {10, -1, 10}, {-10, -1, 10} };
    body.current = body.previous;
    body.triangles = { 0, 2, 1, 0, 3, 2 }; // outward normal is +Y
    return body;
}

bool isFinite3(Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

float pointTriangleDistSqBary_silent(Vec3 p, Vec3 a, Vec3 b, Vec3 c,
                                       Vec3& closest, float& bu, float& bv) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { closest = a; bu = 0; bv = 0; return dot(ap, ap); }
    const Vec3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { closest = b; bu = 1; bv = 0; return dot(bp, bp); }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        closest = a + ab * v; bu = v; bv = 0;
        const Vec3 d = p - closest; return dot(d, d);
    }
    const Vec3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { closest = c; bu = 0; bv = 1; return dot(cp, cp); }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        closest = a + ac * w; bu = 0; bv = w;
        const Vec3 d = p - closest; return dot(d, d);
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest = b + (c - b) * w; bu = 1.0f - w; bv = w;
        const Vec3 d = p - closest; return dot(d, d);
    }
    const float denom = 1.0f / (va + vb + vc);
    const float fv = vb * denom;
    const float fw = vc * denom;
    closest = a + ab * fv + ac * fw; bu = fv; bv = fw;
    const Vec3 d = p - closest; return dot(d, d);
}

float edgeEdgeDistSq(Vec3 a0, Vec3 a1, Vec3 b0, Vec3 b1, Vec3& pa, Vec3& pb) {
    const Vec3 d1 = a1 - a0;
    const Vec3 d2 = b1 - b0;
    const Vec3 r = a0 - b0;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);
    float s, t;
    if (a <= 1e-12f && e <= 1e-12f) { pa = a0; pb = b0; const Vec3 d = pa - pb; return dot(d, d); }
    if (a <= 1e-12f) { s = 0.0f; t = std::clamp(f / e, 0.0f, 1.0f); }
    else {
        const float c = dot(d1, r);
        if (e <= 1e-12f) { t = 0.0f; s = std::clamp(-c / a, 0.0f, 1.0f); }
        else {
            const float b_val = dot(d1, d2);
            const float denom = a * e - b_val * b_val;
            s = denom != 0.0f ? std::clamp((b_val * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;
            t = (b_val * s + f) / e;
            if (t < 0.0f) { t = 0.0f; s = std::clamp(-c / a, 0.0f, 1.0f); }
            else if (t > 1.0f) { t = 1.0f; s = std::clamp((b_val - c) / a, 0.0f, 1.0f); }
        }
    }
    pa = a0 + d1 * s;
    pb = b0 + d2 * t;
    const Vec3 d = pa - pb;
    return dot(d, d);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void testGravityFreeFall() {
    TEST("gravity free fall");
    Mesh mesh = makeGridMesh();
    mesh.pinned = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    PhysicsSolver solver;
    solver.build(mesh);

    PhysicsConfig config;
    config.gravity = -9.81f;
    config.iterations = 0;
    config.dampingPerSecond = 0.0f;

    std::vector<Vec3> pinTargets(9, Vec3{});
    float dt = 1.0f / 60.0f;

    for (int frame = 0; frame < 60; ++frame) {
        solver.step(dt, pinTargets, nullptr, nullptr, config);
    }

    const auto& pos = solver.positions();
    for (int i = 0; i < 9; ++i) {
        EXPECT(pos[i].y < -0.1f, "vertex " + std::to_string(i) + " should have fallen");
    }
    PASS();
}

void testPinnedInvariance() {
    TEST("pinned invariance");
    Mesh mesh = makeGridMesh();
    mesh.pinned = {1, 0, 0, 0, 0, 0, 0, 0, 1};

    PhysicsSolver solver;
    solver.build(mesh);

    PhysicsConfig config;
    config.gravity = -9.81f;
    config.iterations = 0;

    std::vector<Vec3> pinTargets = mesh.rest;
    float dt = 1.0f / 60.0f;

    solver.step(dt, pinTargets, nullptr, nullptr, config);

    const auto& pos = solver.positions();
    EXPECT_NEAR(pos[0].x, 0.0f, 1e-6f, "pinned vertex 0 x");
    EXPECT_NEAR(pos[0].y, 0.0f, 1e-6f, "pinned vertex 0 y");
    EXPECT_NEAR(pos[0].z, 0.0f, 1e-6f, "pinned vertex 0 z");
    EXPECT_NEAR(pos[8].x, 2.0f, 1e-6f, "pinned vertex 8 x");
    EXPECT_NEAR(pos[8].y, 2.0f, 1e-6f, "pinned vertex 8 y");
    EXPECT_NEAR(pos[8].z, 0.0f, 1e-6f, "pinned vertex 8 z");
    PASS();
}

void testNoMLInitialization() {
    TEST("no-ML initialization (nullptr guide)");
    Mesh mesh = makeGridMesh();

    PhysicsSolver solver;
    solver.build(mesh);

    PhysicsConfig config;
    config.guideCompliance = -1.0f;

    std::vector<Vec3> pinTargets(9, Vec3{});
    float dt = 1.0f / 60.0f;

    solver.step(dt, pinTargets, nullptr, nullptr, config);
    PASS();
}

void testTimeBasedDamping() {
    TEST("time-based damping (exp formula)");
    Mesh mesh = makeGridMesh();
    mesh.pinned = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    PhysicsSolver solver;
    solver.build(mesh);

    PhysicsConfig config;
    config.gravity = 0.0f;
    config.iterations = 0;
    config.dampingPerSecond = 1.0f;  // 1 per second

    std::vector<Vec3> vel(9, {1.0f, 0.0f, 0.0f});
    solver.reset(mesh.rest, vel);

    std::vector<Vec3> pinTargets(9, Vec3{});
    float dt = 1.0f;

    solver.step(dt, pinTargets, nullptr, nullptr, config);

    // exp(-1.0 * 1.0) = exp(-1) ≈ 0.3679
    const auto& velocities = solver.velocities();
    const float expected = std::exp(-1.0f);
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(velocities[i].x, expected, 1e-3f,
                    "damped velocity at vertex " + std::to_string(i));
    }
    PASS();
}

void testZeroDampingPreservesVelocity() {
    TEST("zero damping preserves velocity");
    Mesh mesh = makeGridMesh();
    mesh.pinned = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    PhysicsSolver solver;
    solver.build(mesh);

    PhysicsConfig config;
    config.gravity = 0.0f;
    config.iterations = 0;
    config.dampingPerSecond = 0.0f;

    std::vector<Vec3> vel(9, {1.0f, 2.0f, 3.0f});
    solver.reset(mesh.rest, vel);

    std::vector<Vec3> pinTargets(9, Vec3{});
    float dt = 1.0f / 60.0f;

    solver.step(dt, pinTargets, nullptr, nullptr, config);

    const auto& velocities = solver.velocities();
    for (int i = 0; i < 9; ++i) {
        // With zero damping, gravity=0, and iterations=0, velocity should be preserved
        EXPECT_NEAR(velocities[i].x, 1.0f, 1e-4f, "preserved vx at " + std::to_string(i));
        EXPECT_NEAR(velocities[i].y, 2.0f, 1e-4f, "preserved vy at " + std::to_string(i));
        EXPECT_NEAR(velocities[i].z, 3.0f, 1e-4f, "preserved vz at " + std::to_string(i));
    }
    PASS();
}

void testResetZerosVelocities() {
    TEST("reset(positions) zeros velocities");
    Mesh mesh = makeGridMesh();

    PhysicsSolver solver;
    solver.build(mesh);

    // First set some velocity
    std::vector<Vec3> vel(9, {5.0f, 5.0f, 5.0f});
    solver.reset(mesh.rest, vel);
    {
        const auto& v = solver.velocities();
        EXPECT_NEAR(v[0].x, 5.0f, 1e-6f, "velocity set before reset");
    }

    // Now reset with positions only — should zero velocities
    solver.reset(mesh.rest);
    {
        const auto& v = solver.velocities();
        for (int i = 0; i < 9; ++i) {
            EXPECT_NEAR(v[i].x, 0.0f, 1e-6f, "zeroed vx at " + std::to_string(i));
            EXPECT_NEAR(v[i].y, 0.0f, 1e-6f, "zeroed vy at " + std::to_string(i));
            EXPECT_NEAR(v[i].z, 0.0f, 1e-6f, "zeroed vz at " + std::to_string(i));
        }
    }
    PASS();
}

void testRestPoseInvarianceWithoutGravity() {
    TEST("rest pose invariance without gravity");
    Mesh mesh = makeGridMesh();
    mesh.pinned = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    PhysicsSolver solver;
    solver.build(mesh);

    PhysicsConfig config;
    config.gravity = 0.0f;
    config.iterations = 1;
    config.dampingPerSecond = 0.0f;

    std::vector<Vec3> pinTargets(9, Vec3{});
    float dt = 1.0f / 60.0f;

    solver.step(dt, pinTargets, nullptr, nullptr, config);

    const auto& pos = solver.positions();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(pos[i].x, mesh.rest[i].x, 1e-5f, "rest x at vertex " + std::to_string(i));
        EXPECT_NEAR(pos[i].y, mesh.rest[i].y, 1e-5f, "rest y at vertex " + std::to_string(i));
        EXPECT_NEAR(pos[i].z, mesh.rest[i].z, 1e-5f, "rest z at vertex " + std::to_string(i));
    }
    PASS();
}

void testShearBendConstraintCorrection() {
    TEST("shear/bend constraint correction");
    Mesh mesh = makeGridMesh();
    mesh.pinned = {1, 0, 0, 0, 0, 0, 0, 0, 1};

    PhysicsSolver solver;
    solver.build(mesh);

    std::vector<Vec3> deformed = mesh.rest;
    deformed[4] = {1.0f, 0.5f, 0.5f};
    solver.reset(deformed);

    PhysicsConfig config;
    config.gravity = 0.0f;
    config.iterations = 10;
    config.stretchCompliance = 0.0f;
    config.shearCompliance = 0.0f;
    config.bendCompliance = 0.0f;

    std::vector<Vec3> pinTargets = mesh.rest;
    float dt = 1.0f / 60.0f;

    solver.step(dt, pinTargets, nullptr, nullptr, config);

    const auto& pos = solver.positions();
    float distBefore = length(deformed[4] - mesh.rest[4]);
    float distAfter = length(pos[4] - mesh.rest[4]);
    EXPECT(distAfter < distBefore, "constraint should reduce deformation");
    PASS();
}

void testShearConstraintGradientFD() {
    TEST("shear constraint gradient (finite difference)");
    // Set up a triangle with known rest dot product
    Mesh mesh;
    mesh.rest = { {0, 0, 0}, {1, 0, 0}, {0, 1, 0} };
    mesh.triangles = { 0, 1, 2 };
    mesh.mass = {1, 1, 1};
    mesh.pinned = {0, 0, 0};

    PhysicsSolver solver;
    solver.build(mesh);

    // Deform slightly
    std::vector<Vec3> pos = { {0.1f, 0.05f, 0}, {1.1f, 0.1f, 0.05f}, {-0.05f, 1.05f, 0.1f} };
    solver.reset(pos);

    // Get the shear constraints
    const auto& scs = solver.shearConstraints();
    EXPECT(!scs.empty(), "should have shear constraints");

    const auto& sc = scs[0];
    const float eps = 1e-5f;

    // Compute C at current position
    auto computeC = [&](Vec3 pa, Vec3 pb, Vec3 pc) -> float {
        Vec3 e1 = pb - pa;
        Vec3 e2 = pc - pa;
        return dot(e1, e2) - sc.restDot;
    };

    // FD gradient for each vertex component
    for (int vi = 0; vi < 3; ++vi) {
        uint32_t idx = (vi == 0) ? sc.a : (vi == 1) ? sc.b : sc.c;
        for (int comp = 0; comp < 3; ++comp) {
            Vec3 pPlus = pos[idx];
            Vec3 pMinus = pos[idx];
            if (comp == 0) { pPlus.x += eps; pMinus.x -= eps; }
            else if (comp == 1) { pPlus.y += eps; pMinus.y -= eps; }
            else { pPlus.z += eps; pMinus.z -= eps; }

            Vec3 pa = (vi == 0) ? pPlus : pos[sc.a];
            Vec3 pb = (vi == 1) ? pPlus : pos[sc.b];
            Vec3 pc = (vi == 2) ? pPlus : pos[sc.c];
            if (vi == 0) { pa = pPlus; } else if (vi == 1) { pb = pPlus; } else { pc = pPlus; }
            float Cplus = computeC(pa, pb, pc);

            pa = (vi == 0) ? pMinus : pos[sc.a];
            pb = (vi == 1) ? pMinus : pos[sc.b];
            pc = (vi == 2) ? pMinus : pos[sc.c];
            float Cminus = computeC(pa, pb, pc);

            float fdGrad = (Cplus - Cminus) / (2.0f * eps);

            // Analytic gradient
            Vec3 e1 = pos[sc.b] - pos[sc.a];
            Vec3 e2 = pos[sc.c] - pos[sc.a];
            Vec3 grad;
            if (vi == 0) grad = -(e1 + e2);  // dC/da
            else if (vi == 1) grad = e2;      // dC/db
            else grad = e1;                    // dC/dc

            float analyticGrad = (comp == 0) ? grad.x : (comp == 1) ? grad.y : grad.z;

            std::string label = "shear grad v" + std::to_string(vi) + " c" + std::to_string(comp);
            EXPECT_NEAR(fdGrad, analyticGrad, 1e-3f, label);
        }
    }
    PASS();
}

void testBendConstraintGradientFD() {
    TEST("bend constraint gradient (finite difference)");
    // Two triangles sharing edge (0,1), opposite vertices 2 and 3
    Mesh mesh;
    mesh.rest = {
        {0, 0, 0}, {1, 0, 0}, {0.5f, 1, 0}, {0.5f, 0, 1}
    };
    mesh.triangles = { 0, 1, 2, 0, 1, 3 };
    mesh.mass = {1, 1, 1, 1};
    mesh.pinned = {0, 0, 0, 0};

    PhysicsSolver solver;
    solver.build(mesh);

    const auto& bcs = solver.bendConstraints();
    EXPECT(!bcs.empty(), "should have bend constraints");

    const auto& bc = bcs[0];
    const float eps = 1e-5f;

    // Compute C at current position
    auto computeC = [&](Vec3 pa, Vec3 pb, Vec3 pc, Vec3 pd) -> float {
        Vec3 e = pb - pa;
        float eLen = length(e);
        if (eLen < 1e-12f) return 0.0f;
        Vec3 n1 = cross(e, pc - pa);
        Vec3 n2 = cross(e, pd - pa);
        float n1Len = length(n1);
        float n2Len = length(n2);
        if (n1Len < 1e-12f || n2Len < 1e-12f) return 0.0f;
        Vec3 n1hat = n1 / n1Len;
        Vec3 n2hat = n2 / n2Len;
        Vec3 ehat = e / eLen;
        float cosA = std::clamp(dot(n1hat, n2hat), -1.0f, 1.0f);
        float sinA = dot(cross(n1hat, n2hat), ehat);
        return std::atan2(sinA, cosA) - bc.restAngle;
    };

    std::vector<Vec3> pos = mesh.rest;
    // Perturb slightly to avoid degeneracy
    pos[2].y += 0.1f;
    pos[3].z += 0.1f;

    // FD gradient for each vertex component
    for (int vi = 0; vi < 4; ++vi) {
        uint32_t idx = static_cast<uint32_t>(vi == 0 ? bc.a : vi == 1 ? bc.b : vi == 2 ? bc.c : bc.d);
        for (int comp = 0; comp < 3; ++comp) {
            std::vector<Vec3> pPlus = pos;
            std::vector<Vec3> pMinus = pos;
            if (comp == 0) { pPlus[idx].x += eps; pMinus[idx].x -= eps; }
            else if (comp == 1) { pPlus[idx].y += eps; pMinus[idx].y -= eps; }
            else { pPlus[idx].z += eps; pMinus[idx].z -= eps; }

            float Cplus = computeC(pPlus[bc.a], pPlus[bc.b], pPlus[bc.c], pPlus[bc.d]);
            float Cminus = computeC(pMinus[bc.a], pMinus[bc.b], pMinus[bc.c], pMinus[bc.d]);
            float fdGrad = (Cplus - Cminus) / (2.0f * eps);

            // Analytic gradient
            Vec3 e = pos[bc.b] - pos[bc.a];
            float eLen = length(e);
            Vec3 n1 = cross(e, pos[bc.c] - pos[bc.a]);
            Vec3 n2 = cross(e, pos[bc.d] - pos[bc.a]);
            float n1Len = length(n1);
            float n2Len = length(n2);
            float hc = n1Len / eLen;
            float hd = n2Len / eLen;
            Vec3 n1hat = n1 / n1Len;
            Vec3 n2hat = n2 / n2Len;
            float tc = dot(pos[bc.c] - pos[bc.a], e) / dot(e, e);
            float td = dot(pos[bc.d] - pos[bc.a], e) / dot(e, e);

            Vec3 grad_c = -n1hat / hc;
            Vec3 grad_d = n2hat / hd;
            Vec3 grad_a = grad_c * (-(1.0f - tc)) + grad_d * (-(1.0f - td));
            Vec3 grad_b = grad_c * (-tc) + grad_d * (-td);

            Vec3 grad;
            if (vi == 0) grad = grad_a;
            else if (vi == 1) grad = grad_b;
            else if (vi == 2) grad = grad_c;
            else grad = grad_d;

            float analyticGrad = (comp == 0) ? grad.x : (comp == 1) ? grad.y : grad.z;

            std::string label = "bend grad v" + std::to_string(vi) + " c" + std::to_string(comp);
            EXPECT_NEAR(fdGrad, analyticGrad, 1e-2f, label);
        }
    }
    PASS();
}

void testCoarseGuidePreservingDetail() {
    TEST("coarse guide preserving detail/history");
    Mesh mesh = makeGridMesh();
    mesh.pinned = {1, 0, 0, 0, 0, 0, 0, 0, 1};

    PhysicsSolver solver;
    solver.build(mesh);

    std::vector<Vec3> initial = mesh.rest;
    initial[4] = {1.0f, 1.0f, 0.5f};
    solver.reset(initial);

    std::vector<Vec3> guide = mesh.rest;

    PhysicsConfig config;
    config.gravity = 0.0f;
    config.iterations = 5;
    config.guideCompliance = 0.1f;

    std::vector<Vec3> pinTargets = mesh.rest;
    float dt = 1.0f / 60.0f;

    solver.step(dt, pinTargets, &guide, nullptr, config);

    const auto& pos = solver.positions();
    float distFromGuide = length(pos[4] - guide[4]);

    EXPECT(distFromGuide < length(initial[4] - guide[4]),
           "guide should pull towards rest");
    PASS();
}

void testPatchCSRIntegrity() {
    TEST("patch CSR integrity (offsets == patchCount+1, all valid)");
    Mesh mesh = makeGridMesh();

    PhysicsSolver solver;
    solver.build(mesh);

    const auto& offsets = solver.patchOffsets();
    const auto& verts = solver.patchVertices();
    uint32_t pc = solver.patchCount();

    EXPECT(offsets.size() == pc + 1,
           "offsets size should be patchCount+1, got " + std::to_string(offsets.size()));

    for (uint32_t p = 0; p < pc; ++p) {
        uint32_t begin = offsets[p];
        uint32_t end = offsets[p + 1];
        EXPECT(begin <= end, "patch " + std::to_string(p) + " begin > end");
        EXPECT(end <= verts.size(), "patch " + std::to_string(p) + " end out of range");
        for (uint32_t i = begin; i < end; ++i) {
            EXPECT(verts[i] < mesh.rest.size(),
                   "patch vertex index out of range at " + std::to_string(i));
        }
    }
    PASS();
}

void testPatchBoundedSize() {
    TEST("patch bounded size (<=32 vertices)");
    Mesh mesh = makeGridMesh();

    PhysicsSolver solver;
    solver.build(mesh);

    const auto& offsets = solver.patchOffsets();
    uint32_t pc = solver.patchCount();

    for (uint32_t p = 0; p < pc; ++p) {
        uint32_t size = offsets[p + 1] - offsets[p];
        // Pinned patches are 1 vertex; non-pinned should be <= 32
        if (size > 1) {
            EXPECT(size <= 32, "patch " + std::to_string(p) + " has " +
                   std::to_string(size) + " vertices, max 32");
        }
    }
    PASS();
}

void testMovingPlaneContactCrossing() {
    TEST("moving plane contact crossing");
    Mesh mesh = makeGridMesh();
    mesh.pinned = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    PhysicsSolver solver;
    solver.build(mesh);

    PhysicsConfig config;
    config.gravity = 0.0f;
    config.iterations = 0;
    config.enableCollision = true;
    config.thickness = 0.1f;

    // Floor moving from below vertex to above vertex (crossing test)
    TriangleCollider body;
    body.previous = { {-10, -0.2f, -10}, {10, -0.2f, -10}, {10, -0.2f, 10}, {-10, -0.2f, 10} };
    body.current = { {-10, 0.1f, -10}, {10, 0.1f, -10}, {10, 0.1f, 10}, {-10, 0.1f, 10} };
    body.triangles = { 0, 2, 1, 0, 3, 2 }; // outward normal is +Y

    std::vector<Vec3> pinTargets(9, Vec3{});
    float dt = 1.0f / 60.0f;

    solver.step(dt, pinTargets, nullptr, &body, config);

    const auto& pos = solver.positions();
    for (int i = 0; i < 9; ++i) {
        EXPECT(pos[i].y >= 0.2f - 1e-5f,
               "vertex " + std::to_string(i) + " should be above floor");
    }

    // Check that contacts were actually resolved
    EXPECT(solver.stats().contactsResolved > 0, "should have resolved contacts");
    PASS();
}

void testSelfCollisionVF() {
    TEST("self-collision vertex-face (quantitative)");
    Mesh mesh;
    mesh.rest = {
        {0, 0, 0}, {1, 0, 0}, {0.5f, 1, 0},
        {0.5f, 0.5f, 0.5f}
    };
    mesh.triangles = { 0, 1, 2 };
    mesh.mass = {1, 1, 1, 1};
    mesh.pinned = {0, 0, 0, 0};

    PhysicsSolver solver;
    solver.build(mesh);

    // Move vertex 3 very close to triangle face
    std::vector<Vec3> positions = mesh.rest;
    positions[3] = {0.5f, 0.3f, 0.01f};
    solver.reset(positions);

    PhysicsConfig config;
    config.gravity = 0.0f;
    config.iterations = 0;
    config.enableSelfCollision = true;
    config.thickness = 0.05f;

    std::vector<Vec3> pinTargets(4, Vec3{});
    float dt = 1.0f / 60.0f;

    // Compute distance before
    Vec3 closestBefore;
    float bu, bv;
    float distBeforeSq = pointTriangleDistSqBary_silent(
        positions[3], positions[0], positions[1], positions[2], closestBefore, bu, bv);

    solver.step(dt, pinTargets, nullptr, nullptr, config);

    const auto& pos = solver.positions();

    // Compute distance after
    Vec3 closestAfter;
    float distAfterSq = pointTriangleDistSqBary_silent(
        pos[3], pos[0], pos[1], pos[2], closestAfter, bu, bv);

    // Distance should have increased (vertex pushed away)
    EXPECT(std::sqrt(distAfterSq) > std::sqrt(distBeforeSq) - 1e-6f,
           "vertex should be pushed away from triangle");

    // All positions should be finite
    for (int i = 0; i < 4; ++i) {
        EXPECT(isFinite3(pos[i]), "position at " + std::to_string(i) + " should be finite");
    }
    PASS();
}

void testSelfCollisionEE() {
    TEST("self-collision edge-edge (quantitative, non-adjacent triangles)");
    // Two non-adjacent triangles with nearby crossing edges
    Mesh mesh;
    mesh.rest = {
        {0, 0, 0}, {1, 0, 0}, {0.5f, 1, 0},      // triangle 0
        {0, 0, 0.5f}, {1, 0, 0.5f}, {0.5f, 1, 0.5f}  // triangle 1 (non-adjacent)
    };
    mesh.triangles = { 0, 1, 2, 3, 4, 5 };
    mesh.mass = {1, 1, 1, 1, 1, 1};
    mesh.pinned = {0, 0, 0, 0, 0, 0};

    PhysicsSolver solver;
    solver.build(mesh);

    // Move vertices so edges 0-1 and 3-4 cross in proximity
    std::vector<Vec3> positions = mesh.rest;
    positions[0] = {0, 0, 0.2f};
    positions[1] = {1, 0, 0.2f};
    positions[3] = {0.5f, -0.5f, 0};
    positions[4] = {0.5f, 0.5f, 0.4f};
    solver.reset(positions);

    // Compute closest edge-edge distance before
    Vec3 paBefore, pbBefore;
    float distBeforeSq = edgeEdgeDistSq(positions[0], positions[1], positions[3], positions[4],
                                          paBefore, pbBefore);
    float distBefore = std::sqrt(distBeforeSq);

    PhysicsConfig config;
    config.gravity = 0.0f;
    config.iterations = 0;
    config.enableSelfCollision = true;
    config.thickness = 0.3f;

    std::vector<Vec3> pinTargets(6, Vec3{});
    float dt = 1.0f / 60.0f;

    solver.step(dt, pinTargets, nullptr, nullptr, config);

    const auto& pos = solver.positions();

    // Compute closest edge-edge distance after
    Vec3 paAfter, pbAfter;
    float distAfterSq = edgeEdgeDistSq(pos[0], pos[1], pos[3], pos[4], paAfter, pbAfter);
    float distAfter = std::sqrt(distAfterSq);

    // Closest distance should have increased (edges pushed apart)
    EXPECT(distAfter > distBefore - 1e-6f,
           "closest edge-edge distance should increase after self-collision");

    // All positions should be finite
    for (int i = 0; i < 6; ++i) {
        EXPECT(isFinite3(pos[i]), "position at " + std::to_string(i) + " should be finite");
    }

    // Self-collision contacts should have been resolved
    EXPECT(solver.stats().selfContactsResolved > 0, "should have resolved self-contacts");
    PASS();
}

void testDeterministicReset() {
    TEST("deterministic reset");
    Mesh mesh = makeGridMesh();

    PhysicsSolver solver;
    solver.build(mesh);

    PhysicsConfig config;
    config.gravity = -9.81f;
    config.iterations = 2;
    config.dampingPerSecond = 0.5f;

    std::vector<Vec3> pinTargets(9, Vec3{});
    std::vector<Vec3> zeroVel(9, Vec3{});
    float dt = 1.0f / 60.0f;

    solver.reset(mesh.rest, zeroVel);
    for (int i = 0; i < 10; ++i) solver.step(dt, pinTargets, nullptr, nullptr, config);
    std::vector<Vec3> result1 = solver.positions();

    solver.reset(mesh.rest, zeroVel);
    for (int i = 0; i < 10; ++i) solver.step(dt, pinTargets, nullptr, nullptr, config);
    std::vector<Vec3> result2 = solver.positions();

    for (int i = 0; i < 9; ++i) {
        if (std::abs(result1[i].x - result2[i].x) > 1e-6f ||
            std::abs(result1[i].y - result2[i].y) > 1e-6f ||
            std::abs(result1[i].z - result2[i].z) > 1e-6f) {
            std::cout << "\n  Mismatch at vertex " << i << ":\n";
            std::cout << "    run1: (" << result1[i].x << ", " << result1[i].y << ", " << result1[i].z << ")\n";
            std::cout << "    run2: (" << result2[i].x << ", " << result2[i].y << ", " << result2[i].z << ")\n";
            FAIL("deterministic at vertex " + std::to_string(i));
            return;
        }
    }
    PASS();
}

void testMalformedInput() {
    TEST("malformed input validation");
    PhysicsSolver solver;

    Mesh emptyMesh;
    bool threw = false;
    try { solver.build(emptyMesh); } catch (const std::runtime_error&) { threw = true; }
    EXPECT(threw, "should throw on empty mesh");

    Mesh badMesh;
    badMesh.rest = {{0, 0, 0}};
    badMesh.triangles = {0, 1, 2};
    badMesh.mass = {1};
    badMesh.pinned = {0, 0};
    threw = false;
    try { solver.build(badMesh); } catch (const std::runtime_error&) { threw = true; }
    EXPECT(threw, "should throw on mismatched pinned size");

    Mesh badMass;
    badMass.rest = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    badMass.triangles = {0, 1, 2};
    badMass.mass = {-1, 1, 1};
    badMass.pinned = {0, 0, 0};
    threw = false;
    try { solver.build(badMass); } catch (const std::runtime_error&) { threw = true; }
    EXPECT(threw, "should throw on negative mass");

    Mesh validMesh = makeGridMesh();
    solver.build(validMesh);

    threw = false;
    std::vector<Vec3> pinTargets(9, Vec3{});
    try { solver.step(-1.0f, pinTargets, nullptr, nullptr, PhysicsConfig{}); } catch (const std::runtime_error&) { threw = true; }
    EXPECT(threw, "should throw on negative dt");

    threw = false;
    std::vector<Vec3> badPins(5, Vec3{});
    try { solver.step(1.0f/60.0f, badPins, nullptr, nullptr, PhysicsConfig{}); } catch (const std::runtime_error&) { threw = true; }
    EXPECT(threw, "should throw on wrong pinTargets size");

    // Damping validation
    threw = false;
    PhysicsConfig badDamping;
    badDamping.dampingPerSecond = -1.0f;
    try { solver.step(1.0f/60.0f, pinTargets, nullptr, nullptr, badDamping); } catch (const std::runtime_error&) { threw = true; }
    EXPECT(threw, "should throw on negative dampingPerSecond");

    threw = false;
    PhysicsConfig nanDamping;
    nanDamping.dampingPerSecond = std::numeric_limits<float>::quiet_NaN();
    try { solver.step(1.0f/60.0f, pinTargets, nullptr, nullptr, nanDamping); } catch (const std::runtime_error&) { threw = true; }
    EXPECT(threw, "should throw on NaN dampingPerSecond");

    PASS();
}

void testNegativeDampingRejected() {
    TEST("negative damping rejected");
    Mesh mesh = makeGridMesh();
    PhysicsSolver solver;
    solver.build(mesh);

    PhysicsConfig config;
    config.dampingPerSecond = -0.5f;

    std::vector<Vec3> pinTargets(9, Vec3{});
    bool threw = false;
    try { solver.step(1.0f/60.0f, pinTargets, nullptr, nullptr, config); }
    catch (const std::runtime_error&) { threw = true; }
    EXPECT(threw, "should reject negative damping");
    PASS();
}

void testSweptSelfCollisionAndPins(){
    TEST("swept self VF, pins and isolated-contact momentum");
    Mesh mesh;mesh.rest={{-.2f,1,-.2f},{.2f,1,-.2f},{0,1,.2f},{0,1.05f,0}};
    mesh.triangles={0,2,1};mesh.mass={1,2,3,4};mesh.pinned={1,1,1,0};
    PhysicsSolver solver;solver.build(mesh);std::vector<Vec3> velocity(4);velocity[3].y=-24;
    solver.reset(mesh.rest,velocity);PhysicsConfig config;config.gravity=0;config.iterations=0;config.dampingPerSecond=0;config.enableSelfCollision=true;
    solver.step(1.f/240,mesh.rest,nullptr,nullptr,config);
    EXPECT(solver.positions()[3].y>=1.003f-1e-6f,"fast point crossed a fixed face");
    for(int i=0;i<3;++i)EXPECT(length(solver.positions()[i]-mesh.rest[i])<1e-7f,"self contact moved a pin");
    mesh.pinned={0,0,0,0};mesh.rest[3].y=1.0005f;solver.build(mesh);solver.reset(mesh.rest);
    solver.step(1.f/240,mesh.rest,nullptr,nullptr,config);
    Vec3 weightedCorrection{};for(int i=0;i<4;++i)weightedCorrection+=(solver.positions()[i]-mesh.rest[i])*mesh.mass[i];
    EXPECT(length(weightedCorrection)<1e-6f,"isolated contact violated mass-weighted momentum");
    EXPECT(solver.stats().selfContactsResolved==1,"isolated VF produced duplicate contacts");
    mesh.pinned={1,1,1,1};solver.build(mesh);solver.reset(mesh.rest);solver.step(1.f/240,mesh.rest,nullptr,nullptr,config);
    for(int i=0;i<4;++i)EXPECT(length(solver.positions()[i]-mesh.rest[i])<1e-7f,"all-pinned contact moved mesh");
    PASS();
}
void testSelfTopologyExclusion(){
    TEST("self contact excludes one-ring neighbors");
    Mesh mesh;mesh.rest={{0,1,0},{.1f,1,0},{0,1,.1f},{.02f,1.0001f,.02f}};
    mesh.triangles={0,1,2,0,2,3};mesh.mass.assign(4,1);mesh.pinned.assign(4,0);
    PhysicsSolver solver;solver.build(mesh);solver.reset(mesh.rest);
    PhysicsConfig config;config.gravity=0;config.iterations=0;config.enableSelfCollision=true;
    solver.step(1.f/240,mesh.rest,nullptr,nullptr,config);
    EXPECT(solver.stats().selfContactsResolved==0,"adjacent triangles generated self contact");
    for(int i=0;i<4;++i)EXPECT(length(solver.positions()[i]-mesh.rest[i])<1e-7f,"topology exclusion failed");
    PASS();
}
void testSubThicknessSampling(){
    TEST("dense cloth samples do not repel their own rest sheet");
    auto mesh=makeGridMesh();for(auto& v:mesh.rest)v=v*.001f;PhysicsSolver solver;solver.build(mesh);solver.reset(mesh.rest);
    PhysicsConfig config;config.gravity=0;config.iterations=0;config.enableSelfCollision=true;
    solver.step(1.f/240,mesh.rest,nullptr,nullptr,config);
    EXPECT(solver.stats().selfContactsResolved==0,"contact shell repelled geodesic neighbors");
    for(size_t i=0;i<mesh.rest.size();++i)EXPECT(length(solver.positions()[i]-mesh.rest[i])<1e-7f,"rest sheet inflated");
    PASS();
}

void testFixedEdgeBarycentricRoundoff(){
    TEST("fixed contact edge cannot launch the opposite free corner");
    Mesh mesh;
    mesh.rest={{-.0283006262f,1.3153568506f,-.0717698559f},
               {-.0212492384f,1.3111006021f,-.0663453862f},
               {-.0304723401f,1.3106176853f,-.0769380853f},
               {-.0178833846f,1.3271845579f,-.0613009743f}};
    mesh.triangles={1,2,3};mesh.pinned={1,0,1,1};
    mesh.mass={6.1768769e-6f,1.8012863e-5f,1.0518872e-5f,1.0642520e-5f};
    auto pins=mesh.rest;
    pins[0]={-.0283006225f,1.3153568506f,-.0717698187f};
    pins[2]={-.0304723401f,1.3106175661f,-.0769380480f};
    pins[3]={-.0178833809f,1.3271845579f,-.0613009371f};
    PhysicsSolver solver;solver.build(mesh);solver.reset(mesh.rest);
    PhysicsConfig config;config.gravity=0;config.iterations=0;config.enableSelfCollision=true;
    solver.step(1.f/240,pins,nullptr,nullptr,config);
    EXPECT(length(solver.positions()[1]-mesh.rest[1])<1e-6f,"negative roundoff barycentric weight launched the free corner");
    PASS();
}

void testSweptCapsuleContact(){
    TEST("moving leg capsules, crossing and embedded recovery");
    Mesh mesh;mesh.rest={{0,1,.2f},{1,1,0},{1,1,.1f}};mesh.triangles={0,1,2};mesh.mass={1,1,1};mesh.pinned={0,1,1};PhysicsSolver solver;solver.build(mesh);
    auto reset=[&](Vec3 p,Vec3 v=Vec3{}){auto points=mesh.rest;points[0]=p;std::vector<Vec3> velocity(3);velocity[0]=v;solver.reset(points,velocity);};
    TriangleCollider body;body.capsules.push_back({{0,.8f,0},{0,1.2f,0},{0,.8f,0},{0,1.2f,0},.05f});
    PhysicsConfig config;config.iterations=0;config.gravity=0;config.dampingPerSecond=0;config.enableCollision=true;
    reset(mesh.rest[0],{0,0,-96});solver.step(1.f/240,mesh.rest,nullptr,&body,config);
    EXPECT(solver.positions()[0].z>=.053f-1e-5f,"fast particle crossed a leg capsule");
    reset({0,1,.01f});solver.step(1.f/240,mesh.rest,nullptr,&body,config);
    EXPECT(std::abs(solver.positions()[0].z-.053f)<1e-5f,"embedded particle did not leave capsule");
    EXPECT(length(solver.velocities()[0])<1e-5f,"embedded capsule recovery injected energy");
    body.capsules={{{-.1f,.8f,0},{-.1f,1.2f,0},{.1f,.8f,0},{.1f,1.2f,0},.03f}};
    reset({0,1,0});solver.step(1.f/240,mesh.rest,nullptr,&body,config);
    EXPECT(solver.positions()[0].x>=.133f-1e-5f,"moving capsule swept through a resting particle");
    PASS();
}

void testConcaveBodyRecovery(){
    TEST("swept contact exits both faces of a concave body corner");
    Mesh mesh;mesh.rest={{.1f,1,.1f},{2,1,2},{2.1f,1,2}};mesh.triangles={0,1,2};mesh.mass={1,1,1};mesh.pinned={0,1,1};
    TriangleCollider body;body.current={{-1,.5f,-1},{1,.5f,-1},{1,.5f,0},{0,.5f,0},{0,.5f,1},{-1,.5f,1}};
    for(int i=0;i<6;++i){auto p=body.current[i];p.y=1.5f;body.current.push_back(p);}body.previous=body.current;
    const uint32_t cap[]={0,1,3,1,2,3,0,3,5,3,4,5};
    for(int i=0;i<12;i+=3){body.triangles.insert(body.triangles.end(),{cap[i],cap[i+1],cap[i+2],cap[i]+6,cap[i+2]+6,cap[i+1]+6});}
    for(uint32_t a=0;a<6;++a){auto b=(a+1)%6;body.triangles.insert(body.triangles.end(),{a,a+6,b,b,a+6,b+6});}
    PhysicsSolver solver;solver.build(mesh);std::vector<Vec3> velocity(3);velocity[0]={-48,0,-48};solver.reset(mesh.rest,velocity);
    PhysicsConfig config;config.gravity=0;config.iterations=0;config.dampingPerSecond=0;config.enableCollision=true;
    solver.step(1.f/240,mesh.rest,nullptr,&body,config);auto result=solver.positions()[0];
    EXPECT(result.x>=.003f-1e-5f&&result.z>=.003f-1e-5f,"single swept plane left the particle inside the concave body");
    PASS();
}

void testRelativeSweptTranslation(){
    TEST("common root translation preserves swept self contact");
    Mesh mesh;mesh.rest={{-.1f,1,-.1f},{.1f,1,-.1f},{0,1,.1f},{0,1.05f,0}};mesh.triangles={0,2,1};mesh.mass.assign(4,1);mesh.pinned={1,1,1,0};
    PhysicsSolver baseline,translated;baseline.build(mesh);translated.build(mesh);
    std::vector<Vec3> velocity(4);velocity[3]={0,-24,0};baseline.reset(mesh.rest,velocity);
    PhysicsConfig config;config.gravity=0;config.iterations=0;config.dampingPerSecond=0;config.enableSelfCollision=true;
    baseline.step(1.f/240,mesh.rest,nullptr,nullptr,config);
    const Vec3 shift{.4f,.3f,-.2f};velocity[3]+=shift*240;translated.reset(mesh.rest,velocity);
    auto pins=mesh.rest;for(auto& p:pins)p+=shift;translated.step(1.f/240,pins,nullptr,nullptr,config);
    for(size_t i=0;i<pins.size();++i)EXPECT(length(translated.positions()[i]-shift-baseline.positions()[i])<1e-5f,"common translation changed the contact result");
    EXPECT(baseline.positions()[3].y>=1.003f-1e-5f,"relative bound missed the swept face");
    PASS();
}

void testGeodesicTethers(){
    TEST("geodesic tethers preserve folded slack and disconnected free islands");
    Mesh mesh;mesh.rest={{0,1,0},{1,1,0},{1,2,0},{0,2,0},{0,2,.1f},{3,1,0},{3.1f,1,0},{3,1,.1f}};
    mesh.triangles={0,1,2,2,3,4,5,6,7};mesh.mass.assign(8,1);mesh.pinned={1,0,0,0,0,0,0,0};
    PhysicsSolver solver;solver.build(mesh);
    const float path=std::sqrt(2.f)+std::sqrt(1.01f);
    EXPECT_NEAR(solver.tethers()[4][0].length,path,1e-6f,"rest limit must follow cloth edges, not a shortcut across folds");
    EXPECT(solver.tethers()[4][0].anchor==0&&solver.tethers()[4][1].length==0,"invalid extra tether");
    EXPECT(solver.tethers()[5][0].length==0,"unattached component linked through space");
    PhysicsConfig config;config.gravity=0;config.dampingPerSecond=0;config.iterations=1;
    config.stretchCompliance=config.shearCompliance=config.bendCompliance=1e20f;config.enableTethers=true;config.tetherScale=1;
    auto points=mesh.rest;points[4]={0,1,path-.1f};solver.reset(points);solver.step(1.f/240,mesh.rest,nullptr,nullptr,config);
    EXPECT(length(solver.positions()[4]-points[4])<1e-6f,"slack folded particle was pulled toward rest");
    points[4]={0,1,path+1};points[5].y=2;solver.reset(points);solver.step(1.f/240,mesh.rest,nullptr,nullptr,config);
    EXPECT_NEAR(length(solver.positions()[4]-mesh.rest[0]),path,1e-6f,"single tether did not cap extension in one pass");
    EXPECT(length(solver.positions()[5]-points[5])<1e-6f,"unattached island changed");
    config.enableTethers=false;solver.reset(points);solver.step(1.f/240,mesh.rest,nullptr,nullptr,config);
    EXPECT(length(solver.positions()[4]-points[4])<1e-6f,"disabled tethers still applied");
    PASS();
}
void testTetherRegions(){
    TEST("tethers use distinct pin regions and current skinned targets");
    Mesh mesh=makeGridMesh();mesh.pinned[0]=mesh.pinned[1]=mesh.pinned[8]=1;
    PhysicsSolver solver;solver.build(mesh);auto slots=solver.tethers()[4];
    EXPECT(slots[0].length>0&&slots[1].length>0&&slots[2].length==0,"adjacent pins consumed multiple source regions");
    EXPECT(slots[0].anchor==1&&slots[1].anchor==8,"nearest source in each region was not selected");
    auto pins=mesh.rest;const Vec3 shift{.4f,.3f,-.2f};for(auto& p:pins)p+=shift;
    auto points=pins;points[4].z+=10;solver.reset(points);
    PhysicsConfig config;config.gravity=0;config.dampingPerSecond=0;config.iterations=8;config.enableTethers=true;
    config.stretchCompliance=config.shearCompliance=config.bendCompliance=1e20f;
    solver.step(1.f/240,pins,nullptr,nullptr,config);
    EXPECT(length(solver.positions()[4]-pins[1])<=slots[0].length*config.tetherScale+.01f,"kinematic target was not used");
    EXPECT(length(solver.positions()[0]-pins[0])<1e-6f,"tether moved a pinned source");
    bool rejected=false;config.tetherScale=.5f;try{solver.step(1.f/240,pins,nullptr,nullptr,config);}catch(...){rejected=true;}
    EXPECT(rejected,"invalid scale accepted");PASS();
}

void testBodyFeatureSigns(){
    for(bool vertex:{false,true}){
        TEST((vertex?"convex vertex pseudonormal prevents a false body interior":"convex edge pseudonormal prevents a false body interior"));
        TriangleCollider body;
        if(vertex){body.current={{0,1,0},{-1,1.2f,.2f},{-1,.8f,.2f},{-1,1,-.2f}};body.triangles={0,2,3,0,1,2,0,3,1,1,3,2};}
        else{body.current={{0,1,-.2f},{-1,1.2f,-.2f},{-1,.8f,-.2f},{0,1,.2f},{-1,1.2f,.2f},{-1,.8f,.2f}};
            body.triangles={0,3,2,2,3,5,0,1,3,1,4,3,1,2,4,2,5,4,0,2,1,3,4,5};}
        Vec3 center{};for(auto p:body.current)center+=p;center=center/static_cast<float>(body.current.size());
        for(size_t t=0;t<body.triangles.size();t+=3){auto a=body.current[body.triangles[t]],b=body.current[body.triangles[t+1]],c=body.current[body.triangles[t+2]];
            if(dot(cross(b-a,c-a),(a+b+c)/3.f-center)<0)std::swap(body.triangles[t+1],body.triangles[t+2]);}
        body.previous=body.current;const Vec3 point{.04f,1.1f,vertex?.03f:0};
        Mesh mesh;mesh.rest={point,{2,1,0},{2,1,.1f}};mesh.triangles={0,1,2};mesh.mass={1,1,1};mesh.pinned={0,1,1};
        PhysicsSolver solver;solver.build(mesh);PhysicsConfig config;config.iterations=0;config.gravity=0;config.enableCollision=true;
        solver.step(1.f/240,mesh.rest,nullptr,&body,config);
        EXPECT(length(solver.positions()[0]-point)<1e-6f,"outside cloth was projected by an arbitrary closest face normal");
        EXPECT(length(solver.velocities()[0])<1e-6f,"false feature contact injected velocity");PASS();
    }
}

void testFirstBodyImpact(){
    TEST("earliest body impact wins when the endpoint is near the opposite face");
    TriangleCollider body;body.current={{-.3f,.7f,-.02f},{.3f,.7f,-.02f},{.3f,1.3f,-.02f},{-.3f,1.3f,-.02f},
        {-.3f,.7f,.02f},{.3f,.7f,.02f},{.3f,1.3f,.02f},{-.3f,1.3f,.02f}};body.previous=body.current;
    body.triangles={0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5};
    Mesh mesh;mesh.rest={{0,1,.08f},{2,1,0},{2,1,.1f}};mesh.triangles={0,1,2};mesh.mass={1,1,1};mesh.pinned={0,1,1};
    PhysicsSolver solver;solver.build(mesh);std::vector<Vec3> velocity(3);velocity[0].z=-23.76f;solver.reset(mesh.rest,velocity);
    PhysicsConfig config;config.iterations=0;config.gravity=0;config.dampingPerSecond=0;config.enableCollision=true;
    solver.step(1.f/240,mesh.rest,nullptr,&body,config);
    EXPECT(solver.positions()[0].z>=.023f-1e-5f,"nearby exit face replaced the earlier entrance contact");PASS();
}

void testContactAwareGuide(){
    for(int kind=0;kind<3;++kind){
        TEST((kind==0?"coarse guide respects ground proximity":kind==1?"coarse guide respects STM proximity":"coarse guide respects capsule proximity"));
        const float surface=kind==0?0.f:1.f;
        Mesh mesh;mesh.rest={{0,surface+.009f,0},{.001f,surface+.009f,0},{0,surface+.009f,.001f}};
        mesh.triangles={0,2,1};mesh.mass={.01f,.01f,.01f};mesh.pinned={0,0,0};
        TriangleCollider body;if(kind==1){body.current={{-1,1,-1},{-1,1,1},{1,1,-1},{1,1,1}};body.previous=body.current;body.triangles={0,1,2,2,1,3};}
        if(kind==2)body.capsules={{{-1,.94f,0},{1,.94f,0},{-1,.94f,0},{1,.94f,0},.06f}};
        PhysicsSolver aware,baseline;aware.build(mesh);baseline.build(mesh);
        PhysicsConfig config;config.gravity=0;config.dampingPerSecond=0;config.enableCollision=true;config.guideCompliance=.001f;
        for(Vec3 shift:{Vec3{0,-.02f,0},Vec3{0,.02f,0},Vec3{.02f,0,0},Vec3{0,.03f,0}}){
            auto guide=mesh.rest;for(auto& v:guide)v+=shift;if(shift.y==.03f)guide[0].y-=.06f;const auto original=guide;
            aware.reset(mesh.rest);baseline.reset(mesh.rest);config.contactAwareGuide=true;
            aware.step(1.f/240,mesh.rest,&guide,kind?&body:nullptr,config);config.contactAwareGuide=false;
            baseline.step(1.f/240,mesh.rest,&guide,kind?&body:nullptr,config);
            if(shift.y<0)EXPECT(aware.positions()[0].y>baseline.positions()[0].y+1e-5f,"inward coarse guide was not reduced before physical contact");
            else EXPECT(length(aware.positions()[0]-baseline.positions()[0])<1e-6f,"outward/tangent guidance was unnecessarily weakened");
            EXPECT(length(guide[0]-original[0])==0,"contact correction modified ML source");
        }
        auto guide=mesh.rest;for(auto& v:guide)v.y-=.02f;
        aware.reset(mesh.rest);baseline.reset(mesh.rest);config.enableCollision=false;config.contactAwareGuide=true;
        aware.step(1.f/240,mesh.rest,&guide,kind?&body:nullptr,config);config.contactAwareGuide=false;
        baseline.step(1.f/240,mesh.rest,&guide,kind?&body:nullptr,config);
        EXPECT(length(aware.positions()[0]-baseline.positions()[0])<1e-6f,"disabled collision retained stale guide confidence");PASS();
    }
}

}  // namespace

int main() {
    std::cout << std::unitbuf;
    std::cout << "Demo physics tests:\n";

    testGravityFreeFall();
    testPinnedInvariance();
    testNoMLInitialization();
    testTimeBasedDamping();
    testZeroDampingPreservesVelocity();
    testResetZerosVelocities();
    testRestPoseInvarianceWithoutGravity();
    testShearBendConstraintCorrection();
    testShearConstraintGradientFD();
    testBendConstraintGradientFD();
    testCoarseGuidePreservingDetail();
    testPatchCSRIntegrity();
    testPatchBoundedSize();
    testMovingPlaneContactCrossing();
    testSelfCollisionVF();
    testSelfCollisionEE();
    testSweptSelfCollisionAndPins();
    testSelfTopologyExclusion();
    testSubThicknessSampling();
    testFixedEdgeBarycentricRoundoff();
    testSweptCapsuleContact();
    testConcaveBodyRecovery();
    testRelativeSweptTranslation();
    testGeodesicTethers();
    testTetherRegions();
    testBodyFeatureSigns();
    testFirstBodyImpact();
    testContactAwareGuide();
    testDeterministicReset();
    testMalformedInput();
    testNegativeDampingRejected();

    std::cout << "\n" << g_tests << " tests, " << g_failures << " failures\n";
    return g_failures > 0 ? 1 : 0;
}
