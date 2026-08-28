#pragma once
// Soft-guided XPBD projection on the CPU, for the MLCloth front end.
//
// Why the CPU rather than a compute shader
// ----------------------------------------
// The measured split on this machine is MNN CPU inference 2.558 ms median against
// 0.0174 ms for the whole GPU pass (5,294 vertices, one inference thread). The CPU is
// the bottleneck and the GPU is 99.3% idle, so a compute-shader solve would be close to
// free while a CPU solve is additive -- that is the real argument for the GPU, and it is
// not being dismissed. It is deferred, for three reasons:
//
//   * The open question in this PoC is calibration, not throughput. Sweeping
//     {rest, chaos, driver} x quantile across 86 clips is a loop here and a windowed
//     Vulkan session there.
//   * This becomes the oracle for a later port. The two cross-implementation checks in
//     this PoC (Python/C++ topology, offline/Unreal drivers) each caught a real bug that
//     a single implementation would have shipped.
//   * A CPU port can match the Python reference to near rounding error, which a GPU port
//     cannot: `hood_xpbd.comp` records that its 1.68e-3 gap against Python is not a
//     summation artifact but the one-sided `max(residual, 0)` and the contact's
//     `signed < 0` sitting exactly on a branch boundary, with ~90 constraints and ~330
//     contacts per step within 1e-7 of it. So the tight comparison is only available
//     here, and it is worth having before a looser one.
//
// What this is a port of
// ----------------------
// `vulkan-gnn-poc/real_scene/xpbd.py::project` with `sweep="jacobi"`: `_apply_jacobi`,
// then `_apply_area`, then `_resolve_guide`, then the contacts, that order, every
// iteration. The two-pass Jacobi rather than `_apply_fused` on purpose -- the fused
// per-(vertex, slot) layout exists only to collapse an iteration into one GPU dispatch,
// and `test_fused_sweep_matches_jacobi` pins it to this form, so this is the shape the
// other one was validated against.
//
// Units are **metres**, not the centimetres the rest of this PoC carries, so the
// compliance, contact-offset and area-floor constants tuned in the sibling PoC transfer
// unchanged. `positions_from_cm` / `positions_to_cm` are the only conversion points.
//
// Bending is a distance constraint too, between the opposite corners of two triangles
// sharing an edge, so it needs no topology beyond the triangle list. `derive_bend_pairs`
// produces that list deterministically, and the calibrator has to derive the same one:
// target lengths are calibrated over a trajectory, so the pair ordering is a contract
// between the two, and the constraint array the solver receives is stretch pairs followed
// by bend pairs with `bendStart` marking the boundary.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlcloth {

struct XpbdConfig {
    int   iterations{ 8 };
    float timestep{ 1.0f / 30.0f };   // seconds
    float relaxation{ 1.0f };
    bool  oneSided{ true };           // resist stretch only; a short edge is legal

    float stretchCompliance{ 0.0f };  // m^2/N, constraints below `bendStart`
    float bendCompliance{ 0.0f };     // m^2/N, constraints at or above `bendStart`

    // Area floor as a fraction of the calibrated target area. <= 0 disables the pass.
    float areaFloor{ 0.0f };
    float areaCompliance{ 0.0f };

    // Soft pull towards the network prediction. Negative disables it, which is the
    // "standard" mode where the prediction is only the initial state. Zero snaps onto
    // the prediction; larger leaves the vertex where the constraints put it.
    float guideCompliance{ -1.0f };
    // Per-vertex trust ramp against the vertex's own shortest edge. <= 0 means trust 1
    // everywhere.
    float guideTrustRatio{ 0.0f };

    bool  collision{ false };
    float contactOffset{ 0.0f };      // metres
};

// A capsule already placed in the solver's frame: the line segment is
// centre +/- axis * halfLength, and the surface is `radius` from it.
struct Capsule {
    float centre[3]{};
    float axis[3]{};
    float radius{};
    float halfLength{};
};

// Signed distance from a point to a capsule: negative inside, and monotone from the
// surface to the axis. Unlike the sibling PoC's nearest-proxy half-plane this does not
// turn positive again past the far surface, which is why a vertex driven through a limb
// still reads as a violation. `results/PROGRESS.md` section 6 records that criterion
// reporting 28.5% penetration at 0.12 m and 1% at 0.25 m.
float capsule_signed_distance(const float point[3], const Capsule& capsule);

// Opposite-corner pairs of every triangle couple sharing an edge, sorted and deduplicated
// so the ordering is a property of the mesh rather than of iteration order -- the solve is
// meant to be reproducible, and the calibrator indexes target lengths by this ordering.
// Appended to the stretch pairs to form the constraint list, with the boundary reported as
// `bendStart`. Returns 2 * pairs entries.
std::vector<uint32_t> derive_bend_pairs(uint32_t vertexCount, const uint32_t* triangles, uint32_t triangleCount);

// Everything that depends on the mesh and the configuration but not on the state. Built
// once per asset; `configure` recomputes only the parts a config change touches.
class XpbdSolver {
public:
    // `constraintPairs` is 2*constraintCount: stretch pairs first, then bend pairs, with
    // `bendStart` the index of the first bend pair (pass `constraintCount` for none).
    // `triangles` is 3*triangleCount, `vertexMassKg` and `pinMask` are per vertex. Target
    // lengths are in metres and are **calibrated**, not rest lengths: skinning an authored
    // rest mesh into frame 0 already puts a large fraction of edges well past rest, so
    // aiming at rest would contract them hard. See `plans/gnn/gnn-xpbd-v2.md` section 1.4.
    bool build(uint32_t vertexCount,
               const uint32_t* constraintPairs, uint32_t constraintCount, uint32_t bendStart,
               const uint32_t* triangles, uint32_t triangleCount,
               const float* vertexMassKg,
               const uint32_t* pinMask,
               const float* targetLengthM,
               const float* targetAreaM2,
               std::string& err);

    void configure(const XpbdConfig& config);

    // Per-vertex opt-in for contacts. Null means every vertex, which is the wrong default for a
    // multi-piece garment and is kept only so a single-piece one needs no mask: measured on
    // ChaosCloth's own run-clip solution, this garment's four pieces sit 69 / 98 / 14 / 100%
    // inside the capsules, because the body capsules are a ragdoll envelope wider than the skin
    // and a fitted bodice belongs inside it. Resolving contacts there does not fix penetration,
    // it lifts the garment off the chest.
    void set_collision_mask(const uint8_t* mask);
    const XpbdConfig& config() const { return config_; }

    // One projection step. `guideM` is the network's prediction for this frame and is
    // both the initial state and, in guide mode, the target the guide pass pulls towards.
    //
    // `inertialM` is the Verlet extrapolation of the previous two frames,
    // `2 * x_now - x_previous`, and is only read by the guide trust gate: confidence is
    // how far the network's prediction sits from where inertia alone would have put the
    // vertex, measured against that vertex's own shortest edge. Pass null to disable the
    // gate (confidence 1 everywhere). It is a caller argument rather than solver state
    // because the runtime owns the frame history, and because a stateful solver would
    // make the sweep harness replay clips in order for no reason.
    //
    // Pinned vertices are overwritten with their guide value at the end, matching
    // `project`'s closing `torch.where(pinned, pin_target, current)`.
    //
    // With `iterations == 0` this returns `guideM` unchanged, byte for byte: no pass
    // runs, and the pin overwrite is a copy of a value already there. That is the
    // no-regression gate -- the pure-network output hash must be reproduced exactly.
    void step(const float* guideM, const float* inertialM,
              const std::vector<Capsule>& capsules, float* outM);

    uint32_t vertexCount() const { return vertexCount_; }
    uint32_t constraintCount() const { return constraintCount_; }
    uint32_t bendStart() const { return bendStart_; }
    uint32_t triangleCount() const { return triangleCount_; }

    // Diagnostics from the most recent `step`, for the metrics column.
    struct StepStats {
        uint32_t contactsResolved{};
        uint32_t areaFloorsActive{};
        uint32_t stretchActive{};
        float    maxCorrectionM{};
    };
    const StepStats& lastStep() const { return stats_; }

private:
    void apply_stretch(const float* current, float* next);
    void apply_area(const float* current, float* next);
    void apply_guide(const float* guide, float* position);
    void apply_contacts(const std::vector<Capsule>& capsules, float* position);
    uint32_t vertexCount_{};
    uint32_t constraintCount_{};
    uint32_t bendStart_{};
    uint32_t triangleCount_{};
    XpbdConfig config_{};
    StepStats stats_{};

    std::vector<uint32_t> pairs_;          // 2 * constraintCount
    std::vector<uint32_t> triangles_;      // 3 * triangleCount
    std::vector<float>    targetLength_;   // constraintCount, metres
    std::vector<float>    targetArea_;     // triangleCount, m^2
    std::vector<float>    inverseMass_;    // vertexCount
    std::vector<uint8_t>  pinned_;         // vertexCount
    std::vector<uint8_t>  collisionMask_;  // vertexCount, empty means all
    std::vector<float>    minEdge_;        // vertexCount, metres -- the guide trust scale

    // Baked per-constraint scalars. `weightSum` is stored rather than recomputed for the
    // same reason `SolverTables` gives: it removes any question of w_a + w_b against
    // w_b + w_a, and it keeps this port's arithmetic identical to the reference's.
    std::vector<float>    weightSum_;
    std::vector<float>    alpha_;
    std::vector<float>    denominator_;
    std::vector<uint8_t>  alive_;
    std::vector<float>    averaging_;      // constraints incident to each vertex
    std::vector<float>    areaIncident_;   // triangles incident to each vertex

    // Per-step scratch.
    std::vector<float>    multiplier_;
    std::vector<float>    areaMultiplier_;
    std::vector<float>    guideMultiplier_;
    std::vector<float>    confidence_;     // vertexCount, computed once per step
    std::vector<float>    accumulator_;    // 3 * vertexCount
    std::vector<float>    scratch_;        // 3 * vertexCount
};

}  // namespace mlcloth
