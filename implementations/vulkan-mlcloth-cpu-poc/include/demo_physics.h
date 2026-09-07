#pragma once
// CPU dynamic cloth solver for the Demo rebuild.
// C++17, dependency-free, XPBD-based.

#include <cmath>
#include <cstdint>
#include <vector>
#include <array>

namespace mlcloth::demo {

// ---------------------------------------------------------------------------
// Vec3 — dependency-free three-component vector
// ---------------------------------------------------------------------------

struct Vec3 {
    float x{}, y{}, z{};
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline Vec3 operator*(Vec3 v, float s) { return { v.x * s, v.y * s, v.z * s }; }
inline Vec3 operator*(float s, Vec3 v) { return v * s; }
inline Vec3 operator/(Vec3 v, float s) { return { v.x / s, v.y / s, v.z / s }; }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
inline Vec3& operator-=(Vec3& a, Vec3 b) { a = a - b; return a; }
inline Vec3& operator*=(Vec3& v, float s) { v = v * s; return v; }
inline Vec3 operator-(Vec3 v) { return { -v.x, -v.y, -v.z }; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
inline float length(Vec3 v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(Vec3 v) {
    const float len = length(v);
    return len > 1e-12f ? v / len : Vec3{};
}

// Component-wise division for Vec3 (used in vector XPBD lambda)
inline Vec3 operator/(Vec3 a, Vec3 b) { return { a.x / b.x, a.y / b.y, a.z / b.z }; }

// ---------------------------------------------------------------------------
// Mesh — input cloth geometry
// ---------------------------------------------------------------------------

struct Mesh {
    std::vector<Vec3> rest;           // rest positions, world metres
    std::vector<uint32_t> triangles;  // triangle indices (3 per triangle)
    std::vector<float> mass;          // per-vertex mass (kg)
    std::vector<uint8_t> pinned;      // per-vertex pin flag (0 or 1)
};

// ---------------------------------------------------------------------------
// PhysicsConfig — solver parameters
// ---------------------------------------------------------------------------

struct PhysicsConfig {
    int iterations{ 2 };
    float stretchCompliance{ 0.0f };
    float shearCompliance{ 0.0f };
    float bendCompliance{ 0.0f };
    float guideCompliance{ -1.0f };   // negative disables guide
    float dampingPerSecond{ 0.99f };
    float friction{ 0.1f };
    float thickness{ 0.003f };        // collision thickness, metres
    float gravity{ -9.81f };          // m/s^2, applied to Y
    bool enableCollision{ false };
    bool enableSelfCollision{ false };
    bool enableTethers{ false };
    bool contactAwareGuide{ true };  // attenuate coarse ML attraction into nearby colliders
    float tetherScale{ 1.05f };       // upper bound / rest edge-path length; >= 1
    bool gnnAcceleration{ false };   // GPU held acceleration replaces analytic gravity
};

// ---------------------------------------------------------------------------
// TriangleCollider — moving body triangles for collision
// ---------------------------------------------------------------------------

struct MovingCapsule {
    Vec3 previousA{},previousB{},currentA{},currentB{};
    float radius{};
    // Stable material-frame directions, supplied by the animation adapter.
    // Physics uses endpoints/radius only; surface sampling also needs axial twist.
    Vec3 surfaceU{},surfaceV{};
    bool hasSurfaceFrame{};
};
struct TriangleCollider {
    std::vector<Vec3> previous;       // previous frame positions
    std::vector<Vec3> current;        // current frame positions
    std::vector<uint32_t> triangles;  // triangle indices (3 per triangle)
    std::vector<MovingCapsule> capsules;
};
// Static incidence for angle-weighted vertex and edge pseudonormals.
struct BodySurfaceTopology {
    uint32_t vertices{};
    std::vector<uint32_t> triangles,offsets,incident,edgeNeighbors;
    void build(uint32_t vertexCount,const std::vector<uint32_t>& indices);
    std::vector<uint32_t> packed()const;
};
std::vector<Vec3> bodySurfaceNormals(const TriangleCollider&,const BodySurfaceTopology&);
Vec3 bodyFeatureNormal(const BodySurfaceTopology&,const std::vector<Vec3>& normals,uint32_t triangle,Vec3 bary);

// ---------------------------------------------------------------------------
// PhysicsStats — diagnostics from the most recent step
// ---------------------------------------------------------------------------

struct PhysicsStats {
    uint32_t stretchActive{};
    uint32_t shearActive{};
    uint32_t bendActive{};
    uint32_t contactsResolved{};
    uint32_t selfContactsResolved{};
    float maxCorrection{};
};

struct SelfContactRecord {
    std::array<uint32_t,4> ids{};
    std::array<float,4> weights{};
    Vec3 delta{};
    float distance{};
};

// ---------------------------------------------------------------------------
// Constraint group tables (read-only, for GPU port)
// ---------------------------------------------------------------------------

struct ConstraintGroups {
    std::vector<uint32_t> groupStart;  // offset into constraintIndices per group
    std::vector<uint32_t> groupCount;  // count per group
    std::vector<uint32_t> constraintIndices;  // constraint indices sorted by group
    uint32_t groupCountN{};
};

// ---------------------------------------------------------------------------
// GPU-ready constraint records (public for porting)
// ---------------------------------------------------------------------------

// Shear constraint: triangle dot-product/strain metric.
// Vertices (a, b, c) form a triangle. restDot = dot(b_rest-a_rest, c_rest-a_rest).
// Gradient: dC/da = -(e1+e2), dC/db = e2, dC/dc = e1 where e1=b-a, e2=c-a.
struct ShearConstraint {
    uint32_t a, b, c;
    float restDot;
};

// Four nearest distinct connected pin regions; invalid slots have zero length.
struct Tether { uint32_t anchor{}; float length{}; };

// Bend constraint: four-point signed dihedral angle.
// Vertices (a, b) share the edge; (c, d) are opposite vertices in adjacent triangles.
// restAngle = signed dihedral angle at rest, in radians.
// Convention: angle = atan2(dot(cross(n1hat,n2hat), ehat), dot(n1hat,n2hat))
//   where n1 = cross(b-a, c-a), n2 = cross(b-a, d-a), ehat = normalize(b-a).
// Gradients: dC/dc = -n1hat/hc, dC/dd = +n2hat/hd,
//   dC/da = -(1-tc)*dC/dc - (1-td)*dC/dd, dC/db = -tc*dC/dc - td*dC/dd
//   where hc = |n1|/|e|, hd = |n2|/|e|, tc = dot(c-a,e)/|e|^2, td = dot(d-a,e)/|e|^2.
struct BendConstraint {
    uint32_t a, b, c, d;
    float restAngle;
};

// ---------------------------------------------------------------------------
// BVH types (for body collision)
// ---------------------------------------------------------------------------

struct AABB {
    Vec3 min, max;
};

struct BVHNode {
    AABB bounds;
    uint32_t left{ 0 }, right{ 0 };
    uint32_t triBegin{ 0 }, triEnd{ 0 };  // range into sorted triangle index array
    bool isLeaf{ false };
};

// ---------------------------------------------------------------------------
// PhysicsSolver
// ---------------------------------------------------------------------------

class PhysicsSolver {
public:
    PhysicsSolver() = default;

    // Build from mesh. Derives constraints, BVH, coarse patches.
    // Throws on invalid inputs.
    void build(const Mesh& mesh);

    // Reset to new positions (zeros velocities — snapshot restore uses the two-arg overload).
    void reset(const std::vector<Vec3>& positions);
    // Reset to new positions and velocities (snapshot restore).
    void reset(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities);

    // Advance one timestep.
    // dt: seconds, pinTargets: target positions for pinned vertices,
    // guide: optional coarse guide (nullptr = pure XPBD),
    // body: optional body collider (nullptr = no body collision).
    void step(float dt,
              const std::vector<Vec3>& pinTargets,
              const std::vector<Vec3>* guide,
              const TriangleCollider* body,
              const PhysicsConfig& config);

    // Accessors
    const std::vector<Vec3>& positions() const { return positions_; }
    const std::vector<Vec3>& velocities() const { return velocities_; }
    const PhysicsStats& stats() const { return stats_; }
    bool captureContacts{}; // Explicit validation only.
    std::vector<SelfContactRecord> contactRecords;

    // Read-only tables for GPU port (backward-compatible edge-pair form)
    const std::vector<uint32_t>& stretchPairs() const { return stretchPairs_; }
    const std::vector<uint32_t>& shearPairs() const { return shearPairs_; }
    const std::vector<uint32_t>& bendPairs() const { return bendPairs_; }
    const std::vector<float>& stretchRestLength() const { return stretchRestLen_; }
    const std::vector<float>& shearRestLength() const { return shearRestLen_; }
    const std::vector<float>& bendRestLength() const { return bendRestLen_; }
    const ConstraintGroups& stretchGroups() const { return stretchGroups_; }
    const ConstraintGroups& shearGroups() const { return shearGroups_; }
    const ConstraintGroups& bendGroups() const { return bendGroups_; }

    // GPU-ready constraint records
    const std::vector<ShearConstraint>& shearConstraints() const { return shearConstraints_; }
    const std::vector<BendConstraint>& bendConstraints() const { return bendConstraints_; }

    // Coarse patch info for guide projection
    const std::vector<uint32_t>& patchAssignment() const { return patchAssignment_; }
    const std::vector<uint32_t>& patchOffsets() const { return patchOffsets_; }
    const std::vector<uint32_t>& patchVertices() const { return patchVertices_; }
    uint32_t patchCount() const { return patchCount_; }
    const std::vector<uint32_t>& selfExclusionOffsets()const{return selfOffsets_;}
    const std::vector<uint32_t>& selfExclusionVertices()const{return selfVertices_;}
    const std::vector<float>& selfExclusionDistances()const{return selfDistances_;}
    const std::vector<std::array<Tether,4>>& tethers()const{return tethers_;}

private:
    void buildConstraints(const Mesh& mesh);
    void buildCoarsePatches(const Mesh& mesh);
    void colorConstraints();
    void buildSelfExclusions(const Mesh& mesh);
    void buildTethers(const Mesh& mesh);
    void applyTethers(const std::vector<Vec3>& pins,float scale);

    void applyStretch(std::vector<Vec3>& position, float compliance, float dt);
    void applyShear(std::vector<Vec3>& position, float compliance, float dt);
    void applyBend(std::vector<Vec3>& position, float compliance, float dt);
    void applyGuide(const std::vector<Vec3>& guide, std::vector<Vec3>& position, float compliance, float dt);
    void prepareBodyCollision(const TriangleCollider& body,float thickness);
    void prepareGuideContacts(const TriangleCollider* body,float thickness);
    void applyBodyCollision(std::vector<Vec3>& position, std::vector<Vec3>& velocity,
                            const TriangleCollider& body, float thickness, float friction, float dt);
    void applySelfCollision(std::vector<Vec3>& position, float thickness);
    void applyCapsuleCollision(std::vector<Vec3>& position,const std::vector<MovingCapsule>&,float thickness,float dt,
                               bool swept=true,const std::vector<uint8_t>* filter=nullptr);
    void recoverBodyContacts(std::vector<Vec3>& position,const TriangleCollider&,float thickness,float dt);

    uint32_t vertexCount_{};

    // Mesh topology
    std::vector<uint32_t> triangles_;
    std::vector<float> inverseMass_;
    std::vector<uint8_t> pinned_;
    std::vector<uint32_t> selfOffsets_,selfVertices_;
    std::vector<float> selfDistances_;
    std::vector<std::array<Tether,4>> tethers_;

    // Stretch constraints (edge pairs, bidirectional)
    std::vector<uint32_t> stretchPairs_;
    std::vector<float> stretchRestLen_;
    ConstraintGroups stretchGroups_;

    // Shear constraints (triangle dot-product/strain, GPU-ready)
    std::vector<ShearConstraint> shearConstraints_;
    std::vector<uint32_t> shearPairs_;       // backward-compatible flat pair form
    std::vector<float> shearRestLen_;        // backward-compatible (unused internally)
    ConstraintGroups shearGroups_;

    // Bend constraints (four-point signed dihedral angle, GPU-ready)
    std::vector<BendConstraint> bendConstraints_;
    std::vector<uint32_t> bendPairs_;        // backward-compatible flat pair form
    std::vector<float> bendRestLen_;         // backward-compatible (unused internally)
    ConstraintGroups bendGroups_;

    // State
    std::vector<Vec3> positions_;
    std::vector<Vec3> velocities_;
    std::vector<Vec3> previous_;
    PhysicsStats stats_;

    // Coarse patches for guide projection
    std::vector<uint32_t> patchAssignment_;  // vertex -> patch
    std::vector<uint32_t> patchOffsets_;     // CSR offsets, length = patchCount+1
    std::vector<uint32_t> patchVertices_;    // CSR vertex list
    uint32_t patchCount_{};

    // Scratch for constraint solving
    std::vector<Vec3> accumulator_;
    std::vector<float> stretchMultiplier_;
    std::vector<float> shearMultiplier_;
    std::vector<float> bendMultiplier_;
    std::vector<Vec3> guideLambda_;          // Vec3 per patch (vector XPBD lambda)
    struct GuideContact {Vec3 normal{};float proximity{};};
    std::vector<std::array<GuideContact,3>> guideContacts_; // floor, capsule, STM; derived each substep
    float guideContactBand_{.012f};

    // Body collision BVH (rebuilt/refit per step)
    std::vector<BVHNode> bodyBvhNodes_;
    std::vector<uint32_t> bodyBvhTriIndices_;
    uint32_t bodyBvhRoot_{ 0xFFFFFFFFu };
    BodySurfaceTopology bodySurface_;
    std::vector<Vec3> bodyNormals_;

    // Body contact record for friction pass
    struct BodyContact {
        uint32_t vertex;
        Vec3 normal;
        Vec3 contactPoint;
        uint32_t bodyTri;
        float baryU, baryV;
        float normalImpulse{};
        bool recovery{};
        Vec3 bodyVelocity{};
    };
    std::vector<BodyContact> bodyContacts_;
};

}  // namespace mlcloth::demo
