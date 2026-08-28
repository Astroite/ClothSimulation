#pragma once
// MLCloth binary/model format layer.
// C++17, dependency-free, Windows-compatible.

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <array>

namespace mlcloth {

// ---------------------------------------------------------------------------
// SHA-256 (self-contained, little-endian friendly)
// ---------------------------------------------------------------------------

struct Sha256Digest {
    std::array<uint8_t, 32> bytes{};
};

inline bool operator==(const Sha256Digest& a, const Sha256Digest& b) noexcept { return a.bytes == b.bytes; }
inline bool operator!=(const Sha256Digest& a, const Sha256Digest& b) noexcept { return !(a == b); }

// Hash arbitrary byte range.
Sha256Digest sha256(const uint8_t* data, size_t len);

// Convenience: hash a std::vector<uint8_t>.
Sha256Digest sha256(const std::vector<uint8_t>& v);

// Lowercase hex string (64 chars).
std::string sha256_hex(const Sha256Digest& d);

// Driver-name list hash: UTF-8 names joined with single '\n', no trailing newline.
Sha256Digest sha256_driver_names(const std::vector<std::string>& names);

// ---------------------------------------------------------------------------
// Model format (.mlclothmodel)
// ---------------------------------------------------------------------------
// Layout: [uint32 LE json_len][json_len bytes UTF-8 JSON][opaque payload]
//
// Required JSON fields (validated strictly):
//   modelType        (int)    == 2
//   driverFeatureLen (int)    == 1969
//   drivenFeatureLen (int)    == 16394
//   pcaDim           (int)    == 512
//   driverNames      (array of exactly 45 strings, first == "Root_M")
//   vertexCount      (optional int) == 5294 when present
// Vertex count is always derived as (drivenFeatureLen - pcaDim) / 3 and must
// equal 5294; the production model does not declare vertexCount.
//
// The original bytes must be kept alive for the AILab API.

struct ModelInfo {
    int modelType{};
    int driverFeatureLen{};
    int drivenFeatureLen{};
    int pcaDim{};
    int vertices{};
    std::vector<std::string> driverNames;

    // Byte ranges inside the original buffer (for zero-copy API use).
    size_t jsonOffset{};   // offset of JSON start (after 4-byte length)
    size_t jsonLen{};      // length of JSON in bytes
    size_t payloadOffset{};// offset of opaque payload start
    size_t payloadLen{};   // length of opaque payload in bytes
};

// Parse model from memory buffer.  Returns false on any validation failure.
// On success, out is populated; the buffer must stay alive.
bool parse_model(const uint8_t* data, size_t len, ModelInfo& out, std::string& err);

// ---------------------------------------------------------------------------
// Clip format (.mlclothclip) — MLDRV001
// ---------------------------------------------------------------------------
// Packed little-endian 144-byte header, followed by payload float32 arrays.
//
// Header fields (offsets from start):
//   [0]   magic[8]                = "MLDRV001"
//   [8]   version          u32    = 1
//   [12]  headerBytes      u32    = 144
//   [16]  frameCount       u32
//   [20]  fpsNumerator     u32    = 30
//   [24]  fpsDenominator   u32    = 1
//   [28]  driverCount      u32    = 45
//   [32]  rootDriverIndex  u32    = 0
//   [36]  localFloatCount  u32    = frameCount * 45 * 6
//   [40]  componentFloatCount u32 = frameCount * 45 * 6
//   [44]  positionFloatCount  u32 = frameCount * 45 * 3
//   [48]  modelSha256[32]
//   [80]  driverListSha256[32]
//   [112] payloadSha256[32]
//   [144] — end of header
//
// Payload (contiguous float32, little-endian):
//   local_fu        : frameCount * 45 * 6 floats
//   component_fu    : frameCount * 45 * 6 floats
//   component_pos_cm: frameCount * 45 * 3 floats

static constexpr size_t kClipHeaderBytes   = 144;
static constexpr size_t kMagicLen          = 8;
static constexpr uint32_t kClipVersion     = 1;
static constexpr uint32_t kDriverCount     = 45;
static constexpr uint32_t kRootDriverIndex = 0;
static constexpr uint32_t kFpsNum          = 30;
static constexpr uint32_t kFpsDen          = 1;

struct ClipHeader {
    char     magic[kMagicLen]{};      // "MLDRV001"
    uint32_t version{};
    uint32_t headerBytes{};
    uint32_t frameCount{};
    uint32_t fpsNumerator{};
    uint32_t fpsDenominator{};
    uint32_t driverCount{};
    uint32_t rootDriverIndex{};
    uint32_t localFloatCount{};
    uint32_t componentFloatCount{};
    uint32_t positionFloatCount{};
    Sha256Digest modelSha256{};
    Sha256Digest driverListSha256{};
    Sha256Digest payloadSha256{};
};

struct ClipInfo {
    ClipHeader header{};

    // Pointers into the original buffer (must stay alive).
    const float* localFu{};         // frameCount * 45 * 6
    const float* componentFu{};     // frameCount * 45 * 6
    const float* componentPosCm{};  // frameCount * 45 * 3
};

// Parse clip from memory buffer.  expectedModelHash and expectedDriverListHash
// are the SHA-256 digests that must appear in the header.  Returns false on any
// validation failure; on success, out is populated.
bool parse_clip(const uint8_t* data, size_t len,
                const Sha256Digest& expectedModelHash,
                const Sha256Digest& expectedDriverListHash,
                ClipInfo& out, std::string& err);

// ---------------------------------------------------------------------------
// Cloth mesh format (.mlmesh) — MLMSH001
// ---------------------------------------------------------------------------
// The topology the model's 5,294 outputs actually belong to.  It is the LOD0 sim
// mesh of the ChaosClothAsset the shipping plugin reads
// (FChaosClothSimulationModel::GetIndices / GetPositions, see
// MLClothSimulationProxy.cpp::CacheSimModel), so `positions_cm` is in the same
// Root_M-local centimetre space the inference output is already in — the model
// output and this rest pose need no registration between them.
//
// Container is the sectioned layout shared with the sibling GNN PoC's VCLTH001
// family, so `real_scene/formats.py::write_sectioned` writes it unmodified:
//
//   [0]  magic[8]          = "MLMSH001"
//   [8]  version      u32  = 1
//   [12] sectionCount u32
//   [16] fileBytes    u64  = exact file length
//   [24] payloadOffset u64 = align16(96 + 32 * sectionCount)
//   [32] payloadSha256[32] = SHA-256 of everything from payloadOffset on
//   [64] sourceSha256[32]
//   [96] section directory: sectionCount * { name[16], offset u64, count u32, stride u32 }
//
// `sourceSha256` carries the *model* digest rather than the source asset's.  The
// asset a mesh was exported from is provenance and lives in the JSON sidecar;
// what has to be mechanically enforced is that a mesh is never paired with a
// different `.enc`, because a topology from another garment would parse cleanly
// and then be silently wrong.  This is the same lock `.mldrv` already applies,
// so `parse_mesh` takes an expected model hash exactly like `parse_clip` does.
//
// Required sections (strides in bytes):
//   info           8 x 4   u32: vertices, triangles, edges, boundaryEdges,
//                               boundaryLoops, maxTriangleValence, pinned, reserved
//   positions      V x 12  float3, Root_M-local cm
//   triangles      T x 12  uint3, stored winding order (load-bearing: see
//                          real_scene/xpbd.py::triangle_areas)
//   edges          E x 8   uint2, undirected, a < b, strictly ascending
//   edge_csr_offs  V+1 x 4 u32, directed neighbour CSR
//   edge_csr_nbr   2E x 4  u32
//   tri_csr_offs   V+1 x 4 u32, incident-triangle CSR (for the normal pass)
//   tri_csr_idx    3T x 4  u32
//   vertex_mass    V x 4   float, kg, area-weighted
//   pin_mask       V x 4   u32, 0 or 1

static constexpr size_t kSectionHeaderBytes = 96;
static constexpr size_t kSectionEntryBytes  = 32;
static constexpr size_t kSectionNameBytes   = 16;
static constexpr size_t kSectionAlignment   = 16;
static constexpr uint32_t kMeshVersion      = 1;

// The model's vertex count, derived from (drivenFeatureLen - pcaDim) / 3 and
// required by both parsers so a mesh and a model can never disagree.
static constexpr uint32_t kClothVertexCount = 5294;

struct MeshInfo {
    uint32_t vertices{};
    uint32_t triangles{};
    uint32_t edges{};              // undirected
    uint32_t boundaryEdges{};      // used by exactly one triangle
    uint32_t boundaryLoops{};
    uint32_t maxTriangleValence{};
    uint32_t pinnedVertices{};
    Sha256Digest modelSha256{};    // from sourceSha256
    Sha256Digest payloadSha256{};

    // Pointers into the original buffer (must stay alive).
    const float*    positionsCm{};          // vertices * 3
    const uint32_t* triangleIndices{};      // triangles * 3
    const uint32_t* edgePairs{};            // edges * 2
    const uint32_t* edgeCsrOffsets{};       // vertices + 1
    const uint32_t* edgeCsrNeighbours{};    // edges * 2
    const uint32_t* triangleCsrOffsets{};   // vertices + 1
    const uint32_t* triangleCsrIndices{};   // triangles * 3
    const float*    vertexMassKg{};         // vertices
    const uint32_t* pinMask{};              // vertices
};

// Parse a cloth mesh from a memory buffer.  `expectedModelHash` must equal the
// container's sourceSha256.  Returns false on any validation failure.
bool parse_mesh(const uint8_t* data, size_t len,
                const Sha256Digest& expectedModelHash,
                MeshInfo& out, std::string& err);

// ---------------------------------------------------------------------------
// Capsule collision format (.mlcap) — MLCAP001
// ---------------------------------------------------------------------------
// The body capsules of the physics asset the character references at runtime, in
// bone-local centimetres, each tagged with the index of the driver bone it rides.
// Every bone is one of the model's 45 drivers, so the `.mldrv` clip already carries
// their per-frame transforms: collision needs no new per-frame data, no vertex proxy
// and no runtime skinning.
//
// Sections: `info` 4x4 = [count, driverCount, 0, 0], `driver` N x 4,
//           `center` N x 12, `axis` N x 12, `size` N x 8 (radius, halfLength).
//
// `axis` is the capsule's local Z after Unreal's rotator, already unit length.
// `size.halfLength` is half of `FKSphylElem::Length`, which is the *line segment*
// length -- the total height of the capsule is Length + 2 * Radius.

static constexpr uint32_t kCapsuleVersion = 1;

struct CapsuleInfo {
    uint32_t count{};
    uint32_t driverCount{};
    Sha256Digest modelSha256{};
    Sha256Digest payloadSha256{};
    const uint32_t* driverIndices{};   // count
    const float*    centresCm{};       // count * 3, bone-local
    const float*    axes{};            // count * 3, bone-local, unit
    const float*    sizesCm{};         // count * 2, radius then halfLength
};

bool parse_capsules(const uint8_t* data, size_t len,
                    const Sha256Digest& expectedModelHash,
                    CapsuleInfo& out, std::string& err);

// ---------------------------------------------------------------------------
// Writers (for tests / fixture generation)
// ---------------------------------------------------------------------------

// Build a valid model binary.
std::vector<uint8_t> write_model(int modelType, int driverFeatureLen,
                                  int drivenFeatureLen, int pcaDim,
                                  int vertices,
                                  const std::vector<std::string>& driverNames,
                                  const std::vector<uint8_t>& payload);

// Build a valid clip binary.
// localFu, componentFu, componentPosCm are raw float arrays (host endianness).
// modelHash and driverListHash are pre-computed SHA-256 digests.
std::vector<uint8_t> write_clip(uint32_t frameCount,
                                 const Sha256Digest& modelHash,
                                 const Sha256Digest& driverListHash,
                                 const float* localFu,
                                 const float* componentFu,
                                 const float* componentPosCm);

// Build a valid mesh binary.  Edges, both CSRs and the boundary/valence counts
// are derived here rather than supplied, so a valid fixture is one call; the
// negative tests corrupt the returned bytes afterwards, the way the clip tests
// do.  Deriving them twice is deliberate: `parse_mesh` checks that the edge list
// exactly covers the triangle edge set, so a real asset whose Python baker
// disagrees with this derivation fails to load rather than loading wrongly.
// Returns an empty vector and sets `err` when the inputs cannot form a mesh.
std::vector<uint8_t> write_mesh(const Sha256Digest& modelHash,
                                 const std::vector<float>& positionsCm,
                                 const std::vector<uint32_t>& triangleIndices,
                                 const std::vector<float>& vertexMassKg,
                                 const std::vector<uint32_t>& pinMask,
                                 std::string& err);

} // namespace mlcloth
