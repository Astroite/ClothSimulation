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

} // namespace mlcloth
