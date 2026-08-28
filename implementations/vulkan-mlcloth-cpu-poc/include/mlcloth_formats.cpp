#include "mlcloth_formats.h"

#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace mlcloth {

// ===========================================================================
// SHA-256
// ===========================================================================

namespace {

// Initial hash values (first 32 bits of the fractional parts of the square
// roots of the first 8 primes).
constexpr uint32_t kInit[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

// Round constants (first 32 bits of the fractional parts of the cube roots of
// the first 64 primes).
constexpr uint32_t kRound[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static inline uint32_t rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static inline uint32_t bsig0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
static inline uint32_t bsig1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
static inline uint32_t ssig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
static inline uint32_t ssig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

// Read a big-endian uint32 from a byte pointer.
static inline uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) <<  8) |  uint32_t(p[3]);
}

// Write a big-endian uint32 to a byte pointer.
static inline void put_be32(uint8_t* p, uint32_t v) {
    p[0] = uint8_t(v >> 24);
    p[1] = uint8_t(v >> 16);
    p[2] = uint8_t(v >>  8);
    p[3] = uint8_t(v      );
}

// Process one 64-byte block.
void sha256_block(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = be32(block + 4 * i);
    for (int i = 16; i < 64; ++i) w[i] = ssig1(w[i-2]) + w[i-7] + ssig0(w[i-15]) + w[i-16];

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + bsig1(e) + ch(e, f, g) + kRound[i] + w[i];
        uint32_t t2 = bsig0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

} // anonymous namespace

Sha256Digest sha256(const uint8_t* data, size_t len) {
    if (len != 0 && data == nullptr) return {};
    uint32_t state[8];
    std::memcpy(state, kInit, sizeof(kInit));

    // Process full 64-byte blocks.
    size_t off = 0;
    while (off + 64 <= len) {
        sha256_block(state, data + off);
        off += 64;
    }

    // Pad: append 1-bit, zeros, then 64-bit big-endian bit length.
    uint8_t buf[128]; // at most 2 blocks needed
    size_t tail = len - off;
    if (tail != 0) std::memcpy(buf, data + off, tail);
    buf[tail] = 0x80;

    size_t padLen = (tail < 56) ? 64 : 128;
    std::memset(buf + tail + 1, 0, padLen - tail - 1);

    uint64_t bits = len * 8;
    buf[padLen - 8] = uint8_t(bits >> 56);
    buf[padLen - 7] = uint8_t(bits >> 48);
    buf[padLen - 6] = uint8_t(bits >> 40);
    buf[padLen - 5] = uint8_t(bits >> 32);
    buf[padLen - 4] = uint8_t(bits >> 24);
    buf[padLen - 3] = uint8_t(bits >> 16);
    buf[padLen - 2] = uint8_t(bits >>  8);
    buf[padLen - 1] = uint8_t(bits      );

    for (size_t i = 0; i < padLen; i += 64) sha256_block(state, buf + i);

    Sha256Digest d;
    for (int i = 0; i < 8; ++i) put_be32(d.bytes.data() + 4 * i, state[i]);
    return d;
}

Sha256Digest sha256(const std::vector<uint8_t>& v) {
    return sha256(v.data(), v.size());
}

std::string sha256_hex(const Sha256Digest& d) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string s;
    s.resize(64);
    for (size_t i = 0; i < 32; ++i) {
        s[2*i]   = hex[d.bytes[i] >> 4];
        s[2*i+1] = hex[d.bytes[i] & 0x0f];
    }
    return s;
}

Sha256Digest sha256_driver_names(const std::vector<std::string>& names) {
    // Join with single '\n', no trailing newline.
    size_t total = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        total += names[i].size();
        if (i + 1 < names.size()) total += 1; // '\n'
    }
    std::vector<uint8_t> buf(total);
    size_t pos = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        std::memcpy(buf.data() + pos, names[i].data(), names[i].size());
        pos += names[i].size();
        if (i + 1 < names.size()) {
            buf[pos] = '\n';
            ++pos;
        }
    }
    return sha256(buf);
}

// ===========================================================================
// Minimal JSON field extraction (no external library)
// ===========================================================================

namespace {

// Skip whitespace starting at *p; advance *p past whitespace.
void skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
}

// Expect a literal character; return false if not matched.
bool expect(const char*& p, const char* end, char c) {
    if (p >= end || *p != c) return false;
    ++p;
    return true;
}

// Parse a JSON string (without surrounding quotes).  Returns the unescaped
// content (only handles \\ and \" minimally — sufficient for our field names).
bool parse_string(const char*& p, const char* end, std::string& out) {
    if (!expect(p, end, '"')) return false;
    out.clear();
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) {
            ++p;
            if (*p == '"' || *p == '\\') out += *p;
            else return false; // unsupported escape
        } else {
            out += *p;
        }
        ++p;
    }
    if (!expect(p, end, '"')) return false;
    return true;
}

// Parse a JSON integer (no exponent, no fraction).  Returns false on failure.
bool parse_int(const char*& p, const char* end, int& out) {
    skip_ws(p, end);
    bool neg = false;
    if (p < end && *p == '-') { neg = true; ++p; }
    if (p >= end || *p < '0' || *p > '9') return false;
    long long val = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        if (val > std::numeric_limits<int>::max()) return false;
        ++p;
    }
    out = neg ? -static_cast<int>(val) : static_cast<int>(val);
    return true;
}

// Find a top-level key in a JSON object.  Sets p to just after the colon on
// success.  Keys are compared literally (no escapes in key names we care about).
bool find_key(const char*& p, const char* end, const char* key) {
    // Start from beginning of object.
    skip_ws(p, end);
    if (!expect(p, end, '{')) return false;

    int depth = 1;
    while (p < end && depth > 0) {
        skip_ws(p, end);
        if (p >= end) return false;
        if (*p == '}') { --depth; if (depth == 0) { ++p; break; } ++p; continue; }
        if (*p == ',') { ++p; continue; }
        if (*p == '{') { ++depth; ++p; continue; }
        if (*p == '[') { ++depth; ++p; continue; }
        if (*p == ']' || *p == '}') { --depth; ++p; continue; }

        // Must be a string (key).
        if (*p != '"') {
            // Could be a value we need to skip.
            // Skip until next comma or closing brace at current depth.
            while (p < end && depth > 0) {
                if (*p == '"') { // skip string value
                    ++p;
                    while (p < end && *p != '"') { if (*p == '\\') ++p; ++p; }
                    if (p < end) ++p; // closing quote
                    break;
                }
                if (*p == '{' || *p == '[') ++depth;
                if (*p == '}' || *p == ']') { --depth; if (depth <= 1) break; }
                ++p;
            }
            continue;
        }

        // Parse key string.
        std::string keyStr;
        if (!parse_string(p, end, keyStr)) return false;
        skip_ws(p, end);
        if (!expect(p, end, ':')) return false;

        if (keyStr == key) return true; // p is now after the colon

        // Skip the value.
        skip_ws(p, end);
        if (p >= end) return false;
        if (*p == '"') { // string value
            std::string dummy;
            if (!parse_string(p, end, dummy)) return false;
        } else if (*p == '[') { // array value
            int ad = 1;
            ++p;
            while (p < end && ad > 0) {
                if (*p == '"') { std::string dummy; if (!parse_string(p, end, dummy)) return false; }
                else { if (*p == '[' || *p == '{') ++ad; if (*p == ']' || *p == '}') --ad; ++p; }
            }
        } else if (*p == '{') { // object value
            int od = 1;
            ++p;
            while (p < end && od > 0) {
                if (*p == '"') { std::string dummy; if (!parse_string(p, end, dummy)) return false; }
                else { if (*p == '{') ++od; if (*p == '}') --od; ++p; }
            }
        } else { // number / bool / null — skip until comma or close
            while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
        }
    }
    return false;
}

// Parse a top-level integer value (p is just after the key's colon).
bool parse_int_value(const char*& p, const char* end, int& out) {
    skip_ws(p, end);
    return parse_int(p, end, out);
}

// Parse a top-level array of strings (p is just after the key's colon).
bool parse_string_array(const char*& p, const char* end, std::vector<std::string>& out) {
    skip_ws(p, end);
    if (!expect(p, end, '[')) return false;
    out.clear();
    bool first = true;
    while (p < end) {
        skip_ws(p, end);
        if (*p == ']') { ++p; return true; }
        if (!first) {
            if (!expect(p, end, ',')) return false;
            skip_ws(p, end);
        }
        first = false;
        std::string s;
        if (!parse_string(p, end, s)) return false;
        out.push_back(std::move(s));
    }
    return false;
}

// Check that a key does NOT appear again at the top level of the JSON object.
// This catches duplicate keys.
bool check_no_duplicate(const char* jsonStart, const char* jsonEnd, const char* key) {
    int objectDepth = 0;
    int arrayDepth = 0;
    int occurrences = 0;
    const char* p = jsonStart;
    while (p < jsonEnd) {
        if (*p == '"') {
            const bool possibleTopLevelKey = objectDepth == 1 && arrayDepth == 0;
            std::string value;
            if (!parse_string(p, jsonEnd, value)) return false;
            const char* after = p;
            skip_ws(after, jsonEnd);
            if (possibleTopLevelKey && after < jsonEnd && *after == ':' && value == key) ++occurrences;
            if (occurrences > 1) return false;
            continue;
        }
        if (*p == '{') ++objectDepth;
        else if (*p == '}') --objectDepth;
        else if (*p == '[') ++arrayDepth;
        else if (*p == ']') --arrayDepth;
        ++p;
    }
    return occurrences <= 1;
}

} // anonymous namespace

// ===========================================================================
// Model parser
// ===========================================================================

bool parse_model(const uint8_t* data, size_t len, ModelInfo& out, std::string& err) {
    // Need at least 4 bytes for the JSON length prefix.
    if (len < 4) { err = "model: truncated (need 4 bytes for json_len)"; return false; }

    const uint32_t jsonLen = uint32_t(data[0]) | (uint32_t(data[1]) << 8) |
        (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);

    if (jsonLen > len - 4) { err = "model: json_len exceeds buffer"; return false; }
    if (jsonLen == 0) { err = "model: empty JSON"; return false; }

    const char* json = reinterpret_cast<const char*>(data + 4);
    const char* jsonEnd = json + jsonLen;

    // Validate UTF-8 minimally: no null bytes in JSON section.
    for (size_t i = 0; i < jsonLen; ++i) {
        if (json[i] == '\0') { err = "model: null byte in JSON"; return false; }
    }

    // Extract required integer fields.
    auto extract_int = [&](const char* name, int& dest, int expected) -> bool {
        const char* p = json;
        if (!find_key(p, jsonEnd, name)) { err = std::string("model: missing key \"") + name + "\""; return false; }
        if (!parse_int_value(p, jsonEnd, dest)) { err = std::string("model: bad int for \"") + name + "\""; return false; }
        if (dest != expected) { err = std::string("model: ") + name + " expected " + std::to_string(expected) + " got " + std::to_string(dest); return false; }
        if (!check_no_duplicate(json, jsonEnd, name)) { err = std::string("model: duplicate key \"") + name + "\""; return false; }
        return true;
    };

    if (!extract_int("modelType",        out.modelType,        2))    return false;
    if (!extract_int("driverFeatureLen", out.driverFeatureLen, 1969)) return false;
    if (!extract_int("drivenFeatureLen", out.drivenFeatureLen, 16394))return false;
    if (!extract_int("pcaDim",           out.pcaDim,           512))  return false;
    if (out.drivenFeatureLen <= out.pcaDim || (out.drivenFeatureLen - out.pcaDim) % 3 != 0) {
        err = "model: drivenFeatureLen is not V*3+pcaDim";
        return false;
    }
    out.vertices = (out.drivenFeatureLen - out.pcaDim) / 3;
    if (out.vertices != static_cast<int>(kClothVertexCount)) {
        err = "model: derived vertex count != " + std::to_string(kClothVertexCount);
        return false;
    }
    {
        const char* p = json;
        if (find_key(p, jsonEnd, "vertexCount")) {
            int declared = 0;
            if (!parse_int_value(p, jsonEnd, declared) || declared != out.vertices) {
                err = "model: vertexCount does not match derived vertex count";
                return false;
            }
            if (!check_no_duplicate(json, jsonEnd, "vertexCount")) { err = "model: duplicate key \"vertexCount\""; return false; }
        }
    }

    // Extract driverNames array.
    {
        const char* p = json;
        if (!find_key(p, jsonEnd, "driverNames")) { err = "model: missing key \"driverNames\""; return false; }
        if (!parse_string_array(p, jsonEnd, out.driverNames)) { err = "model: bad driverNames array"; return false; }
        if (out.driverNames.size() != 45) {
            err = "model: driverNames expected 45, got " + std::to_string(out.driverNames.size());
            return false;
        }
        if (out.driverNames[0] != "Root_M") { err = "model: first driver must be \"Root_M\""; return false; }
        const std::unordered_set<std::string> unique(out.driverNames.begin(), out.driverNames.end());
        if (unique.size() != out.driverNames.size()) { err = "model: driverNames must be unique"; return false; }
        if (!check_no_duplicate(json, jsonEnd, "driverNames")) { err = "model: duplicate key \"driverNames\""; return false; }
    }

    // Record byte ranges.
    out.jsonOffset  = 4;
    out.jsonLen     = jsonLen;
    out.payloadOffset = 4 + jsonLen;
    out.payloadLen  = len - (4 + jsonLen);
    if (out.payloadLen == 0) { err = "model: empty opaque payload"; return false; }

    return true;
}

// ===========================================================================
// Clip parser
// ===========================================================================

namespace {

// Read a little-endian uint32 from a byte pointer.
static inline uint32_t le32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

// Write a little-endian uint32 to a byte pointer.
static inline void put_le32(uint8_t* p, uint32_t v) {
    p[0] = uint8_t(v);  p[1] = uint8_t(v >> 8);
    p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24);
}

// Check that all floats in a range are finite.
bool all_finite(const float* f, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(f[i])) return false;
    }
    return true;
}

} // anonymous namespace

bool parse_clip(const uint8_t* data, size_t len,
                const Sha256Digest& expectedModelHash,
                const Sha256Digest& expectedDriverListHash,
                ClipInfo& out, std::string& err) {
    if (len < kClipHeaderBytes) { err = "clip: truncated header"; return false; }

    ClipHeader& h = out.header;

    // Read magic.
    std::memcpy(h.magic, data, kMagicLen);
    if (std::memcmp(h.magic, "MLDRV001", kMagicLen) != 0) { err = "clip: bad magic"; return false; }

    // Read header fields (little-endian).
    h.version            = le32(data + 8);
    h.headerBytes        = le32(data + 12);
    h.frameCount         = le32(data + 16);
    h.fpsNumerator       = le32(data + 20);
    h.fpsDenominator     = le32(data + 24);
    h.driverCount        = le32(data + 28);
    h.rootDriverIndex    = le32(data + 32);
    h.localFloatCount    = le32(data + 36);
    h.componentFloatCount= le32(data + 40);
    h.positionFloatCount = le32(data + 44);

    // Copy SHA-256 digests.
    std::memcpy(h.modelSha256.bytes.data(),      data + 48,  32);
    std::memcpy(h.driverListSha256.bytes.data(),  data + 80,  32);
    std::memcpy(h.payloadSha256.bytes.data(),     data + 112, 32);

    // Validate header fields.
    if (h.version != kClipVersion) { err = "clip: version != 1"; return false; }
    if (h.headerBytes != kClipHeaderBytes) { err = "clip: headerBytes != 144"; return false; }
    if (h.fpsNumerator != kFpsNum || h.fpsDenominator != kFpsDen) { err = "clip: fps != 30/1"; return false; }
    if (h.driverCount != kDriverCount) { err = "clip: driverCount != 45"; return false; }
    if (h.rootDriverIndex != kRootDriverIndex) { err = "clip: rootDriverIndex != 0"; return false; }

    // Validate expected hashes.
    if (h.modelSha256 != expectedModelHash) { err = "clip: model hash mismatch"; return false; }
    if (h.driverListSha256 != expectedDriverListHash) { err = "clip: driver list hash mismatch"; return false; }

    // Validate derived float counts.
    if (h.frameCount == 0) { err = "clip: frameCount is zero"; return false; }
    const uint64_t expectedLocal64 = uint64_t(h.frameCount) * kDriverCount * 6;
    const uint64_t expectedComp64 = expectedLocal64;
    const uint64_t expectedPos64 = uint64_t(h.frameCount) * kDriverCount * 3;
    if (expectedLocal64 > std::numeric_limits<uint32_t>::max() || expectedPos64 > std::numeric_limits<uint32_t>::max()) {
        err = "clip: float counts overflow uint32";
        return false;
    }
    const uint32_t expectedLocal = static_cast<uint32_t>(expectedLocal64);
    const uint32_t expectedComp = static_cast<uint32_t>(expectedComp64);
    const uint32_t expectedPos = static_cast<uint32_t>(expectedPos64);

    if (h.localFloatCount != expectedLocal) { err = "clip: localFloatCount mismatch"; return false; }
    if (h.componentFloatCount != expectedComp) { err = "clip: componentFloatCount mismatch"; return false; }
    if (h.positionFloatCount != expectedPos) { err = "clip: positionFloatCount mismatch"; return false; }

    // Validate total file length.
    const uint64_t payloadBytes64 = (expectedLocal64 + expectedComp64 + expectedPos64) * sizeof(float);
    if (payloadBytes64 > std::numeric_limits<size_t>::max() - kClipHeaderBytes) { err = "clip: payload byte size overflow"; return false; }
    const size_t payloadBytes = static_cast<size_t>(payloadBytes64);
    const size_t expectedLen  = kClipHeaderBytes + payloadBytes;
    if (len != expectedLen) { err = "clip: file length mismatch"; return false; }

    // Set payload pointers.
    const uint8_t* payload = data + kClipHeaderBytes;
    out.localFu        = reinterpret_cast<const float*>(payload);
    out.componentFu    = out.localFu + expectedLocal;
    out.componentPosCm = out.componentFu + expectedComp;

    // Validate all floats are finite.
    if (!all_finite(out.localFu, expectedLocal)) { err = "clip: nonfinite float in localFu"; return false; }
    if (!all_finite(out.componentFu, expectedComp)) { err = "clip: nonfinite float in componentFu"; return false; }
    if (!all_finite(out.componentPosCm, expectedPos)) { err = "clip: nonfinite float in componentPosCm"; return false; }

    // Verify payload SHA-256.
    Sha256Digest payloadHash = sha256(payload, payloadBytes);
    if (payloadHash != h.payloadSha256) { err = "clip: payload hash mismatch"; return false; }

    return true;
}

// ===========================================================================
// Mesh parser
// ===========================================================================

namespace {

static inline uint64_t le64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | uint64_t(p[i]);
    return v;
}

static inline void put_le64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i));
}

static inline size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct SectionSpan {
    std::string name;
    uint64_t offset{};
    uint32_t count{};
    uint32_t stride{};
};

// Everything the sectioned container guarantees, independent of what the sections mean.
// Extracted from `parse_mesh` rather than copied for `parse_capsules`: this is the layer that
// decides whether a file is trustworthy at all, and two copies of it would be two places for
// a bounds check to go missing. `label` only shapes the messages, so a caller still gets to
// say which format was rejected.
static bool parse_container(const uint8_t* data, size_t len,
                            const char* magic, const char* label, uint32_t expectedVersion,
                            const Sha256Digest& expectedSourceHash,
                            std::vector<SectionSpan>& spans,
                            Sha256Digest& payloadSha, Sha256Digest& sourceSha,
                            std::string& err) {
    const std::string tag = std::string(label) + ": ";
    if (len < kSectionHeaderBytes) { err = tag + "truncated header"; return false; }
    if (std::memcmp(data, magic, kMagicLen) != 0) { err = tag + "bad magic"; return false; }

    const uint32_t version = le32(data + 8);
    const uint32_t sectionCount = le32(data + 12);
    const uint64_t fileBytes = le64(data + 16);
    const uint64_t payloadOffset = le64(data + 24);
    std::memcpy(payloadSha.bytes.data(), data + 32, 32);
    std::memcpy(sourceSha.bytes.data(), data + 64, 32);

    // Names the required version rather than saying "unexpected": the caller supplies it, so a
    // generic message would drop the one piece of information the reader needs.
    if (version != expectedVersion) { err = tag + "version != " + std::to_string(expectedVersion); return false; }
    if (fileBytes != len) { err = tag + "fileBytes does not match the file length"; return false; }
    if (sectionCount == 0 || sectionCount > 64) { err = tag + "implausible section count"; return false; }
    const size_t directoryBytes = kSectionHeaderBytes + kSectionEntryBytes * size_t(sectionCount);
    if (directoryBytes > len) { err = tag + "section directory is truncated"; return false; }
    if (payloadOffset != align_up(directoryBytes, kSectionAlignment)) {
        err = tag + "payloadOffset does not follow the section directory";
        return false;
    }
    if (payloadOffset > len) { err = tag + "payloadOffset is past the end"; return false; }
    if (sourceSha != expectedSourceHash) { err = tag + "model hash mismatch"; return false; }
    if (sha256(data + payloadOffset, len - size_t(payloadOffset)) != payloadSha) {
        err = tag + "payload hash mismatch";
        return false;
    }

    spans.assign(sectionCount, SectionSpan{});
    for (uint32_t i = 0; i < sectionCount; ++i) {
        const uint8_t* entry = data + kSectionHeaderBytes + kSectionEntryBytes * size_t(i);
        size_t nameLen = 0;
        while (nameLen < kSectionNameBytes && entry[nameLen] != 0) ++nameLen;
        if (nameLen == 0 || nameLen == kSectionNameBytes) { err = tag + "section name is empty or unterminated"; return false; }
        for (size_t j = 0; j < nameLen; ++j) {
            const uint8_t c = entry[j];
            if (c < 0x21 || c > 0x7e) { err = tag + "section name is not printable ASCII"; return false; }
        }
        for (size_t j = nameLen; j < kSectionNameBytes; ++j) {
            if (entry[j] != 0) { err = tag + "section name has trailing bytes after its NUL"; return false; }
        }
        spans[i].name.assign(reinterpret_cast<const char*>(entry), nameLen);
        spans[i].offset = le64(entry + kSectionNameBytes);
        spans[i].count = le32(entry + kSectionNameBytes + 8);
        spans[i].stride = le32(entry + kSectionNameBytes + 12);
        if (spans[i].stride == 0 || spans[i].stride % 4 != 0) { err = tag + "section stride must be a non-zero multiple of 4"; return false; }
        if (spans[i].offset % kSectionAlignment != 0) { err = tag + "section is not 16-byte aligned"; return false; }
        if (spans[i].offset < payloadOffset || spans[i].offset > len) { err = tag + "section starts outside the payload"; return false; }
        const uint64_t bytes = uint64_t(spans[i].count) * spans[i].stride;
        if (bytes > uint64_t(len) - spans[i].offset) { err = tag + "section extends past the end"; return false; }
    }
    for (uint32_t i = 0; i < sectionCount; ++i) {
        for (uint32_t j = i + 1; j < sectionCount; ++j) {
            if (spans[i].name == spans[j].name) { err = tag + "duplicate section name " + spans[i].name; return false; }
            const uint64_t ai = spans[i].offset, bi = ai + uint64_t(spans[i].count) * spans[i].stride;
            const uint64_t aj = spans[j].offset, bj = aj + uint64_t(spans[j].count) * spans[j].stride;
            if (ai < bj && aj < bi) { err = tag + "sections " + spans[i].name + " and " + spans[j].name + " overlap"; return false; }
        }
    }
    return true;
}

// Locate one section and require its exact shape, so a wrong count or stride is a rejection
// rather than a reinterpretation.
static bool find_section(const uint8_t* data, const std::vector<SectionSpan>& spans,
                         const char* label, const char* name,
                         uint32_t count, uint32_t stride,
                         const uint8_t*& base, std::string& err) {
    for (const SectionSpan& span : spans) {
        if (span.name != name) continue;
        if (span.count != count) {
            err = std::string(label) + ": section " + name + " count is " + std::to_string(span.count)
                + ", expected " + std::to_string(count);
            return false;
        }
        if (span.stride != stride) {
            err = std::string(label) + ": section " + name + " stride is " + std::to_string(span.stride)
                + ", expected " + std::to_string(stride);
            return false;
        }
        base = data + span.offset;
        return true;
    }
    err = std::string(label) + ": missing required section " + name;
    return false;
}

// Locate a section that is allowed to be absent. Returns false only for a section that is
// present with the wrong shape -- absence is reported through `found`, so a caller can
// require the group to be all-or-nothing rather than treating a half-written pair as valid.
static bool find_optional_section(const uint8_t* data, const std::vector<SectionSpan>& spans,
                                  const char* label, const char* name,
                                  uint32_t count, uint32_t stride,
                                  const uint8_t*& base, bool& found, std::string& err) {
    found = false;
    for (const SectionSpan& span : spans) {
        if (span.name != name) continue;
        found = true;
        return find_section(data, spans, label, name, count, stride, base, err);
    }
    return true;
}

// Disjoint-set over vertices, used to count boundary loops as the connected
// components of the boundary-edge graph.
struct DisjointSet {
    std::vector<uint32_t> parent;
    explicit DisjointSet(uint32_t n) : parent(n) {
        for (uint32_t i = 0; i < n; ++i) parent[i] = i;
    }
    uint32_t find(uint32_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    void unite(uint32_t a, uint32_t b) {
        const uint32_t ra = find(a), rb = find(b);
        if (ra != rb) parent[ra] = rb;
    }
};

// Locate an undirected edge in the strictly-ascending `edges` array.
// Returns the edge index, or UINT32_MAX when absent.
uint32_t find_edge(const uint32_t* edges, uint32_t edgeCount, uint32_t a, uint32_t b) {
    if (a > b) { const uint32_t t = a; a = b; b = t; }
    uint32_t lo = 0, hi = edgeCount;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const uint32_t ea = edges[2 * mid], eb = edges[2 * mid + 1];
        if (ea < a || (ea == a && eb < b)) lo = mid + 1; else hi = mid;
    }
    if (lo < edgeCount && edges[2 * lo] == a && edges[2 * lo + 1] == b) return lo;
    return std::numeric_limits<uint32_t>::max();
}

} // anonymous namespace

bool parse_mesh(const uint8_t* data, size_t len,
                const Sha256Digest& expectedModelHash,
                MeshInfo& out, std::string& err) {
    std::vector<SectionSpan> spans;
    if (!parse_container(data, len, "MLMSH001", "mesh", kMeshVersion, expectedModelHash,
                         spans, out.payloadSha256, out.modelSha256, err)) {
        return false;
    }

    auto section = [&](const char* name, uint32_t count, uint32_t stride, const uint8_t*& base) -> bool {
        return find_section(data, spans, "mesh", name, count, stride, base, err);
    };

    const uint8_t* infoBytes = nullptr;
    if (!section("info", 8, 4, infoBytes)) return false;
    out.vertices           = le32(infoBytes + 0);
    out.triangles          = le32(infoBytes + 4);
    out.edges              = le32(infoBytes + 8);
    out.boundaryEdges      = le32(infoBytes + 12);
    out.boundaryLoops      = le32(infoBytes + 16);
    out.maxTriangleValence = le32(infoBytes + 20);
    out.pinnedVertices     = le32(infoBytes + 24);
    if (le32(infoBytes + 28) != 0) { err = "mesh: reserved info word is not zero"; return false; }

    const uint32_t V = out.vertices, T = out.triangles, E = out.edges;
    if (V != kClothVertexCount) {
        err = "mesh: vertex count is " + std::to_string(V) + ", expected " + std::to_string(kClothVertexCount);
        return false;
    }
    if (T == 0 || E == 0) { err = "mesh: mesh has no triangles or no edges"; return false; }
    constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
    if (T > kMax / 3 || E > kMax / 2) { err = "mesh: triangle or edge count overflows"; return false; }

    const uint8_t* positions = nullptr; const uint8_t* triangles = nullptr;
    const uint8_t* edges = nullptr; const uint8_t* edgeOffs = nullptr; const uint8_t* edgeNbr = nullptr;
    const uint8_t* triOffs = nullptr; const uint8_t* triIdx = nullptr;
    const uint8_t* mass = nullptr; const uint8_t* pins = nullptr;
    if (!section("positions", V, 12, positions)) return false;
    if (!section("triangles", T, 12, triangles)) return false;
    if (!section("edges", E, 8, edges)) return false;
    if (!section("edge_csr_offs", V + 1, 4, edgeOffs)) return false;
    if (!section("edge_csr_nbr", 2 * E, 4, edgeNbr)) return false;
    if (!section("tri_csr_offs", V + 1, 4, triOffs)) return false;
    if (!section("tri_csr_idx", 3 * T, 4, triIdx)) return false;
    if (!section("vertex_mass", V, 4, mass)) return false;
    if (!section("pin_mask", V, 4, pins)) return false;

    out.positionsCm         = reinterpret_cast<const float*>(positions);
    out.triangleIndices     = reinterpret_cast<const uint32_t*>(triangles);
    out.edgePairs           = reinterpret_cast<const uint32_t*>(edges);
    out.edgeCsrOffsets      = reinterpret_cast<const uint32_t*>(edgeOffs);
    out.edgeCsrNeighbours   = reinterpret_cast<const uint32_t*>(edgeNbr);
    out.triangleCsrOffsets  = reinterpret_cast<const uint32_t*>(triOffs);
    out.triangleCsrIndices  = reinterpret_cast<const uint32_t*>(triIdx);
    out.vertexMassKg        = reinterpret_cast<const float*>(mass);
    out.pinMask             = reinterpret_cast<const uint32_t*>(pins);

    if (!all_finite(out.positionsCm, size_t(V) * 3)) { err = "mesh: nonfinite vertex position"; return false; }
    if (!all_finite(out.vertexMassKg, V)) { err = "mesh: nonfinite vertex mass"; return false; }
    for (uint32_t v = 0; v < V; ++v) {
        if (!(out.vertexMassKg[v] > 0.0f)) { err = "mesh: vertex mass must be positive"; return false; }
    }

    uint32_t pinned = 0;
    for (uint32_t v = 0; v < V; ++v) {
        const uint32_t flag = out.pinMask[v];
        if (flag > 1) { err = "mesh: pin_mask must be 0 or 1"; return false; }
        pinned += flag;
    }
    if (pinned != out.pinnedVertices) { err = "mesh: pinned vertex count disagrees with pin_mask"; return false; }
    // An entirely free garment has no kinematic anchor, so the solver would let it
    // fall away from the body forever. That is a bake mistake, not a valid asset.
    if (pinned == 0) { err = "mesh: no pinned vertices"; return false; }
    if (pinned == V) { err = "mesh: every vertex is pinned"; return false; }

    // Optional pin bind. Required to be a matched pair: one section without the other would
    // give a bone with no offset or an offset with no bone, and either is a half-written file
    // rather than a mesh baked without reference clips.
    const uint8_t* pinDriver = nullptr; const uint8_t* pinLocal = nullptr;
    bool haveDriver = false, haveLocal = false;
    if (!find_optional_section(data, spans, "mesh", "pin_driver", V, 4, pinDriver, haveDriver, err)) return false;
    if (!find_optional_section(data, spans, "mesh", "pin_local_cm", V, 12, pinLocal, haveLocal, err)) return false;
    if (haveDriver != haveLocal) {
        err = "mesh: pin_driver and pin_local_cm must be written together";
        return false;
    }
    if (haveDriver) {
        out.pinDriverIndices = reinterpret_cast<const uint32_t*>(pinDriver);
        out.pinLocalCm       = reinterpret_cast<const float*>(pinLocal);
        if (!all_finite(out.pinLocalCm, size_t(V) * 3)) { err = "mesh: nonfinite pin_local_cm entry"; return false; }
        for (uint32_t v = 0; v < V; ++v) {
            const bool isPinned = out.pinMask[v] != 0;
            const uint32_t driver = out.pinDriverIndices[v];
            if (isPinned && driver >= kDriverCount) {
                err = "mesh: pinned vertex " + std::to_string(v) + " names driver " + std::to_string(driver)
                    + ", which is not one of the " + std::to_string(kDriverCount) + " model drivers";
                return false;
            }
            // A free vertex carrying a driver would be bound to the body by a branch that
            // reads this section, so it is refused rather than ignored.
            if (!isPinned && driver != kNoDriver) {
                err = "mesh: free vertex " + std::to_string(v) + " carries driver " + std::to_string(driver);
                return false;
            }
        }
    }

    // Triangles: in range, and no repeated corner (a repeated corner has zero area
    // for every configuration, so its area constraint and normal are meaningless).
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t a = out.triangleIndices[3 * t], b = out.triangleIndices[3 * t + 1], c = out.triangleIndices[3 * t + 2];
        if (a >= V || b >= V || c >= V) { err = "mesh: triangle index out of range"; return false; }
        if (a == b || b == c || a == c) { err = "mesh: degenerate triangle with a repeated corner"; return false; }
    }

    // Edges: a < b, in range, and strictly ascending so duplicates are impossible.
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t a = out.edgePairs[2 * e], b = out.edgePairs[2 * e + 1];
        if (a >= b || b >= V) { err = "mesh: edge is not an ordered in-range pair"; return false; }
        if (e > 0) {
            const uint32_t pa = out.edgePairs[2 * e - 2], pb = out.edgePairs[2 * e - 1];
            if (pa > a || (pa == a && pb >= b)) { err = "mesh: edges are not strictly ascending"; return false; }
        }
    }

    // Both CSRs: zero-based, monotonic, exact total, in-range entries.
    auto checkCsr = [&](const uint32_t* offsets, const uint32_t* values, uint32_t total, uint32_t limit,
                        const char* label, uint32_t& maximumRun) -> bool {
        if (offsets[0] != 0) { err = std::string("mesh: ") + label + " CSR does not start at zero"; return false; }
        if (offsets[V] != total) { err = std::string("mesh: ") + label + " CSR total is wrong"; return false; }
        maximumRun = 0;
        for (uint32_t v = 0; v < V; ++v) {
            if (offsets[v + 1] < offsets[v]) { err = std::string("mesh: ") + label + " CSR offsets are not monotonic"; return false; }
            maximumRun = std::max(maximumRun, offsets[v + 1] - offsets[v]);
        }
        for (uint32_t i = 0; i < total; ++i) {
            if (values[i] >= limit) { err = std::string("mesh: ") + label + " CSR entry out of range"; return false; }
        }
        return true;
    };
    uint32_t maximumDegree = 0, maximumValence = 0;
    if (!checkCsr(out.edgeCsrOffsets, out.edgeCsrNeighbours, 2 * E, V, "edge", maximumDegree)) return false;
    if (!checkCsr(out.triangleCsrOffsets, out.triangleCsrIndices, 3 * T, T, "triangle", maximumValence)) return false;
    if (maximumValence != out.maxTriangleValence) { err = "mesh: maxTriangleValence disagrees with the triangle CSR"; return false; }
    for (uint32_t v = 0; v < V; ++v) {
        for (uint32_t i = out.edgeCsrOffsets[v]; i < out.edgeCsrOffsets[v + 1]; ++i) {
            if (out.edgeCsrNeighbours[i] == v) { err = "mesh: edge CSR contains a self neighbour"; return false; }
            if (find_edge(out.edgePairs, E, v, out.edgeCsrNeighbours[i]) == kMax) {
                err = "mesh: edge CSR names a pair that is not in the edge list";
                return false;
            }
        }
    }

    // Manifoldness and coverage in one pass: every triangle edge must be in the
    // edge list, and every edge must carry one or two triangles. This also proves
    // the edge list is exactly the triangle edge set rather than merely a subset,
    // which is what makes it safe for the constraint baker to use either one.
    std::vector<uint8_t> edgeUse(E, 0);
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t corner[3] = { out.triangleIndices[3 * t], out.triangleIndices[3 * t + 1], out.triangleIndices[3 * t + 2] };
        for (uint32_t k = 0; k < 3; ++k) {
            const uint32_t index = find_edge(out.edgePairs, E, corner[k], corner[(k + 1) % 3]);
            if (index == kMax) { err = "mesh: a triangle edge is missing from the edge list"; return false; }
            if (edgeUse[index] >= 2) { err = "mesh: non-manifold edge shared by three or more triangles"; return false; }
            ++edgeUse[index];
        }
    }
    uint32_t boundaryEdges = 0;
    std::vector<uint32_t> boundaryDegree(V, 0);
    for (uint32_t e = 0; e < E; ++e) {
        if (edgeUse[e] == 0) { err = "mesh: edge list contains an edge no triangle uses"; return false; }
        if (edgeUse[e] == 1) {
            ++boundaryEdges;
            ++boundaryDegree[out.edgePairs[2 * e]];
            ++boundaryDegree[out.edgePairs[2 * e + 1]];
        }
    }
    if (boundaryEdges != out.boundaryEdges) { err = "mesh: boundaryEdges disagrees with the triangle adjacency"; return false; }
    // On a manifold-with-boundary every boundary vertex sits on exactly two
    // boundary edges, which is what makes "the highest boundary loop" a
    // well-defined pin rule rather than a heuristic over a branching curve.
    for (uint32_t v = 0; v < V; ++v) {
        if (boundaryDegree[v] != 0 && boundaryDegree[v] != 2) { err = "mesh: boundary is non-manifold at a vertex"; return false; }
    }
    DisjointSet loops(V);
    for (uint32_t e = 0; e < E; ++e) {
        if (edgeUse[e] == 1) loops.unite(out.edgePairs[2 * e], out.edgePairs[2 * e + 1]);
    }
    uint32_t loopCount = 0;
    for (uint32_t v = 0; v < V; ++v) {
        if (boundaryDegree[v] != 0 && loops.find(v) == v) ++loopCount;
    }
    if (loopCount != out.boundaryLoops) { err = "mesh: boundaryLoops disagrees with the boundary edge graph"; return false; }

    return true;
}

// ===========================================================================
// Writers (test fixture generation)
// ===========================================================================

std::vector<uint8_t> write_model(int modelType, int driverFeatureLen,
                                  int drivenFeatureLen, int pcaDim,
                                  int vertices,
                                  const std::vector<std::string>& driverNames,
                                  const std::vector<uint8_t>& payload) {
    // Build JSON string manually (no library).
    auto int_val = [](int v) -> std::string {
        return std::to_string(v);
    };
    auto str_val = [](const std::string& s) -> std::string {
        std::string r = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') r += '\\';
            r += c;
        }
        r += '"';
        return r;
    };

    std::string json = "{";
    json += "\"modelType\":" + int_val(modelType) + ",";
    json += "\"driverFeatureLen\":" + int_val(driverFeatureLen) + ",";
    json += "\"drivenFeatureLen\":" + int_val(drivenFeatureLen) + ",";
    json += "\"pcaDim\":" + int_val(pcaDim) + ",";
    json += "\"vertexCount\":" + int_val(vertices) + ",";
    json += "\"driverNames\":[";
    for (size_t i = 0; i < driverNames.size(); ++i) {
        if (i > 0) json += ",";
        json += str_val(driverNames[i]);
    }
    json += "]";
    json += "}";

    uint32_t jsonLen = static_cast<uint32_t>(json.size());
    size_t total = 4 + json.size() + payload.size();
    std::vector<uint8_t> buf(total);
    put_le32(buf.data(), jsonLen);
    std::memcpy(buf.data() + 4, json.data(), json.size());
    if (!payload.empty()) {
        std::memcpy(buf.data() + 4 + json.size(), payload.data(), payload.size());
    }
    return buf;
}

std::vector<uint8_t> write_clip(uint32_t frameCount,
                                 const Sha256Digest& modelHash,
                                 const Sha256Digest& driverListHash,
                                 const float* localFu,
                                 const float* componentFu,
                                 const float* componentPosCm) {
    uint32_t localCount  = frameCount * kDriverCount * 6;
    uint32_t compCount   = frameCount * kDriverCount * 6;
    uint32_t posCount    = frameCount * kDriverCount * 3;
    size_t payloadBytes  = size_t(localCount + compCount + posCount) * sizeof(float);
    size_t total         = kClipHeaderBytes + payloadBytes;

    std::vector<uint8_t> buf(total, 0);

    // Magic.
    std::memcpy(buf.data(), "MLDRV001", kMagicLen);

    // Header fields (little-endian).
    auto w32 = [&](size_t off, uint32_t v) { put_le32(buf.data() + off, v); };
    w32(8,  kClipVersion);
    w32(12, uint32_t(kClipHeaderBytes));
    w32(16, frameCount);
    w32(20, kFpsNum);
    w32(24, kFpsDen);
    w32(28, kDriverCount);
    w32(32, kRootDriverIndex);
    w32(36, localCount);
    w32(40, compCount);
    w32(44, posCount);

    // Compute payload hash.
    const uint8_t* payloadPtr = buf.data() + kClipHeaderBytes;
    // Copy payload data first.
    if (localCount != 0) std::memcpy(buf.data() + kClipHeaderBytes, localFu, localCount * sizeof(float));
    if (compCount != 0) std::memcpy(buf.data() + kClipHeaderBytes + localCount * sizeof(float),
                componentFu, compCount * sizeof(float));
    if (posCount != 0) std::memcpy(buf.data() + kClipHeaderBytes + (localCount + compCount) * sizeof(float),
                componentPosCm, posCount * sizeof(float));

    Sha256Digest payloadHash = sha256(payloadPtr, payloadBytes);

    // Write hashes.
    std::memcpy(buf.data() + 48,  modelHash.bytes.data(), 32);
    std::memcpy(buf.data() + 80,  driverListHash.bytes.data(), 32);
    std::memcpy(buf.data() + 112, payloadHash.bytes.data(), 32);

    return buf;
}

namespace {

struct MeshSection {
    const char* name;
    uint32_t count;
    uint32_t stride;
    std::vector<uint8_t> data;
};

// Mirror of real_scene/formats.py::write_sectioned, so a file produced here and
// one produced by the Python baker are byte-identical for identical sections.
std::vector<uint8_t> pack_sections(const char* magic, uint32_t version,
                                   const Sha256Digest& sourceHash,
                                   const std::vector<MeshSection>& sections) {
    const size_t directoryBytes = kSectionHeaderBytes + kSectionEntryBytes * sections.size();
    const size_t payloadOffset = align_up(directoryBytes, kSectionAlignment);
    std::vector<uint8_t> out(payloadOffset, 0);
    std::vector<uint64_t> offsets(sections.size());
    for (size_t i = 0; i < sections.size(); ++i) {
        const size_t aligned = align_up(out.size(), kSectionAlignment);
        out.resize(aligned, 0);
        offsets[i] = aligned;
        out.insert(out.end(), sections[i].data.begin(), sections[i].data.end());
    }
    const Sha256Digest payloadHash = sha256(out.data() + payloadOffset, out.size() - payloadOffset);
    std::memcpy(out.data(), magic, kMagicLen);
    put_le32(out.data() + 8, version);
    put_le32(out.data() + 12, static_cast<uint32_t>(sections.size()));
    put_le64(out.data() + 16, static_cast<uint64_t>(out.size()));
    put_le64(out.data() + 24, static_cast<uint64_t>(payloadOffset));
    std::memcpy(out.data() + 32, payloadHash.bytes.data(), 32);
    std::memcpy(out.data() + 64, sourceHash.bytes.data(), 32);
    for (size_t i = 0; i < sections.size(); ++i) {
        uint8_t* entry = out.data() + kSectionHeaderBytes + kSectionEntryBytes * i;
        const size_t nameLen = std::strlen(sections[i].name);
        std::memcpy(entry, sections[i].name, nameLen);
        put_le64(entry + kSectionNameBytes, offsets[i]);
        put_le32(entry + kSectionNameBytes + 8, sections[i].count);
        put_le32(entry + kSectionNameBytes + 12, sections[i].stride);
    }
    return out;
}

std::vector<uint8_t> pack_u32(const std::vector<uint32_t>& values) {
    std::vector<uint8_t> bytes(values.size() * 4);
    for (size_t i = 0; i < values.size(); ++i) put_le32(bytes.data() + 4 * i, values[i]);
    return bytes;
}

std::vector<uint8_t> pack_f32(const std::vector<float>& values) {
    std::vector<uint8_t> bytes(values.size() * 4);
    if (!values.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

} // anonymous namespace

std::vector<uint8_t> write_mesh(const Sha256Digest& modelHash,
                                 const std::vector<float>& positionsCm,
                                 const std::vector<uint32_t>& triangleIndices,
                                 const std::vector<float>& vertexMassKg,
                                 const std::vector<uint32_t>& pinMask,
                                 std::string& err,
                                 const std::vector<uint32_t>& pinDrivers,
                                 const std::vector<float>& pinLocalCm) {
    err.clear();
    if (positionsCm.empty() || positionsCm.size() % 3 != 0) { err = "write_mesh: positions must be a non-empty multiple of three"; return {}; }
    const size_t vertexCount = positionsCm.size() / 3;
    if (vertexCount > std::numeric_limits<uint32_t>::max()) { err = "write_mesh: too many vertices"; return {}; }
    const uint32_t V = static_cast<uint32_t>(vertexCount);
    if (vertexMassKg.size() != vertexCount || pinMask.size() != vertexCount) { err = "write_mesh: mass/pin arrays must be per-vertex"; return {}; }
    if (triangleIndices.empty() || triangleIndices.size() % 3 != 0) { err = "write_mesh: triangles must be a non-empty multiple of three"; return {}; }
    const uint32_t T = static_cast<uint32_t>(triangleIndices.size() / 3);
    for (size_t i = 0; i < triangleIndices.size(); ++i) {
        if (triangleIndices[i] >= V) { err = "write_mesh: triangle index out of range"; return {}; }
    }

    // Undirected edges with their triangle-use counts, from one sorted key array.
    // Counts above two are deliberately NOT rejected here: this writer exists so
    // the tests can build files that parse_mesh must refuse, and a
    // three-triangle edge is one of those cases.
    std::vector<uint64_t> keys;
    keys.reserve(size_t(T) * 3);
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t corner[3] = { triangleIndices[3 * t], triangleIndices[3 * t + 1], triangleIndices[3 * t + 2] };
        for (uint32_t k = 0; k < 3; ++k) {
            uint32_t a = corner[k], b = corner[(k + 1) % 3];
            if (a == b) { err = "write_mesh: degenerate triangle with a repeated corner"; return {}; }
            if (a > b) { const uint32_t s = a; a = b; b = s; }
            keys.push_back((uint64_t(a) << 32) | uint64_t(b));
        }
    }
    std::sort(keys.begin(), keys.end());
    std::vector<uint32_t> edgePairs, edgeUse;
    for (size_t i = 0; i < keys.size();) {
        size_t j = i;
        while (j < keys.size() && keys[j] == keys[i]) ++j;
        edgePairs.push_back(static_cast<uint32_t>(keys[i] >> 32));
        edgePairs.push_back(static_cast<uint32_t>(keys[i] & 0xffffffffull));
        edgeUse.push_back(static_cast<uint32_t>(j - i));
        i = j;
    }
    const uint32_t E = static_cast<uint32_t>(edgeUse.size());

    // Directed neighbour CSR. Sorting the (from, to) keys keeps every vertex's
    // neighbour run ascending, which is what parse_mesh's monotonicity and
    // membership checks expect.
    std::vector<uint64_t> directed;
    directed.reserve(size_t(E) * 2);
    for (uint32_t e = 0; e < E; ++e) {
        const uint64_t a = edgePairs[2 * e], b = edgePairs[2 * e + 1];
        directed.push_back((a << 32) | b);
        directed.push_back((b << 32) | a);
    }
    std::sort(directed.begin(), directed.end());
    std::vector<uint32_t> edgeOffsets(size_t(V) + 1, 0), edgeNeighbours(directed.size());
    for (size_t i = 0; i < directed.size(); ++i) {
        edgeNeighbours[i] = static_cast<uint32_t>(directed[i] & 0xffffffffull);
        ++edgeOffsets[static_cast<size_t>(directed[i] >> 32) + 1];
    }
    for (uint32_t v = 0; v < V; ++v) edgeOffsets[size_t(v) + 1] += edgeOffsets[v];

    // Incident-triangle CSR, built the same way.
    std::vector<uint64_t> incident;
    incident.reserve(size_t(T) * 3);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t k = 0; k < 3; ++k) {
            incident.push_back((uint64_t(triangleIndices[3 * t + k]) << 32) | uint64_t(t));
        }
    }
    std::sort(incident.begin(), incident.end());
    std::vector<uint32_t> triOffsets(size_t(V) + 1, 0), triIndices(incident.size());
    for (size_t i = 0; i < incident.size(); ++i) {
        triIndices[i] = static_cast<uint32_t>(incident[i] & 0xffffffffull);
        ++triOffsets[static_cast<size_t>(incident[i] >> 32) + 1];
    }
    for (uint32_t v = 0; v < V; ++v) triOffsets[size_t(v) + 1] += triOffsets[v];
    uint32_t maximumValence = 0;
    for (uint32_t v = 0; v < V; ++v) maximumValence = std::max(maximumValence, triOffsets[size_t(v) + 1] - triOffsets[v]);

    // Boundary edges, and loops as the components of the boundary edge graph --
    // the same derivation parse_mesh re-runs, so the two must agree.
    uint32_t boundaryEdges = 0;
    std::vector<uint32_t> boundaryDegree(V, 0);
    DisjointSet loops(V);
    for (uint32_t e = 0; e < E; ++e) {
        if (edgeUse[e] != 1) continue;
        ++boundaryEdges;
        ++boundaryDegree[edgePairs[2 * e]];
        ++boundaryDegree[edgePairs[2 * e + 1]];
        loops.unite(edgePairs[2 * e], edgePairs[2 * e + 1]);
    }
    uint32_t boundaryLoops = 0;
    for (uint32_t v = 0; v < V; ++v) {
        if (boundaryDegree[v] != 0 && loops.find(v) == v) ++boundaryLoops;
    }
    uint32_t pinned = 0;
    for (uint32_t v = 0; v < V; ++v) pinned += (pinMask[v] != 0 ? 1u : 0u);

    const std::vector<uint32_t> info = { V, T, E, boundaryEdges, boundaryLoops, maximumValence, pinned, 0u };
    std::vector<MeshSection> sections = {
        { "info",          8,         4,  pack_u32(info) },
        { "positions",     V,         12, pack_f32(positionsCm) },
        { "triangles",     T,         12, pack_u32(triangleIndices) },
        { "edges",         E,         8,  pack_u32(edgePairs) },
        { "edge_csr_offs", V + 1,     4,  pack_u32(edgeOffsets) },
        { "edge_csr_nbr",  2 * E,     4,  pack_u32(edgeNeighbours) },
        { "tri_csr_offs",  V + 1,     4,  pack_u32(triOffsets) },
        { "tri_csr_idx",   3 * T,     4,  pack_u32(triIndices) },
        { "vertex_mass",   V,         4,  pack_f32(vertexMassKg) },
        { "pin_mask",      V,         4,  pack_u32(pinMask) },
    };
    if (!pinDrivers.empty() || !pinLocalCm.empty()) {
        if (pinDrivers.size() != vertexCount || pinLocalCm.size() != vertexCount * 3) {
            err = "write_mesh: pin bind arrays must be per-vertex and supplied together";
            return {};
        }
        sections.push_back({ "pin_driver",   V, 4,  pack_u32(pinDrivers) });
        sections.push_back({ "pin_local_cm", V, 12, pack_f32(pinLocalCm) });
    }
    return pack_sections("MLMSH001", kMeshVersion, modelHash, sections);
}

bool parse_capsules(const uint8_t* data, size_t len,
                    const Sha256Digest& expectedModelHash,
                    CapsuleInfo& out, std::string& err) {
    std::vector<SectionSpan> spans;
    if (!parse_container(data, len, "MLCAP001", "capsules", kCapsuleVersion, expectedModelHash,
                         spans, out.payloadSha256, out.modelSha256, err)) {
        return false;
    }

    const uint8_t* infoBytes = nullptr;
    if (!find_section(data, spans, "capsules", "info", 4, 4, infoBytes, err)) return false;
    out.count = le32(infoBytes + 0);
    out.driverCount = le32(infoBytes + 4);
    if (out.count == 0 || out.count > 256) { err = "capsules: implausible capsule count"; return false; }
    if (out.driverCount != kDriverCount) { err = "capsules: driverCount != 45"; return false; }

    const uint8_t* driverBytes = nullptr;
    const uint8_t* centreBytes = nullptr;
    const uint8_t* axisBytes = nullptr;
    const uint8_t* sizeBytes = nullptr;
    if (!find_section(data, spans, "capsules", "driver", out.count, 4, driverBytes, err)) return false;
    if (!find_section(data, spans, "capsules", "center", out.count, 12, centreBytes, err)) return false;
    if (!find_section(data, spans, "capsules", "axis", out.count, 12, axisBytes, err)) return false;
    if (!find_section(data, spans, "capsules", "size", out.count, 8, sizeBytes, err)) return false;

    out.driverIndices = reinterpret_cast<const uint32_t*>(driverBytes);
    out.centresCm = reinterpret_cast<const float*>(centreBytes);
    out.axes = reinterpret_cast<const float*>(axisBytes);
    out.sizesCm = reinterpret_cast<const float*>(sizeBytes);

    for (uint32_t i = 0; i < out.count; ++i) {
        if (out.driverIndices[i] >= out.driverCount) {
            err = "capsules: a capsule rides a bone outside the driver table";
            return false;
        }
        for (int k = 0; k < 3; ++k) {
            if (!std::isfinite(out.centresCm[3 * i + k]) || !std::isfinite(out.axes[3 * i + k])) {
                err = "capsules: a centre or axis is not finite";
                return false;
            }
        }
        // A non-unit axis would silently shorten or stretch the segment, because the signed
        // distance projects onto it without normalising. Checked rather than normalised here:
        // the bake writes unit axes, so a non-unit one means the file is not what it claims.
        const float* a = out.axes + 3 * i;
        const float norm = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        if (!(std::fabs(norm - 1.0f) <= 1.0e-4f)) { err = "capsules: an axis is not unit length"; return false; }
        const float radius = out.sizesCm[2 * i];
        const float halfLength = out.sizesCm[2 * i + 1];
        if (!(radius > 0.0f) || !std::isfinite(radius)) { err = "capsules: a radius is not positive and finite"; return false; }
        if (!(halfLength >= 0.0f) || !std::isfinite(halfLength)) { err = "capsules: a half length is negative or not finite"; return false; }
    }
    // Distinct driver bones, because two capsules on one bone would be a bake that merged or
    // duplicated geometry rather than the one-capsule-per-body layout this reads.
    for (uint32_t i = 0; i < out.count; ++i) {
        for (uint32_t j = i + 1; j < out.count; ++j) {
            if (out.driverIndices[i] == out.driverIndices[j]) {
                err = "capsules: two capsules share a driver bone";
                return false;
            }
        }
    }
    return true;
}

bool parse_body(const uint8_t* data, size_t len,
                const Sha256Digest& expectedModelHash,
                BodyInfo& out, std::string& err) {
    std::vector<SectionSpan> spans;
    if (!parse_container(data, len, "MLBDY001", "body", kBodyVersion, expectedModelHash,
                         spans, out.payloadSha256, out.modelSha256, err)) {
        return false;
    }

    const uint8_t* infoBytes = nullptr;
    if (!find_section(data, spans, "body", "info", 6, 4, infoBytes, err)) return false;
    out.vertices    = le32(infoBytes + 0);
    out.triangles   = le32(infoBytes + 4);
    out.drivers     = le32(infoBytes + 8);
    out.influences  = le32(infoBytes + 12);
    out.foldedBones = le32(infoBytes + 16);
    if (le32(infoBytes + 20) != 0) { err = "body: reserved info word is not zero"; return false; }

    if (out.vertices == 0 || out.triangles == 0) { err = "body: no vertices or no triangles"; return false; }
    constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
    if (out.triangles > kMax / 3) { err = "body: triangle count overflows"; return false; }
    if (out.drivers != kDriverCount) {
        err = "body: drivers != " + std::to_string(kDriverCount);
        return false;
    }
    // Zero would leave every vertex unskinned at the origin; above the source rig's twelve
    // means the file is not what the bake produces.
    if (out.influences == 0 || out.influences > kMaxBodyInfluences) {
        err = "body: influences is " + std::to_string(out.influences) + ", outside 1.."
            + std::to_string(kMaxBodyInfluences);
        return false;
    }
    if (out.foldedBones >= out.drivers) { err = "body: every bone cannot be a folded one"; return false; }
    if (out.vertices > kMax / out.influences) { err = "body: influence array overflows"; return false; }

    const uint8_t* positions = nullptr; const uint8_t* normals = nullptr;
    const uint8_t* indices = nullptr; const uint8_t* weights = nullptr;
    const uint8_t* triangles = nullptr; const uint8_t* inverseBind = nullptr;
    const uint32_t influenceStride = out.influences * 4;
    if (!find_section(data, spans, "body", "rest_pos", out.vertices, 12, positions, err)) return false;
    if (!find_section(data, spans, "body", "rest_nrm", out.vertices, 12, normals, err)) return false;
    if (!find_section(data, spans, "body", "bone_idx", out.vertices, influenceStride, indices, err)) return false;
    if (!find_section(data, spans, "body", "bone_weight", out.vertices, influenceStride, weights, err)) return false;
    if (!find_section(data, spans, "body", "tri", out.triangles, 12, triangles, err)) return false;
    if (!find_section(data, spans, "body", "inv_bind", out.drivers, 48, inverseBind, err)) return false;

    out.restPositionsCm = reinterpret_cast<const float*>(positions);
    out.restNormals     = reinterpret_cast<const float*>(normals);
    out.boneIndices     = reinterpret_cast<const uint32_t*>(indices);
    out.boneWeights     = reinterpret_cast<const float*>(weights);
    out.triangleIndices = reinterpret_cast<const uint32_t*>(triangles);
    out.inverseBind     = reinterpret_cast<const float*>(inverseBind);

    const size_t influenceCount = size_t(out.vertices) * out.influences;
    if (!all_finite(out.restPositionsCm, size_t(out.vertices) * 3)) { err = "body: nonfinite rest position"; return false; }
    if (!all_finite(out.restNormals, size_t(out.vertices) * 3)) { err = "body: nonfinite rest normal"; return false; }
    if (!all_finite(out.boneWeights, influenceCount)) { err = "body: nonfinite bone weight"; return false; }
    if (!all_finite(out.inverseBind, size_t(out.drivers) * 12)) { err = "body: nonfinite inverse bind entry"; return false; }

    for (uint32_t t = 0; t < out.triangles; ++t) {
        for (uint32_t k = 0; k < 3; ++k) {
            if (out.triangleIndices[3 * t + k] >= out.vertices) {
                err = "body: triangle index out of range";
                return false;
            }
        }
    }

    for (uint32_t v = 0; v < out.vertices; ++v) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < out.influences; ++k) {
            const uint32_t driver = out.boneIndices[size_t(v) * out.influences + k];
            const float weight = out.boneWeights[size_t(v) * out.influences + k];
            if (driver >= out.drivers) { err = "body: an influence names a bone outside the driver table"; return false; }
            if (weight < 0.0f) { err = "body: negative bone weight"; return false; }
            sum += weight;
        }
        // Weights that do not sum to one scale the vertex towards the component origin,
        // which reads as a dent rather than as a broken file, so it is refused here. The
        // bake normalises, so a deviation means the payload was edited or truncated.
        if (!(std::fabs(sum - 1.0f) <= 1.0e-4f)) {
            err = "body: vertex " + std::to_string(v) + " has skin weights summing to " + std::to_string(sum);
            return false;
        }
    }

    // A unit rest normal, for the same reason the capsule axis is checked: the skinning pass
    // rotates it without renormalising, so a short one shades wrongly rather than failing.
    for (uint32_t v = 0; v < out.vertices; ++v) {
        const float* n = out.restNormals + size_t(v) * 3;
        const float norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (!(std::fabs(norm - 1.0f) <= 1.0e-3f)) { err = "body: a rest normal is not unit length"; return false; }
    }
    return true;
}

} // namespace mlcloth
