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
    if (out.vertices != 5294) { err = "model: derived vertex count != 5294"; return false; }
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

} // namespace mlcloth
