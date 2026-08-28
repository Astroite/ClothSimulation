#include "mlcloth_formats.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>

using namespace mlcloth;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

#define TEST(name) static void name(); \
    struct name##_reg { name##_reg() { /* register */ } } name##_inst; \
    static void name()

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::cerr << "  FAIL: " << #expr << "  (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        ++g_fail; return; \
    } \
} while(0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

#define RUN(name) do { \
    std::cerr << "  " << #name << "... "; \
    const int failuresBefore = g_fail; \
    name(); \
    if (g_fail == failuresBefore) { std::cerr << "ok\n"; ++g_pass; } \
} while(0)

// Assert that a mesh is rejected AND that it is rejected by the intended check.
// Without the message assertion a negative test can quietly start passing for an
// earlier reason -- a patched triangle index also breaks edge coverage, a patched
// count also breaks a bounds check -- and stop covering what it was written for.
#define CHECK_MESH_REJECTS(bytes, hash, needle) do { \
    MeshInfo rejected{}; \
    std::string reason; \
    if (parse_mesh((bytes).data(), (bytes).size(), hash, rejected, reason)) { \
        std::cerr << "  FAIL: expected rejection containing \"" << (needle) << "\"  (" \
                  << __FILE__ << ":" << __LINE__ << ")\n"; \
        ++g_fail; return; \
    } \
    if (reason.find(needle) == std::string::npos) { \
        std::cerr << "  FAIL: rejected for the wrong reason: got \"" << reason \
                  << "\", expected to contain \"" << (needle) << "\"  (" \
                  << __FILE__ << ":" << __LINE__ << ")\n"; \
        ++g_fail; return; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Fixture generators
// ---------------------------------------------------------------------------

static std::vector<std::string> make_driver_names() {
    std::vector<std::string> names(45);
    names[0] = "Root_M";
    for (int i = 1; i < 45; ++i) names[i] = "Driver_" + std::to_string(i);
    return names;
}

static std::vector<uint8_t> make_model_bytes(const std::vector<std::string>& names,
                                              const std::vector<uint8_t>& payload = {}) {
    return write_model(2, 1969, 16394, 512, 5294, names, payload);
}

static std::vector<uint8_t> make_model_bytes() {
    return make_model_bytes(make_driver_names(), {0x01, 0x02, 0x03, 0x04});
}

static Sha256Digest compute_model_hash(const std::vector<uint8_t>& model) {
    return sha256(model);
}

static Sha256Digest compute_driver_list_hash(const std::vector<std::string>& names) {
    return sha256_driver_names(names);
}

// Generate a valid clip with deterministic float data.
static std::vector<uint8_t> make_clip_bytes(uint32_t frameCount,
                                             const Sha256Digest& modelHash,
                                             const Sha256Digest& driverListHash,
                                             std::vector<float>& localOut,
                                             std::vector<float>& compOut,
                                             std::vector<float>& posOut) {
    size_t localN = size_t(frameCount) * kDriverCount * 6;
    size_t compN  = size_t(frameCount) * kDriverCount * 6;
    size_t posN   = size_t(frameCount) * kDriverCount * 3;

    localOut.resize(localN);
    compOut.resize(compN);
    posOut.resize(posN);

    for (size_t i = 0; i < localN; ++i) localOut[i] = float(i % 100) * 0.01f;
    for (size_t i = 0; i < compN;  ++i) compOut[i]  = float(i % 50) * 0.02f;
    for (size_t i = 0; i < posN;   ++i) posOut[i]   = float(i % 200) * 0.005f;

    return write_clip(frameCount, modelHash, driverListHash,
                      localOut.data(), compOut.data(), posOut.data());
}

// ---------------------------------------------------------------------------
// SHA-256 tests
// ---------------------------------------------------------------------------

static void test_sha256_empty() {
    Sha256Digest d = sha256(nullptr, 0);
    std::string hex = sha256_hex(d);
    CHECK_EQ(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

static void test_sha256_abc() {
    const uint8_t data[] = {'a', 'b', 'c'};
    Sha256Digest d = sha256(data, 3);
    std::string hex = sha256_hex(d);
    CHECK_EQ(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

static void test_sha256_driver_names_single() {
    std::vector<std::string> names = {"Root_M"};
    Sha256Digest d = sha256_driver_names(names);
    // Should be sha256("Root_M")
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>("Root_M");
    Sha256Digest expected = sha256(bytes, 6);
    CHECK_EQ(d, expected);
}

static void test_sha256_driver_names_multiple() {
    std::vector<std::string> names = {"A", "B", "C"};
    Sha256Digest d = sha256_driver_names(names);
    // Should be sha256("A\nB\nC")
    const char* joined = "A\nB\nC";
    Sha256Digest expected = sha256(reinterpret_cast<const uint8_t*>(joined), 5);
    CHECK_EQ(d, expected);
}

static void test_sha256_hex_lowercase() {
    Sha256Digest d = sha256(nullptr, 0);
    std::string hex = sha256_hex(d);
    for (char c : hex) CHECK((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
}

// ---------------------------------------------------------------------------
// Model parser tests
// ---------------------------------------------------------------------------

static void test_model_valid() {
    auto bytes = make_model_bytes();
    ModelInfo info;
    std::string err;
    CHECK(parse_model(bytes.data(), bytes.size(), info, err));
    CHECK_EQ(info.modelType, 2);
    CHECK_EQ(info.driverFeatureLen, 1969);
    CHECK_EQ(info.drivenFeatureLen, 16394);
    CHECK_EQ(info.pcaDim, 512);
    CHECK_EQ(info.vertices, 5294);
    CHECK_EQ(info.driverNames.size(), 45u);
    CHECK_EQ(info.driverNames[0], "Root_M");
    CHECK(info.payloadLen > 0);
}

static void test_model_truncated_header() {
    std::vector<uint8_t> bytes = {0x01, 0x00}; // only 2 bytes, need 4
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_truncated_json() {
    auto full = make_model_bytes();
    // Truncate to just past the length prefix.
    std::vector<uint8_t> bytes(full.begin(), full.begin() + 10);
    // Fix length to claim more than we have.
    uint32_t bigLen = 1000;
    std::memcpy(bytes.data(), &bigLen, 4);
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_bad_model_type() {
    auto names = make_driver_names();
    auto bytes = write_model(99, 1969, 16394, 512, 5294, names, {});
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_bad_vertices() {
    auto names = make_driver_names();
    auto bytes = write_model(2, 1969, 16394, 512, 9999, names, {});
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_bad_driver_feature_len() {
    auto bytes = write_model(2, 1968, 16394, 512, 5294, make_driver_names(), {});
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_bad_driven_feature_len() {
    auto bytes = write_model(2, 1969, 16391, 512, 5293, make_driver_names(), {});
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_bad_pca_dim() {
    auto bytes = write_model(2, 1969, 16394, 256, 5294, make_driver_names(), {});
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_duplicate_driver() {
    auto names = make_driver_names();
    names[2] = names[1];
    auto bytes = write_model(2, 1969, 16394, 512, 5294, names, {});
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_bad_driver_count() {
    std::vector<std::string> names(10, "D"); // too few
    auto bytes = write_model(2, 1969, 16394, 512, 5294, names, {});
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_bad_first_driver() {
    auto names = make_driver_names();
    names[0] = "Wrong"; // not "Root_M"
    auto bytes = write_model(2, 1969, 16394, 512, 5294, names, {});
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_null_byte_in_json() {
    auto full = make_model_bytes();
    // Inject a null byte in the JSON section (after the length prefix).
    std::vector<uint8_t> bytes = full;
    bytes[10] = 0; // inside JSON
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

static void test_model_empty_buffer() {
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(nullptr, 0, info, err));
}

static void test_model_json_len_zero() {
    std::vector<uint8_t> bytes = {0, 0, 0, 0}; // json_len = 0
    ModelInfo info;
    std::string err;
    CHECK(!parse_model(bytes.data(), bytes.size(), info, err));
}

// ---------------------------------------------------------------------------
// Clip parser tests
// ---------------------------------------------------------------------------

static void test_clip_valid() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(3, modelHash, driverHash, localF, compF, posF);

    ClipInfo info;
    std::string err;
    CHECK(parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
    CHECK_EQ(info.header.frameCount, 3u);
    CHECK_EQ(info.header.driverCount, kDriverCount);
    CHECK_EQ(info.header.rootDriverIndex, 0u);
    CHECK_EQ(info.header.version, 1u);
    CHECK(info.localFu != nullptr);
    CHECK(info.componentFu != nullptr);
    CHECK(info.componentPosCm != nullptr);

    // Verify data integrity.
    size_t localN = 3 * kDriverCount * 6;
    size_t compN  = 3 * kDriverCount * 6;
    size_t posN   = 3 * kDriverCount * 3;
    for (size_t i = 0; i < localN; ++i) CHECK(info.localFu[i] == localF[i]);
    for (size_t i = 0; i < compN;  ++i) CHECK(info.componentFu[i] == compF[i]);
    for (size_t i = 0; i < posN;   ++i) CHECK(info.componentPosCm[i] == posF[i]);
}

static void test_clip_truncated_header() {
    std::vector<uint8_t> bytes(100, 0); // less than 144
    ClipInfo info;
    std::string err;
    Sha256Digest zero{};
    CHECK(!parse_clip(bytes.data(), bytes.size(), zero, zero, info, err));
}

static void test_clip_bad_magic() {
    std::vector<uint8_t> bytes(200, 0);
    std::memcpy(bytes.data(), "BADMGIC!", 8);
    ClipInfo info;
    std::string err;
    Sha256Digest zero{};
    CHECK(!parse_clip(bytes.data(), bytes.size(), zero, zero, info, err));
}

static void test_clip_bad_version() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(1, modelHash, driverHash, localF, compF, posF);

    // Tamper version.
    clipBytes[8] = 2; // version = 2

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

static void test_clip_bad_fps() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(1, modelHash, driverHash, localF, compF, posF);

    // Tamper fps numerator.
    uint32_t badFps = 60;
    std::memcpy(clipBytes.data() + 20, &badFps, 4);

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

static void test_clip_bad_driver_count() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(1, modelHash, driverHash, localF, compF, posF);

    // Tamper driver count.
    uint32_t bad = 10;
    std::memcpy(clipBytes.data() + 28, &bad, 4);

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

static void test_clip_bad_root_driver_index() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(1, modelHash, driverHash, localF, compF, posF);

    // Tamper root driver index.
    uint32_t bad = 5;
    std::memcpy(clipBytes.data() + 32, &bad, 4);

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

static void test_clip_bad_model_hash() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(1, modelHash, driverHash, localF, compF, posF);

    // Use wrong expected model hash.
    Sha256Digest wrong{};
    wrong.bytes[0] = 0xFF;

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), wrong, driverHash, info, err));
}

static void test_clip_bad_driver_list_hash() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(1, modelHash, driverHash, localF, compF, posF);

    // Use wrong expected driver list hash.
    Sha256Digest wrong{};
    wrong.bytes[0] = 0xFF;

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, wrong, info, err));
}

static void test_clip_payload_tamper() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(1, modelHash, driverHash, localF, compF, posF);

    // Tamper one payload byte.
    clipBytes[kClipHeaderBytes + 5] ^= 0xFF;

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

static void test_clip_truncated_payload() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(2, modelHash, driverHash, localF, compF, posF);

    // Truncate payload.
    clipBytes.resize(clipBytes.size() - 100);

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

static void test_clip_nonfinite_floats() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(1, modelHash, driverHash, localF, compF, posF);

    // Inject NaN into payload (after header).
    float nanVal = std::nanf("");
    size_t offset = kClipHeaderBytes + 10 * sizeof(float); // somewhere in localFu
    std::memcpy(clipBytes.data() + offset, &nanVal, sizeof(float));

    // Need to recompute payload hash since we tampered.
    // Actually, we want to test that the parser catches nonfinite BEFORE hash check.
    // But the parser checks hash first. So we need to provide the correct hash
    // for the tampered data. Let's just rebuild with NaN in the source data.

    localF[10] = std::nanf("");
    auto clipBytes2 = write_clip(1, modelHash, driverHash,
                                  localF.data(), compF.data(), posF.data());

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes2.data(), clipBytes2.size(), modelHash, driverHash, info, err));
}

static void test_clip_nonfinite_component_fu() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    size_t localN = kDriverCount * 6;
    size_t compN  = kDriverCount * 6;
    localF.resize(localN, 0.0f);
    compF.resize(compN, 0.0f);
    posF.resize(kDriverCount * 3, 0.0f);

    compF[0] = std::numeric_limits<float>::infinity();

    auto clipBytes = write_clip(1, modelHash, driverHash,
                                 localF.data(), compF.data(), posF.data());

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

static void test_clip_nonfinite_pos_cm() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    size_t localN = kDriverCount * 6;
    size_t compN  = kDriverCount * 6;
    size_t posN   = kDriverCount * 3;
    localF.resize(localN, 0.0f);
    compF.resize(compN, 0.0f);
    posF.resize(posN, 0.0f);

    posF[5] = std::numeric_limits<float>::quiet_NaN();

    auto clipBytes = write_clip(1, modelHash, driverHash,
                                 localF.data(), compF.data(), posF.data());

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

static void test_clip_bad_header_bytes_field() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(1, modelHash, driverHash, localF, compF, posF);

    // Tamper headerBytes field.
    uint32_t bad = 200;
    std::memcpy(clipBytes.data() + 12, &bad, 4);

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

static void test_clip_bad_local_float_count() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(1, modelHash, driverHash, localF, compF, posF);

    // Tamper localFloatCount.
    uint32_t bad = 999;
    std::memcpy(clipBytes.data() + 36, &bad, 4);

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

static void test_clip_multi_frame() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(10, modelHash, driverHash, localF, compF, posF);

    ClipInfo info;
    std::string err;
    CHECK(parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
    CHECK_EQ(info.header.frameCount, 10u);
}

static void test_clip_zero_frames() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names);
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    // 0 frames: header only, no payload.
    auto clipBytes = write_clip(0, modelHash, driverHash, nullptr, nullptr, nullptr);

    ClipInfo info;
    std::string err;
    CHECK(!parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, info, err));
}

// ---------------------------------------------------------------------------
// Model + Clip integration
// ---------------------------------------------------------------------------

static void test_model_clip_integration() {
    auto names = make_driver_names();
    auto modelBytes = make_model_bytes(names, {0xAA, 0xBB});
    auto modelHash = compute_model_hash(modelBytes);
    auto driverHash = compute_driver_list_hash(names);

    ModelInfo minfo;
    std::string err;
    CHECK(parse_model(modelBytes.data(), modelBytes.size(), minfo, err));
    CHECK_EQ(minfo.driverNames.size(), 45u);

    std::vector<float> localF, compF, posF;
    auto clipBytes = make_clip_bytes(2, modelHash, driverHash, localF, compF, posF);

    ClipInfo cinfo;
    CHECK(parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, cinfo, err));
    CHECK_EQ(cinfo.header.frameCount, 2u);

    // Verify model hash in clip matches.
    CHECK_EQ(cinfo.header.modelSha256, modelHash);
    CHECK_EQ(cinfo.header.driverListSha256, driverHash);
}

// ---------------------------------------------------------------------------
// Mesh parser tests
// ---------------------------------------------------------------------------

// A closed cylinder strip: `rows` rings of `cols` vertices with the columns
// wrapping. Chosen because 5294 = 2 * 2647 with 2647 prime, so a two-ring
// cylinder is the only grid shape that hits the model's vertex count exactly --
// and unlike a flat patch it has two boundary loops, which exercises the loop
// counter rather than leaving it at one.
struct MeshFixture {
    std::vector<float> positions;
    std::vector<uint32_t> triangles;
    std::vector<float> mass;
    std::vector<uint32_t> pins;
};

static MeshFixture make_cylinder(uint32_t rows, uint32_t cols) {
    MeshFixture fixture;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const double angle = 6.283185307179586 * double(c) / double(cols);
            fixture.positions.push_back(float(20.0 * std::cos(angle)));
            fixture.positions.push_back(float(-20.0 * double(r)));
            fixture.positions.push_back(float(20.0 * std::sin(angle)));
            fixture.mass.push_back(0.002f);
            fixture.pins.push_back(r == 0 ? 1u : 0u);
        }
    }
    for (uint32_t r = 0; r + 1 < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const uint32_t next = (c + 1) % cols;
            const uint32_t v00 = r * cols + c, v01 = r * cols + next;
            const uint32_t v10 = (r + 1) * cols + c, v11 = (r + 1) * cols + next;
            fixture.triangles.insert(fixture.triangles.end(), { v00, v10, v11 });
            fixture.triangles.insert(fixture.triangles.end(), { v00, v11, v01 });
        }
    }
    return fixture;
}

static std::vector<uint8_t> make_mesh_bytes(const Sha256Digest& modelHash, const MeshFixture& fixture) {
    std::string err;
    auto bytes = write_mesh(modelHash, fixture.positions, fixture.triangles, fixture.mass, fixture.pins, err);
    if (bytes.empty()) std::cerr << "  (write_mesh refused the fixture: " << err << ")\n";
    return bytes;
}

static std::vector<uint8_t> make_mesh_bytes(const Sha256Digest& modelHash) {
    return make_mesh_bytes(modelHash, make_cylinder(2, kClothVertexCount / 2));
}

static uint32_t read_le32(const std::vector<uint8_t>& bytes, size_t offset) {
    return uint32_t(bytes[offset]) | (uint32_t(bytes[offset + 1]) << 8) |
           (uint32_t(bytes[offset + 2]) << 16) | (uint32_t(bytes[offset + 3]) << 24);
}

static uint64_t read_le64(const std::vector<uint8_t>& bytes, size_t offset) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) value = (value << 8) | uint64_t(bytes[offset + size_t(i)]);
    return value;
}

static void write_le32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    for (int i = 0; i < 4; ++i) bytes[offset + size_t(i)] = uint8_t(value >> (8 * i));
}

// Byte offset of a named section's payload, or SIZE_MAX when absent.
static size_t mesh_section_offset(const std::vector<uint8_t>& bytes, const char* name) {
    const uint32_t sections = read_le32(bytes, 12);
    for (uint32_t i = 0; i < sections; ++i) {
        const size_t entry = kSectionHeaderBytes + kSectionEntryBytes * size_t(i);
        std::string found(reinterpret_cast<const char*>(bytes.data() + entry),
                          strnlen(reinterpret_cast<const char*>(bytes.data() + entry), kSectionNameBytes));
        if (found == name) return size_t(read_le64(bytes, entry + kSectionNameBytes));
    }
    return SIZE_MAX;
}

// Recompute the payload digest after a deliberate edit. Without this every patch
// would stop at the hash check, so the deeper structural checks would never be
// reached by any test.
static void mesh_reseal(std::vector<uint8_t>& bytes) {
    const size_t payloadOffset = size_t(read_le64(bytes, 24));
    const Sha256Digest digest = sha256(bytes.data() + payloadOffset, bytes.size() - payloadOffset);
    std::memcpy(bytes.data() + 32, digest.bytes.data(), 32);
}

static void mesh_patch_u32(std::vector<uint8_t>& bytes, const char* section, size_t index, uint32_t value) {
    const size_t base = mesh_section_offset(bytes, section);
    write_le32(bytes, base + 4 * index, value);
    mesh_reseal(bytes);
}

static void mesh_patch_f32(std::vector<uint8_t>& bytes, const char* section, size_t index, float value) {
    const size_t base = mesh_section_offset(bytes, section);
    std::memcpy(bytes.data() + base + 4 * index, &value, sizeof(value));
    mesh_reseal(bytes);
}

static void test_mesh_valid() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    CHECK(!bytes.empty());

    MeshInfo info;
    std::string err;
    CHECK(parse_mesh(bytes.data(), bytes.size(), modelHash, info, err));
    const uint32_t cols = kClothVertexCount / 2;
    CHECK_EQ(info.vertices, kClothVertexCount);
    CHECK_EQ(info.triangles, 2u * cols);
    CHECK_EQ(info.edges, 4u * cols);
    CHECK_EQ(info.boundaryEdges, 2u * cols);
    CHECK_EQ(info.boundaryLoops, 2u);
    CHECK_EQ(info.maxTriangleValence, 3u);
    CHECK_EQ(info.pinnedVertices, cols);
    CHECK_EQ(info.modelSha256, modelHash);
    // Euler characteristic of an annulus is zero; a wrong edge derivation shows up here.
    CHECK_EQ(int(info.vertices) - int(info.edges) + int(info.triangles), 0);
    CHECK_EQ(info.edgeCsrOffsets[info.vertices], 2u * info.edges);
    CHECK_EQ(info.triangleCsrOffsets[info.vertices], 3u * info.triangles);
}

static void test_mesh_bad_magic() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    bytes[3] = 'X';
    CHECK_MESH_REJECTS(bytes, modelHash, "bad magic");
}

static void test_mesh_bad_version() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    write_le32(bytes, 8, 2);
    CHECK_MESH_REJECTS(bytes, modelHash, "version != 1");
}

// The lock that stops a topology from another garment loading against this model.
static void test_mesh_bad_model_hash() {
    auto names = make_driver_names();
    auto modelHash = compute_model_hash(make_model_bytes());
    auto otherHash = compute_driver_list_hash(names);
    auto bytes = make_mesh_bytes(modelHash);
    CHECK_MESH_REJECTS(bytes, otherHash, "model hash mismatch");
    MeshInfo info;
    std::string err;
    CHECK(parse_mesh(bytes.data(), bytes.size(), modelHash, info, err));
}

static void test_mesh_payload_tamper() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    const size_t base = mesh_section_offset(bytes, "positions");
    bytes[base] ^= 0x01; // no reseal: the digest must catch this
    CHECK_MESH_REJECTS(bytes, modelHash, "payload hash mismatch");
}

static void test_mesh_truncated() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    bytes.resize(bytes.size() - 64);
    CHECK_MESH_REJECTS(bytes, modelHash, "fileBytes does not match");
    std::vector<uint8_t> stub(kSectionHeaderBytes - 1, 0);
    CHECK_MESH_REJECTS(stub, modelHash, "truncated header");
}

static void test_mesh_wrong_vertex_count() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash, make_cylinder(2, 100));
    CHECK(!bytes.empty());
    CHECK_MESH_REJECTS(bytes, modelHash, "vertex count is 200");
}

static void test_mesh_triangle_index_out_of_range() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    mesh_patch_u32(bytes, "triangles", 1, kClothVertexCount);
    CHECK_MESH_REJECTS(bytes, modelHash, "triangle index out of range");
}

static void test_mesh_degenerate_triangle() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    const size_t base = mesh_section_offset(bytes, "triangles");
    const uint32_t first = read_le32(bytes, base);
    mesh_patch_u32(bytes, "triangles", 1, first);
    CHECK_MESH_REJECTS(bytes, modelHash, "repeated corner");
}

// An edge shared by three triangles. write_mesh deliberately does not refuse
// this, because otherwise no test could reach the parser's manifoldness check.
static void test_mesh_non_manifold_edge() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto fixture = make_cylinder(2, kClothVertexCount / 2);
    const uint32_t a = fixture.triangles[0], b = fixture.triangles[1];
    fixture.triangles.insert(fixture.triangles.end(), { a, b, kClothVertexCount / 2 + 7 });
    auto bytes = make_mesh_bytes(modelHash, fixture);
    CHECK(!bytes.empty());
    CHECK_MESH_REJECTS(bytes, modelHash, "non-manifold edge shared by three");
}

static void test_mesh_no_pins() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto fixture = make_cylinder(2, kClothVertexCount / 2);
    std::fill(fixture.pins.begin(), fixture.pins.end(), 0u);
    auto bytes = make_mesh_bytes(modelHash, fixture);
    CHECK_MESH_REJECTS(bytes, modelHash, "no pinned vertices");
}

static void test_mesh_all_pinned() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto fixture = make_cylinder(2, kClothVertexCount / 2);
    std::fill(fixture.pins.begin(), fixture.pins.end(), 1u);
    auto bytes = make_mesh_bytes(modelHash, fixture);
    CHECK_MESH_REJECTS(bytes, modelHash, "every vertex is pinned");
}

static void test_mesh_nonfinite_position() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    mesh_patch_f32(bytes, "positions", 5, std::numeric_limits<float>::quiet_NaN());
    CHECK_MESH_REJECTS(bytes, modelHash, "nonfinite vertex position");
}

static void test_mesh_nonpositive_mass() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    mesh_patch_f32(bytes, "vertex_mass", 3, 0.0f);
    CHECK_MESH_REJECTS(bytes, modelHash, "mass must be positive");
}

static void test_mesh_pin_mask_not_boolean() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    mesh_patch_u32(bytes, "pin_mask", 0, 2);
    CHECK_MESH_REJECTS(bytes, modelHash, "pin_mask must be 0 or 1");
}

// Every derived count in `info` is re-derived by the parser, so a hand-edited
// header cannot make a mesh claim a structure it does not have.
static void test_mesh_info_disagrees_with_payload() {
    auto modelHash = compute_model_hash(make_model_bytes());
    struct Edit { size_t word; uint32_t value; const char* reason; };
    const Edit edits[] = {
        { 3, 7u,  "boundaryEdges disagrees" },
        { 4, 9u,  "boundaryLoops disagrees" },
        { 5, 8u,  "maxTriangleValence disagrees" },
        { 6, 11u, "pinned vertex count disagrees" },
        { 7, 1u,  "reserved info word is not zero" },
    };
    for (const Edit& edit : edits) {
        auto bytes = make_mesh_bytes(modelHash);
        mesh_patch_u32(bytes, "info", edit.word, edit.value);
        CHECK_MESH_REJECTS(bytes, modelHash, edit.reason);
    }
}

static void test_mesh_edges_not_ascending() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    const size_t base = mesh_section_offset(bytes, "edges");
    const uint32_t a0 = read_le32(bytes, base), b0 = read_le32(bytes, base + 4);
    const uint32_t a1 = read_le32(bytes, base + 8), b1 = read_le32(bytes, base + 12);
    write_le32(bytes, base, a1);
    write_le32(bytes, base + 4, b1);
    write_le32(bytes, base + 8, a0);
    write_le32(bytes, base + 12, b0);
    mesh_reseal(bytes);
    CHECK_MESH_REJECTS(bytes, modelHash, "not strictly ascending");
}

static void test_mesh_duplicate_section_name() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    // Section names live in the directory, not the payload, so no reseal is needed.
    const size_t second = kSectionHeaderBytes + kSectionEntryBytes;
    std::memset(bytes.data() + second, 0, kSectionNameBytes);
    std::memcpy(bytes.data() + second, "info", 4);
    CHECK_MESH_REJECTS(bytes, modelHash, "duplicate section name");
}

static void test_mesh_section_out_of_bounds() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    const size_t entry = kSectionHeaderBytes + kSectionEntryBytes;
    write_le32(bytes, entry + kSectionNameBytes + 8, 0xffffffu); // absurd count
    CHECK_MESH_REJECTS(bytes, modelHash, "extends past the end");
}

static void test_mesh_edge_csr_bad_neighbour() {
    auto modelHash = compute_model_hash(make_model_bytes());
    auto bytes = make_mesh_bytes(modelHash);
    // A neighbour that is in range but is not an edge of the mesh.
    mesh_patch_u32(bytes, "edge_csr_nbr", 0, kClothVertexCount - 1);
    CHECK_MESH_REJECTS(bytes, modelHash, "not in the edge list");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cerr << "=== SHA-256 tests ===\n";
    RUN(test_sha256_empty);
    RUN(test_sha256_abc);
    RUN(test_sha256_driver_names_single);
    RUN(test_sha256_driver_names_multiple);
    RUN(test_sha256_hex_lowercase);

    std::cerr << "\n=== Model parser tests ===\n";
    RUN(test_model_valid);
    RUN(test_model_truncated_header);
    RUN(test_model_truncated_json);
    RUN(test_model_bad_model_type);
    RUN(test_model_bad_vertices);
    RUN(test_model_bad_driver_feature_len);
    RUN(test_model_bad_driven_feature_len);
    RUN(test_model_bad_pca_dim);
    RUN(test_model_bad_driver_count);
    RUN(test_model_bad_first_driver);
    RUN(test_model_duplicate_driver);
    RUN(test_model_null_byte_in_json);
    RUN(test_model_empty_buffer);
    RUN(test_model_json_len_zero);

    std::cerr << "\n=== Clip parser tests ===\n";
    RUN(test_clip_valid);
    RUN(test_clip_truncated_header);
    RUN(test_clip_bad_magic);
    RUN(test_clip_bad_version);
    RUN(test_clip_bad_fps);
    RUN(test_clip_bad_driver_count);
    RUN(test_clip_bad_root_driver_index);
    RUN(test_clip_bad_model_hash);
    RUN(test_clip_bad_driver_list_hash);
    RUN(test_clip_payload_tamper);
    RUN(test_clip_truncated_payload);
    RUN(test_clip_nonfinite_floats);
    RUN(test_clip_nonfinite_component_fu);
    RUN(test_clip_nonfinite_pos_cm);
    RUN(test_clip_bad_header_bytes_field);
    RUN(test_clip_bad_local_float_count);
    RUN(test_clip_multi_frame);
    RUN(test_clip_zero_frames);

    std::cerr << "\n=== Mesh parser tests ===\n";
    RUN(test_mesh_valid);
    RUN(test_mesh_bad_magic);
    RUN(test_mesh_bad_version);
    RUN(test_mesh_bad_model_hash);
    RUN(test_mesh_payload_tamper);
    RUN(test_mesh_truncated);
    RUN(test_mesh_wrong_vertex_count);
    RUN(test_mesh_triangle_index_out_of_range);
    RUN(test_mesh_degenerate_triangle);
    RUN(test_mesh_non_manifold_edge);
    RUN(test_mesh_no_pins);
    RUN(test_mesh_all_pinned);
    RUN(test_mesh_nonfinite_position);
    RUN(test_mesh_nonpositive_mass);
    RUN(test_mesh_pin_mask_not_boolean);
    RUN(test_mesh_info_disagrees_with_payload);
    RUN(test_mesh_edges_not_ascending);
    RUN(test_mesh_duplicate_section_name);
    RUN(test_mesh_section_out_of_bounds);
    RUN(test_mesh_edge_csr_bad_neighbour);

    std::cerr << "\n=== Integration tests ===\n";
    RUN(test_model_clip_integration);

    std::cerr << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
