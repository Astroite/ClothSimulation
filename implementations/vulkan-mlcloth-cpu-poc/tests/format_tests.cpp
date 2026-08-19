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

    std::cerr << "\n=== Integration tests ===\n";
    RUN(test_model_clip_integration);

    std::cerr << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
