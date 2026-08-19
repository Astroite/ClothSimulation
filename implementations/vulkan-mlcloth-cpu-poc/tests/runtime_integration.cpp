#include "mlcloth_formats.h"
#include "mlcloth_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const auto length = stream.tellg();
    if (length <= 0) throw std::runtime_error("empty file " + path.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(bytes.data()), length);
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    return bytes;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t exerciseRuntime(const std::filesystem::path& runtimeDir,
                              const std::vector<std::uint8_t>& modelBytes,
                              const mlcloth::ModelInfo& model,
                              const mlcloth::ClipInfo& clip,
                              const std::vector<float>& local,
                              const std::vector<float>& component,
                              const std::vector<float>& positions) {
    mlcloth::AILabRuntime runtime(runtimeDir, modelBytes.data() + model.payloadOffset, model.payloadLen,
        model.driverFeatureLen, model.drivenFeatureLen, 1);
    mlcloth::MLClothSequenceState state;
    state.reset();
    state.inferFrame(runtime, 0, local, component, positions);
    const std::uint64_t firstHash = state.outputHash64();
    require(firstHash != 0, "first output hash is zero");
    constexpr std::size_t n = mlcloth::MLClothSequenceState::kDriverCount;
    constexpr std::size_t rotationStride = n * 6;
    constexpr std::size_t positionStride = n * 3;
    constexpr std::size_t unusedBegin = n * 12;
    constexpr std::size_t currentPosition = n * 15;
    constexpr std::size_t previousPosition = n * 18;
    constexpr std::size_t previousPca = n * 21;
    constexpr std::size_t previousPca2 = previousPca + mlcloth::MLClothSequenceState::kPcaDim;
    auto requireEqual = [](const std::vector<float>& input, std::size_t inputOffset,
                           const std::vector<float>& expected, std::size_t expectedOffset,
                           std::size_t count, const char* message) {
        require(std::equal(input.begin() + inputOffset, input.begin() + inputOffset + count,
                           expected.begin() + expectedOffset), message);
    };
    const auto& firstInput = state.input();
    requireEqual(firstInput, 0, local, 0, rotationStride, "frame 0 local rotation input mismatch");
    requireEqual(firstInput, n * 6, component, 0, rotationStride, "frame 0 component rotation input mismatch");
    require(std::all_of(firstInput.begin() + unusedBegin, firstInput.begin() + currentPosition,
                        [](float value) { return value == 0.0f; }), "12D..15D reserved input slots are not zero");
    requireEqual(firstInput, currentPosition, positions, 0, positionStride, "frame 0 current position input mismatch");
    requireEqual(firstInput, previousPosition, positions, 0, positionStride, "frame 0 previous position must equal current");
    require(std::all_of(firstInput.begin() + previousPca, firstInput.end(),
                        [](float value) { return value == 0.0f; }), "frame 0 PCA history inputs are not zero");
    std::vector<float> firstOutputPca(state.output().begin() + mlcloth::MLClothSequenceState::kVertexCount * 3,
                                      state.output().end());
    if (clip.header.frameCount >= 2) {
        state.inferFrame(runtime, 1, local, component, positions);
        const auto& secondInput = state.input();
        requireEqual(secondInput, currentPosition, positions, positionStride, positionStride,
                     "frame 1 current position input mismatch");
        requireEqual(secondInput, previousPosition, positions, 0, positionStride,
                     "frame 1 previous position did not roll from frame 0");
        requireEqual(secondInput, previousPca, firstOutputPca, 0, firstOutputPca.size(),
                     "PrevPca did not receive frame 0 output PCA");
        require(std::all_of(secondInput.begin() + previousPca2, secondInput.end(),
                            [](float value) { return value == 0.0f; }), "PrevPca2 must remain zero on frame 1");
    }
    for (int replay = 0; replay < 3; ++replay) {
        state.reset();
        state.inferFrame(runtime, 0, local, component, positions);
        require(state.outputHash64() == firstHash, "repeated reset is not deterministic");
    }

    state.reset();
    float minimum = std::numeric_limits<float>::max();
    float maximum = std::numeric_limits<float>::lowest();
    for (std::uint32_t frame = 0; frame < clip.header.frameCount; ++frame) {
        state.inferFrame(runtime, frame, local, component, positions);
        const auto& output = state.output();
        for (std::size_t i = 0; i < mlcloth::MLClothSequenceState::kVertexCount * 3; ++i) {
            require(std::isfinite(output[i]), "full clip contains NaN or Inf");
            minimum = std::min(minimum, output[i]);
            maximum = std::max(maximum, output[i]);
        }
    }
    require(maximum - minimum > 1.0e-5f, "full-clip output AABB is degenerate");
    state.reset();
    state.inferFrame(runtime, 0, local, component, positions);
    require(state.outputHash64() == firstHash, "loop reset first frame differs");
    return firstHash;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: runtime_integration <runtime-dir> <model.enc> <clip.mldrv>");
        const std::filesystem::path runtimeDir = argv[1];
        const auto modelBytes = readFile(argv[2]);
        const auto clipBytes = readFile(argv[3]);
        mlcloth::ModelInfo model;
        std::string error;
        require(mlcloth::parse_model(modelBytes.data(), modelBytes.size(), model, error), "model: " + error);
        mlcloth::ClipInfo clip;
        const auto modelHash = mlcloth::sha256(modelBytes);
        const auto driverHash = mlcloth::sha256_driver_names(model.driverNames);
        require(mlcloth::parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, clip, error), "clip: " + error);
        std::vector<float> local(clip.localFu, clip.localFu + clip.header.localFloatCount);
        std::vector<float> component(clip.componentFu, clip.componentFu + clip.header.componentFloatCount);
        std::vector<float> positions(clip.componentPosCm, clip.componentPosCm + clip.header.positionFloatCount);
        const auto first = exerciseRuntime(runtimeDir, modelBytes, model, clip, local, component, positions);
        const auto second = exerciseRuntime(runtimeDir, modelBytes, model, clip, local, component, positions);
        require(first == second, "release/recreate changed first output");
        std::cout << "AILab integration passed: " << clip.header.frameCount << " frames, first hash 0x"
                  << std::hex << first << std::dec << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "AILab integration FAILED: " << exception.what() << '\n';
        return 1;
    }
}
