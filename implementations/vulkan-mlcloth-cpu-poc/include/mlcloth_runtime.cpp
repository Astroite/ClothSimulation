#include "mlcloth_runtime.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace mlcloth {
namespace {

std::runtime_error win32Error(const std::string& prefix) {
    return std::runtime_error(prefix + " (Win32 error " + std::to_string(GetLastError()) + ")");
}

HMODULE loadRequired(const std::filesystem::path& path) {
    const HMODULE module = LoadLibraryExW(path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    if (!module) throw win32Error("Could not load " + path.string());
    return module;
}

template <typename T>
T requiredExport(HMODULE module, const char* name) {
    const FARPROC address = GetProcAddress(module, name);
    if (!address) throw win32Error(std::string("AILab export is missing: ") + name);
    return reinterpret_cast<T>(address);
}

} // namespace

AILabRuntime::AILabRuntime(const std::filesystem::path& runtimeDirectory,
                           const std::uint8_t* modelPayload, std::size_t modelPayloadBytes,
                           int inputSize, int outputSize, int threads) {
    const std::filesystem::path absoluteRuntimeDirectory = std::filesystem::absolute(runtimeDirectory);
    if (!modelPayload || modelPayloadBytes == 0 || modelPayloadBytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("AILab model payload size is invalid");
    }
    if (inputSize != static_cast<int>(MLClothSequenceState::kInputSize) ||
        outputSize != static_cast<int>(MLClothSequenceState::kOutputSize)) {
        throw std::runtime_error("AILab model dimensions do not match the MLCloth vertex contract");
    }
    if (threads != 1 && threads != 2 && threads != 4 && threads != 8) {
        throw std::runtime_error("AILab thread count must be 1, 2, 4, or 8");
    }
    if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS)) {
        throw win32Error("SetDefaultDllDirectories failed");
    }
    dllDirectoryCookie_ = AddDllDirectory(absoluteRuntimeDirectory.c_str());
    if (!dllDirectoryCookie_) throw win32Error("AddDllDirectory failed for " + absoluteRuntimeDirectory.string());
    try {
        const wchar_t* names[] = { L"opencv_world440.dll", L"samplerate.dll", L"sent2pron.dll", L"MNN.dll", L"AILab.dll" };
        for (const wchar_t* name : names) modules_.push_back(loadRequired(absoluteRuntimeDirectory / name));
        const HMODULE ailab = static_cast<HMODULE>(modules_.back());
        create_ = requiredExport<CreateFn>(ailab, "createLinearInputModelFromBuffer");
        run_ = requiredExport<RunFn>(ailab, "runLinearInputModel");
        release_ = requiredExport<ReleaseFn>(ailab, "releaseLinearInputModel");
        isEmpty_ = requiredExport<IsEmptyFn>(ailab, "isEmptyLinearInputModel");
        const auto begin = std::chrono::steady_clock::now();
        model_ = create_("", "input0", "output0", inputSize, outputSize, 1, threads,
            const_cast<std::uint8_t*>(modelPayload), static_cast<int>(modelPayloadBytes));
        creationMilliseconds_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (!model_ || isEmpty_(model_)) throw std::runtime_error("AILab failed to synchronously create the MLCloth model");
    } catch (...) {
        if (model_ && release_) release_(model_);
        model_ = 0;
        for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) FreeLibrary(static_cast<HMODULE>(*it));
        modules_.clear();
        if (dllDirectoryCookie_) RemoveDllDirectory(dllDirectoryCookie_);
        dllDirectoryCookie_ = nullptr;
        throw;
    }
}

AILabRuntime::~AILabRuntime() {
    if (model_ && release_) release_(model_);
    for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) FreeLibrary(static_cast<HMODULE>(*it));
    if (dllDirectoryCookie_) RemoveDllDirectory(dllDirectoryCookie_);
}

void AILabRuntime::run(const float* input, float* output) const {
    if (!model_ || !run_ || !input || !output) throw std::runtime_error("AILab run called with an invalid model or buffer");
    run_(model_, input, output);
}

MLClothSequenceState::MLClothSequenceState()
    : input_(kInputSize), output_(kOutputSize), previousPositions_(kDriverCount * 3),
      previousPca_(kPcaDim), previousPca2_(kPcaDim) {}

void MLClothSequenceState::reset() {
    std::fill(input_.begin(), input_.end(), 0.0f);
    std::fill(output_.begin(), output_.end(), 0.0f);
    std::fill(previousPositions_.begin(), previousPositions_.end(), 0.0f);
    std::fill(previousPca_.begin(), previousPca_.end(), 0.0f);
    std::fill(previousPca2_.begin(), previousPca2_.end(), 0.0f);
    hasPreviousPosition_ = false;
}

double MLClothSequenceState::inferFrame(AILabRuntime& runtime, std::uint32_t frame,
                                        const std::vector<float>& localFu,
                                        const std::vector<float>& componentFu,
                                        const std::vector<float>& componentPositionsCm) {
    constexpr std::size_t rotationStride = kDriverCount * 6;
    constexpr std::size_t positionStride = kDriverCount * 3;
    const std::size_t rotationOffset = static_cast<std::size_t>(frame) * rotationStride;
    const std::size_t positionOffset = static_cast<std::size_t>(frame) * positionStride;
    if (rotationOffset + rotationStride > localFu.size() || rotationOffset + rotationStride > componentFu.size() ||
        positionOffset + positionStride > componentPositionsCm.size()) {
        throw std::runtime_error("MLCloth clip frame is outside the validated arrays");
    }
    std::fill(input_.begin(), input_.end(), 0.0f);
    constexpr std::size_t componentRotationInput = 6 * kDriverCount;
    constexpr std::size_t currentPositionInput = 15 * kDriverCount;
    constexpr std::size_t previousPositionInput = 18 * kDriverCount;
    constexpr std::size_t previousPcaInput = 21 * kDriverCount;
    constexpr std::size_t previousPca2Input = previousPcaInput + kPcaDim;
    std::copy_n(localFu.data() + rotationOffset, rotationStride, input_.data());
    std::copy_n(componentFu.data() + rotationOffset, rotationStride, input_.data() + componentRotationInput);
    std::copy_n(componentPositionsCm.data() + positionOffset, positionStride, input_.data() + currentPositionInput);
    const float* previous = hasPreviousPosition_ ? previousPositions_.data() : componentPositionsCm.data() + positionOffset;
    std::copy_n(previous, positionStride, input_.data() + previousPositionInput);
    std::copy(previousPca_.begin(), previousPca_.end(), input_.begin() + previousPcaInput);
    std::copy(previousPca2_.begin(), previousPca2_.end(), input_.begin() + previousPca2Input);

    const auto begin = std::chrono::steady_clock::now();
    runtime.run(input_.data(), output_.data());
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (!std::all_of(output_.begin(), output_.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("AILab produced NaN or Inf");
    }
    previousPca2_ = previousPca_;
    std::copy_n(output_.data() + kVertexCount * 3, kPcaDim, previousPca_.data());
    std::copy_n(componentPositionsCm.data() + positionOffset, positionStride, previousPositions_.data());
    hasPreviousPosition_ = true;
    return elapsed;
}

std::uint64_t MLClothSequenceState::outputHash64() const noexcept {
    constexpr std::uint64_t offset = 1469598103934665603ull;
    constexpr std::uint64_t prime = 1099511628211ull;
    std::uint64_t hash = offset;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(output_.data());
    for (std::size_t i = 0; i < output_.size() * sizeof(float); ++i) { hash ^= bytes[i]; hash *= prime; }
    return hash;
}

} // namespace mlcloth
