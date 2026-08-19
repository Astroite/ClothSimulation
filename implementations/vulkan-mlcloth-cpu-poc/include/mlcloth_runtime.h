#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace mlcloth {

class AILabRuntime final {
public:
    AILabRuntime(const std::filesystem::path& runtimeDirectory,
                 const std::uint8_t* modelPayload, std::size_t modelPayloadBytes,
                 int inputSize, int outputSize, int threads);
    ~AILabRuntime();
    AILabRuntime(const AILabRuntime&) = delete;
    AILabRuntime& operator=(const AILabRuntime&) = delete;
    void run(const float* input, float* output) const;
    double creationMilliseconds() const noexcept { return creationMilliseconds_; }

private:
    using CreateFn = unsigned long long (*)(const char*, const char*, const char*, int, int, int, int, std::uint8_t*, int);
    using RunFn = void (*)(unsigned long long, const float*, float*);
    using ReleaseFn = void (*)(unsigned long long);
    using IsEmptyFn = bool (*)(unsigned long long);
    std::vector<void*> modules_;
    void* dllDirectoryCookie_{};
    CreateFn create_{};
    RunFn run_{};
    ReleaseFn release_{};
    IsEmptyFn isEmpty_{};
    unsigned long long model_{};
    double creationMilliseconds_{};
};

class MLClothSequenceState final {
public:
    static constexpr std::uint32_t kDriverCount = 45;
    static constexpr std::uint32_t kPcaDim = 512;
    static constexpr std::uint32_t kVertexCount = 5294;
    static constexpr std::uint32_t kInputSize = 1969;
    static constexpr std::uint32_t kOutputSize = 16394;

    MLClothSequenceState();
    void reset();
    double inferFrame(AILabRuntime& runtime, std::uint32_t frame,
                      const std::vector<float>& localFu,
                      const std::vector<float>& componentFu,
                      const std::vector<float>& componentPositionsCm);
    const std::vector<float>& output() const noexcept { return output_; }
    const std::vector<float>& input() const noexcept { return input_; }
    std::uint64_t outputHash64() const noexcept;

private:
    std::vector<float> input_;
    std::vector<float> output_;
    std::vector<float> previousPositions_;
    std::vector<float> previousPca_;
    std::vector<float> previousPca2_;
    bool hasPreviousPosition_{};
};

} // namespace mlcloth

