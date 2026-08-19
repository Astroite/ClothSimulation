/*
 * AILab/MNN CPU inference -> Vulkan upload -> compute transform -> point cloud.
 * This validation sample intentionally contains no XPBD or cloth topology.
 */
#include "vulkanexamplebase.h"
#include "mlcloth_formats.h"
#include "mlcloth_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class VulkanExample : public VulkanExampleBase {
public:
    static constexpr uint32_t kVertexCount = mlcloth::MLClothSequenceState::kVertexCount;
    static constexpr VkDeviceSize kPointBytes = static_cast<VkDeviceSize>(kVertexCount) * sizeof(glm::vec4);
    static constexpr uint32_t kBenchmarkWarmup = 200;
    static constexpr uint32_t kBenchmarkSamples = 1000;

    struct FrameResources {
        vks::Buffer upload;
        vks::Buffer points;
        vks::Buffer transformUniform;
        vks::Buffer cameraUniform;
        VkDescriptorSet computeSet{ VK_NULL_HANDLE };
        VkDescriptorSet graphicsSet{ VK_NULL_HANDLE };
        VkQueryPool queryPool{ VK_NULL_HANDLE };
        bool queryIssued{};
    };
    std::array<FrameResources, maxConcurrentFrames> frames{};

    struct alignas(16) TransformUniform {
        glm::vec4 rootPositionAndCount{};
        glm::vec4 rootUp{};
        glm::vec4 rootRight{};
    } transformUniform;
    static_assert(sizeof(TransformUniform) == 48);

    struct CameraUniform {
        glm::mat4 projection{};
        glm::mat4 view{};
    } cameraUniform;

    VkDescriptorSetLayout computeSetLayout{ VK_NULL_HANDLE };
    VkDescriptorSetLayout graphicsSetLayout{ VK_NULL_HANDLE };
    VkPipelineLayout computePipelineLayout{ VK_NULL_HANDLE };
    VkPipelineLayout graphicsPipelineLayout{ VK_NULL_HANDLE };
    VkPipeline computePipeline{ VK_NULL_HANDLE };
    VkPipeline graphicsPipeline{ VK_NULL_HANDLE };

    std::filesystem::path runtimeDirectory;
    std::filesystem::path modelPath;
    std::filesystem::path clipPath;
    std::filesystem::path benchmarkPath{ "mlcloth_benchmark.csv" };
    int threads{ 1 };
    bool verifyMode{};
    bool benchmarkMode{};
    bool simulationPaused{};
    bool resetRequested{};
    bool firstRender{ true };
    bool gpuVerificationDone{};
    bool benchmarkWritten{};
    uint32_t requestedFrames{};
    uint32_t clipFrame{};
    uint64_t simulationSteps{};
    uint64_t droppedSteps{};
    double accumulatorSeconds{};
    double lastInferenceMs{};
    double lastInputBuildMs{};
    double lastTotalStepMs{};
    double lastGpuTransformMs{};
    std::string errorStatus{ "OK" };

    std::vector<uint8_t> modelBytes;
    std::vector<uint8_t> clipBytes;
    mlcloth::ModelInfo modelInfo{};
    mlcloth::ClipInfo clipInfo{};
    std::vector<float> localFu;
    std::vector<float> componentFu;
    std::vector<float> componentPositionsCm;
    std::unique_ptr<mlcloth::AILabRuntime> runtime;
    mlcloth::MLClothSequenceState sequence;
    std::vector<glm::vec4> latestLocalPoints;
    std::vector<glm::vec4> latestWorldPoints;
    uint64_t firstFrameHash{};

    std::vector<double> inferenceSamples;
    std::vector<double> recentInferenceSamples;
    std::vector<double> inputSamples;
    std::vector<double> totalStepSamples;
    std::vector<double> gpuSamples;
    uint64_t benchmarkCpuSteps{};
    uint64_t benchmarkGpuFrames{};

    static bool hasArgument(const char* name) {
        return std::find_if(args.begin(), args.end(), [name](const char* value) { return std::strcmp(value, name) == 0; }) != args.end();
    }

    static std::string argumentValue(const char* name, const std::string& fallback) {
        for (size_t i = 0; i + 1 < args.size(); ++i) if (std::strcmp(args[i], name) == 0) return args[i + 1];
        return fallback;
    }

    VulkanExample() : VulkanExampleBase() {
        runtimeDirectory = argumentValue("--runtime-dir", "");
        modelPath = argumentValue("--model", "");
        clipPath = argumentValue("--clip", "");
        benchmarkPath = argumentValue("--benchmark-output", "mlcloth_benchmark.csv");
        threads = std::stoi(argumentValue("--threads", "1"));
        requestedFrames = static_cast<uint32_t>(std::stoul(argumentValue("--frames", "0")));
        verifyMode = hasArgument("--verify");
        benchmarkMode = hasArgument("--benchmark");
        if (runtimeDirectory.empty() || modelPath.empty() || clipPath.empty()) {
            throw std::runtime_error("--runtime-dir, --model and --clip are required");
        }
        if (hasArgument("--sync-validation")) {
            static const char* synchronizationValidation = "VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT";
            VkLayerSettingEXT setting{};
            setting.pLayerName = "VK_LAYER_KHRONOS_validation";
            setting.pSettingName = "enables";
            setting.type = VK_LAYER_SETTING_TYPE_STRING_EXT;
            setting.valueCount = 1;
            setting.pValues = &synchronizationValidation;
            enabledLayerSettings.push_back(setting);
        }
        if (benchmarkMode || verifyMode || requestedFrames > 0) {
            benchmark.active = true;
            benchmark.warmup = 0;
            benchmark.duration = 600;
            benchmark.outputFrames = static_cast<int32_t>(requestedFrames > 0 ? requestedFrames
                : (benchmarkMode ? kBenchmarkWarmup + kBenchmarkSamples + maxConcurrentFrames + 12 : 8));
            settings.overlay = false;
            vks::tools::errorModeSilent = true;
#if defined(_WIN32)
            setupConsole(benchmarkMode ? "MLCloth CPU/Vulkan benchmark" : "MLCloth CPU/Vulkan verification");
#endif
        }
        title = "MLCloth AILab CPU -> Vulkan point cloud";
        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(55.0f, static_cast<float>(width) / static_cast<float>(height), 0.01f, 1000.0f);
        camera.setRotation(glm::vec3(-8.0f, -18.0f, 0.0f));
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -4.0f));
        camera.setRotationSpeed(0.35f);
        camera.setMovementSpeed(2.0f);
    }

    ~VulkanExample() override {
        if (!device) return;
        vkDeviceWaitIdle(device);
        if (benchmarkMode && !benchmarkWritten) writeBenchmarkCsv();
        for (auto& frame : frames) {
            frame.upload.destroy();
            frame.points.destroy();
            frame.transformUniform.destroy();
            frame.cameraUniform.destroy();
            if (frame.queryPool) vkDestroyQueryPool(device, frame.queryPool, nullptr);
        }
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, computeSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, graphicsSetLayout, nullptr);
    }

    static std::vector<uint8_t> readFile(const std::filesystem::path& path) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) throw std::runtime_error("Could not open " + path.string());
        const std::streamoff length = stream.tellg();
        if (length <= 0) throw std::runtime_error("File is empty: " + path.string());
        std::vector<uint8_t> bytes(static_cast<size_t>(length));
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(bytes.data()), length);
        if (!stream) throw std::runtime_error("Could not read " + path.string());
        return bytes;
    }

    void loadModelClipAndRuntime() {
        modelBytes = readFile(modelPath);
        std::string parseError;
        if (!mlcloth::parse_model(modelBytes.data(), modelBytes.size(), modelInfo, parseError)) {
            throw std::runtime_error("Encoded model validation failed: " + parseError);
        }
        const auto modelHash = mlcloth::sha256(modelBytes);
        const auto driverHash = mlcloth::sha256_driver_names(modelInfo.driverNames);
        clipBytes = readFile(clipPath);
        if (!mlcloth::parse_clip(clipBytes.data(), clipBytes.size(), modelHash, driverHash, clipInfo, parseError)) {
            throw std::runtime_error("MLDRV001 validation failed: " + parseError);
        }
        localFu.assign(clipInfo.localFu, clipInfo.localFu + clipInfo.header.localFloatCount);
        componentFu.assign(clipInfo.componentFu, clipInfo.componentFu + clipInfo.header.componentFloatCount);
        componentPositionsCm.assign(clipInfo.componentPosCm, clipInfo.componentPosCm + clipInfo.header.positionFloatCount);
        runtime = std::make_unique<mlcloth::AILabRuntime>(runtimeDirectory,
            modelBytes.data() + modelInfo.payloadOffset, modelInfo.payloadLen,
            modelInfo.driverFeatureLen, modelInfo.drivenFeatureLen, threads);
        latestLocalPoints.resize(kVertexCount);
        latestWorldPoints.resize(kVertexCount);
        std::cout << "AILab model preloaded in " << runtime->creationMilliseconds() << " ms; clip="
                  << clipInfo.header.frameCount << " frames, drivers=" << clipInfo.header.driverCount << "\n";
    }

    glm::vec3 rootUp(uint32_t frame) const {
        const size_t offset = static_cast<size_t>(frame) * mlcloth::kDriverCount * 6;
        return glm::vec3(componentFu[offset], componentFu[offset + 1], componentFu[offset + 2]);
    }

    glm::vec3 rootRight(uint32_t frame) const {
        const size_t offset = static_cast<size_t>(frame) * mlcloth::kDriverCount * 6 + 3;
        return glm::vec3(componentFu[offset], componentFu[offset + 1], componentFu[offset + 2]);
    }

    glm::vec3 rootPosition(uint32_t frame) const {
        const size_t offset = static_cast<size_t>(frame) * mlcloth::kDriverCount * 3;
        return glm::vec3(componentPositionsCm[offset], componentPositionsCm[offset + 1], componentPositionsCm[offset + 2]);
    }

    void buildPointData(uint32_t frame) {
        const auto& output = sequence.output();
        glm::vec3 up = glm::normalize(rootUp(frame));
        glm::vec3 right = glm::normalize(rootRight(frame));
        glm::vec3 forward = glm::normalize(glm::cross(right, up));
        const glm::vec3 root = rootPosition(frame);
        glm::vec3 minimum(std::numeric_limits<float>::max());
        glm::vec3 maximum(std::numeric_limits<float>::lowest());
        for (uint32_t vertex = 0; vertex < kVertexCount; ++vertex) {
            const glm::vec3 local(output[vertex * 3], output[vertex * 3 + 1], output[vertex * 3 + 2]);
            latestLocalPoints[vertex] = glm::vec4(local, 1.0f);
            const glm::vec3 component = root + forward * local.x + right * local.y + up * local.z;
            const glm::vec3 world(component.x, component.z, -component.y);
            const glm::vec3 metres = world * 0.01f;
            if (!std::isfinite(metres.x) || !std::isfinite(metres.y) || !std::isfinite(metres.z)) {
                throw std::runtime_error("CPU coordinate reference contains NaN or Inf");
            }
            latestWorldPoints[vertex] = glm::vec4(metres, 1.0f);
            minimum = glm::min(minimum, metres);
            maximum = glm::max(maximum, metres);
        }
        const glm::vec3 extent = maximum - minimum;
        if (std::max({ extent.x, extent.y, extent.z }) <= 1.0e-6f) throw std::runtime_error("Predicted cloth AABB is degenerate");
        transformUniform.rootPositionAndCount = glm::vec4(root, 0.0f);
        std::memcpy(&transformUniform.rootPositionAndCount.w, &kVertexCount, sizeof(kVertexCount));
        transformUniform.rootUp = glm::vec4(up, 0.0f);
        transformUniform.rootRight = glm::vec4(right, 0.0f);
    }

    void simulateOneStep(bool collectBenchmark) {
        const uint32_t frame = clipFrame;
        const auto begin = std::chrono::steady_clock::now();
        lastInferenceMs = sequence.inferFrame(*runtime, frame, localFu, componentFu, componentPositionsCm);
        const auto afterInference = std::chrono::steady_clock::now();
        buildPointData(frame);
        const auto end = std::chrono::steady_clock::now();
        const double throughInference = std::chrono::duration<double, std::milli>(afterInference - begin).count();
        lastInputBuildMs = std::max(0.0, throughInference - lastInferenceMs);
        lastTotalStepMs = std::chrono::duration<double, std::milli>(end - begin).count();
        recentInferenceSamples.push_back(lastInferenceMs);
        if (recentInferenceSamples.size() > 120) recentInferenceSamples.erase(recentInferenceSamples.begin());
        ++simulationSteps;
        if (simulationSteps == 1) firstFrameHash = sequence.outputHash64();
        if (collectBenchmark && benchmarkMode) {
            if (benchmarkCpuSteps >= kBenchmarkWarmup && inferenceSamples.size() < kBenchmarkSamples) {
                inferenceSamples.push_back(lastInferenceMs);
                inputSamples.push_back(lastInputBuildMs);
                totalStepSamples.push_back(lastTotalStepMs);
            }
            ++benchmarkCpuSteps;
        }
        ++clipFrame;
        if (clipFrame == clipInfo.header.frameCount) {
            clipFrame = 0;
            sequence.reset();
        }
    }

    void resetSequence() {
        sequence.reset();
        clipFrame = 0;
        accumulatorSeconds = 0.0;
        simulationSteps = 0;
        firstFrameHash = 0;
        simulateOneStep(false);
    }

    void runCpuVerification() {
        sequence.reset();
        clipFrame = 0;
        simulationSteps = 0;
        uint64_t baselineFirstHash = 0;
        for (uint32_t frame = 0; frame < clipInfo.header.frameCount; ++frame) {
            simulateOneStep(false);
            if (frame == 0) baselineFirstHash = sequence.outputHash64();
        }
        if (clipFrame != 0) throw std::runtime_error("Clip loop did not reset sequence state");
        simulateOneStep(false);
        if (sequence.outputHash64() != baselineFirstHash) throw std::runtime_error("CPU reset replay first-frame hash differs");
        std::cout << "CPU full-clip verification passed; first-frame hash=0x" << std::hex << baselineFirstHash << std::dec << "\n";
        resetSequence();
    }

    void autoFrameCamera() {
        glm::vec3 minimum(std::numeric_limits<float>::max());
        glm::vec3 maximum(std::numeric_limits<float>::lowest());
        for (const glm::vec4& point : latestWorldPoints) {
            minimum = glm::min(minimum, glm::vec3(point));
            maximum = glm::max(maximum, glm::vec3(point));
        }
        const glm::vec3 center = (minimum + maximum) * 0.5f;
        const float radius = std::max(0.25f, glm::length(maximum - minimum) * 0.5f);
        camera.setTranslation(glm::vec3(-center.x, -center.y, -center.z - radius * 2.8f));
    }

    void prepareBuffers() {
        const VkQueueFamilyProperties& queueProperties = vulkanDevice->queueFamilyProperties[vulkanDevice->queueFamilyIndices.graphics];
        const bool timestampUsable = queueProperties.timestampValidBits > 0 && deviceProperties.limits.timestampPeriod > 0.0f;
        for (auto& frame : frames) {
            VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &frame.upload, kPointBytes));
            VK_CHECK_RESULT(frame.upload.map());
            VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &frame.points, kPointBytes));
            VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &frame.transformUniform, sizeof(TransformUniform)));
            VK_CHECK_RESULT(frame.transformUniform.map());
            VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &frame.cameraUniform, sizeof(CameraUniform)));
            VK_CHECK_RESULT(frame.cameraUniform.map());
            if (timestampUsable) {
                VkQueryPoolCreateInfo queryInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
                queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
                queryInfo.queryCount = 2;
                VK_CHECK_RESULT(vkCreateQueryPool(device, &queryInfo, nullptr, &frame.queryPool));
            }
        }
    }

    void prepareDescriptors() {
        std::vector<VkDescriptorPoolSize> sizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxConcurrentFrames * 2),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxConcurrentFrames * 2),
        };
        VkDescriptorPoolCreateInfo poolInfo = vks::initializers::descriptorPoolCreateInfo(sizes, maxConcurrentFrames * 2);
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
        std::vector<VkDescriptorSetLayoutBinding> computeBindings = {
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
        };
        VkDescriptorSetLayoutCreateInfo computeLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(computeBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &computeLayoutInfo, nullptr, &computeSetLayout));
        const auto graphicsBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
        VkDescriptorSetLayoutCreateInfo graphicsLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(&graphicsBinding, 1);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &graphicsLayoutInfo, nullptr, &graphicsSetLayout));
        for (auto& frame : frames) {
            VkDescriptorSetAllocateInfo allocate = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &computeSetLayout, 1);
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocate, &frame.computeSet));
            std::array<VkWriteDescriptorSet, 3> writes = {
                vks::initializers::writeDescriptorSet(frame.computeSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &frame.upload.descriptor),
                vks::initializers::writeDescriptorSet(frame.computeSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &frame.points.descriptor),
                vks::initializers::writeDescriptorSet(frame.computeSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2, &frame.transformUniform.descriptor),
            };
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            allocate = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &graphicsSetLayout, 1);
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocate, &frame.graphicsSet));
            const auto graphicsWrite = vks::initializers::writeDescriptorSet(frame.graphicsSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &frame.cameraUniform.descriptor);
            vkUpdateDescriptorSets(device, 1, &graphicsWrite, 0, nullptr);
        }
    }

    void preparePipelines() {
        VkPipelineLayoutCreateInfo computeLayoutInfo = vks::initializers::pipelineLayoutCreateInfo(&computeSetLayout, 1);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &computeLayoutInfo, nullptr, &computePipelineLayout));
        VkComputePipelineCreateInfo computeInfo = vks::initializers::computePipelineCreateInfo(computePipelineLayout);
        computeInfo.stage = loadShader(getShadersPath() + "mlclothcpu/point_transform.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
        VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computeInfo, nullptr, &computePipeline));

        VkPipelineLayoutCreateInfo graphicsLayoutInfo = vks::initializers::pipelineLayoutCreateInfo(&graphicsSetLayout, 1);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &graphicsLayoutInfo, nullptr, &graphicsPipelineLayout));
        VkPipelineInputAssemblyStateCreateInfo inputAssembly = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 0, VK_FALSE);
        VkPipelineRasterizationStateCreateInfo rasterization = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
        VkPipelineColorBlendAttachmentState blendAttachment = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        VkPipelineColorBlendStateCreateInfo blend = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachment);
        VkPipelineDepthStencilStateCreateInfo depth = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
        VkPipelineViewportStateCreateInfo viewport = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
        VkPipelineMultisampleStateCreateInfo multisample = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
        std::vector<VkDynamicState> states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamic = vks::initializers::pipelineDynamicStateCreateInfo(states);
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
            loadShader(getShadersPath() + "mlclothcpu/point.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader(getShadersPath() + "mlclothcpu/point.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
        };
        const auto binding = vks::initializers::vertexInputBindingDescription(0, sizeof(glm::vec4), VK_VERTEX_INPUT_RATE_VERTEX);
        const auto attribute = vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0);
        VkPipelineVertexInputStateCreateInfo vertexInput = vks::initializers::pipelineVertexInputStateCreateInfo();
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount = 1;
        vertexInput.pVertexAttributeDescriptions = &attribute;
        VkGraphicsPipelineCreateInfo pipeline = vks::initializers::pipelineCreateInfo(graphicsPipelineLayout, renderPass);
        pipeline.pVertexInputState = &vertexInput;
        pipeline.pInputAssemblyState = &inputAssembly;
        pipeline.pRasterizationState = &rasterization;
        pipeline.pColorBlendState = &blend;
        pipeline.pMultisampleState = &multisample;
        pipeline.pViewportState = &viewport;
        pipeline.pDepthStencilState = &depth;
        pipeline.pDynamicState = &dynamic;
        pipeline.stageCount = static_cast<uint32_t>(stages.size());
        pipeline.pStages = stages.data();
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipeline, nullptr, &graphicsPipeline));
    }

    void prepare() override {
        try {
            loadModelClipAndRuntime();
            if (verifyMode) runCpuVerification(); else resetSequence();
            autoFrameCamera();
            VulkanExampleBase::prepare();
            prepareBuffers();
            prepareDescriptors();
            preparePipelines();
            prepared = true;
        } catch (const std::exception& exception) {
            errorStatus = exception.what();
            vks::tools::exitFatal(exception.what(), -1);
        }
    }

    void collectGpuTimestamp(FrameResources& frame) {
        if (!frame.queryPool || !frame.queryIssued) return;
        uint64_t values[2]{};
        const VkResult result = vkGetQueryPoolResults(device, frame.queryPool, 0, 2, sizeof(values), values,
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        if (result != VK_SUCCESS) return;
        lastGpuTransformMs = static_cast<double>(values[1] - values[0]) * static_cast<double>(deviceProperties.limits.timestampPeriod) / 1.0e6;
        if (benchmarkMode) {
            if (benchmarkGpuFrames >= kBenchmarkWarmup && gpuSamples.size() < kBenchmarkSamples) gpuSamples.push_back(lastGpuTransformMs);
            ++benchmarkGpuFrames;
        }
    }

    void advanceSimulation() {
        if (resetRequested) { resetRequested = false; resetSequence(); }
        if (simulationPaused) return;
        if (firstRender) { firstRender = false; return; }
        if (benchmarkMode) { simulateOneStep(true); return; }
        accumulatorSeconds += static_cast<double>(frameTimer);
        constexpr double step = 1.0 / 30.0;
        uint32_t catches = 0;
        while (accumulatorSeconds >= step && catches < 4) {
            simulateOneStep(false);
            accumulatorSeconds -= step;
            ++catches;
        }
        if (accumulatorSeconds >= step) {
            const uint64_t dropped = static_cast<uint64_t>(accumulatorSeconds / step);
            droppedSteps += dropped;
            accumulatorSeconds -= static_cast<double>(dropped) * step;
        }
    }

    void updateMappedBuffers(FrameResources& frame) {
        std::memcpy(frame.upload.mapped, latestLocalPoints.data(), static_cast<size_t>(kPointBytes));
        std::memcpy(frame.transformUniform.mapped, &transformUniform, sizeof(transformUniform));
        cameraUniform.projection = camera.matrices.perspective;
        cameraUniform.view = camera.matrices.view;
        std::memcpy(frame.cameraUniform.mapped, &cameraUniform, sizeof(cameraUniform));
    }

    void buildCommandBuffer(FrameResources& frame) {
        VkCommandBuffer command = drawCmdBuffers[currentBuffer];
        VkCommandBufferBeginInfo begin = vks::initializers::commandBufferBeginInfo();
        VK_CHECK_RESULT(vkBeginCommandBuffer(command, &begin));
        if (frame.queryPool) {
            vkCmdResetQueryPool(command, frame.queryPool, 0, 2);
            vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame.queryPool, 0);
        }
        VkBufferMemoryBarrier uploadBarrier = vks::initializers::bufferMemoryBarrier();
        uploadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        uploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        uploadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        uploadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        uploadBarrier.buffer = frame.upload.buffer;
        uploadBarrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, nullptr, 1, &uploadBarrier, 0, nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &frame.computeSet, 0, nullptr);
        vkCmdDispatch(command, (kVertexCount + 127) / 128, 1, 1);
        if (frame.queryPool) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, frame.queryPool, 1);
        VkBufferMemoryBarrier pointBarrier = vks::initializers::bufferMemoryBarrier();
        pointBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        pointBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        pointBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pointBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pointBarrier.buffer = frame.points.buffer;
        pointBarrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0,
            0, nullptr, 1, &pointBarrier, 0, nullptr);

        VkClearValue clear[2]{};
        clear[0].color = { { 0.012f, 0.018f, 0.030f, 1.0f } };
        clear[1].depthStencil = { 1.0f, 0 };
        VkRenderPassBeginInfo renderBegin = vks::initializers::renderPassBeginInfo();
        renderBegin.renderPass = renderPass;
        renderBegin.framebuffer = frameBuffers[currentImageIndex];
        renderBegin.renderArea.extent = { width, height };
        renderBegin.clearValueCount = 2;
        renderBegin.pClearValues = clear;
        vkCmdBeginRenderPass(command, &renderBegin, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport viewport = vks::initializers::viewport(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
        VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &frame.graphicsSet, 0, nullptr);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command, 0, 1, &frame.points.buffer, &offset);
        vkCmdDraw(command, kVertexCount, 1, 0, 0);
        drawUI(command);
        vkCmdEndRenderPass(command);
        VK_CHECK_RESULT(vkEndCommandBuffer(command));
        frame.queryIssued = frame.queryPool != VK_NULL_HANDLE;
    }

    void verifyGpuReadback(FrameResources& frame) {
        if (!verifyMode || gpuVerificationDone) return;
        vks::Buffer readback;
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &readback, kPointBytes));
        VkCommandBufferAllocateInfo allocate = vks::initializers::commandBufferAllocateInfo(cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
        VkCommandBuffer command{};
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &allocate, &command));
        VkCommandBufferBeginInfo begin = vks::initializers::commandBufferBeginInfo();
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK_RESULT(vkBeginCommandBuffer(command, &begin));
        VkBufferMemoryBarrier barrier = vks::initializers::bufferMemoryBarrier();
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = frame.points.buffer;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        VkBufferCopy copy{ 0, 0, kPointBytes };
        vkCmdCopyBuffer(command, frame.points.buffer, readback.buffer, 1, &copy);
        VK_CHECK_RESULT(vkEndCommandBuffer(command));
        VkSubmitInfo submit = vks::initializers::submitInfo();
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        VK_CHECK_RESULT(vkQueueWaitIdle(queue));
        VK_CHECK_RESULT(readback.map());
        const auto* actual = static_cast<const glm::vec4*>(readback.mapped);
        double maxError = 0.0;
        for (uint32_t vertex = 0; vertex < kVertexCount; ++vertex) {
            for (uint32_t axis = 0; axis < 3; ++axis) {
                maxError = std::max(maxError, std::abs(static_cast<double>(actual[vertex][axis]) - latestWorldPoints[vertex][axis]));
            }
        }
        readback.unmap();
        readback.destroy();
        vkFreeCommandBuffers(device, cmdPool, 1, &command);
        std::ofstream report("mlcloth_verify.json");
        report << std::setprecision(10) << "{\n  \"vertices\": " << kVertexCount
               << ",\n  \"max_abs_m\": " << maxError << ",\n  \"limit_m\": 1e-5,\n  \"passed\": "
               << (maxError <= 1.0e-5 ? "true" : "false") << "\n}\n";
        gpuVerificationDone = true;
        std::cout << "GPU coordinate verification max_abs=" << maxError << " m\n";
        if (maxError > 1.0e-5) vks::tools::exitFatal("GPU coordinate verification exceeded 1e-5 m", -1);
    }

    void render() override {
        if (!prepared) return;
        FrameResources& frame = frames[currentBuffer];
        VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[currentBuffer], VK_TRUE, UINT64_MAX));
        collectGpuTimestamp(frame);
        VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentBuffer]));
        advanceSimulation();
        updateMappedBuffers(frame);
        VulkanExampleBase::prepareFrame(false);
        buildCommandBuffer(frame);
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit = vks::initializers::submitInfo();
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &presentCompleteSemaphores[currentBuffer];
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &drawCmdBuffers[currentBuffer];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderCompleteSemaphores[currentImageIndex];
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submit, waitFences[currentBuffer]));
        if (verifyMode && !gpuVerificationDone) {
            VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[currentBuffer], VK_TRUE, UINT64_MAX));
            verifyGpuReadback(frame);
        }
        VulkanExampleBase::submitFrame(true);
    }

    static double percentile(std::vector<double> values, double fraction) {
        if (values.empty()) return 0.0;
        std::sort(values.begin(), values.end());
        const size_t index = std::min(values.size() - 1, static_cast<size_t>(std::ceil(fraction * values.size()) - 1.0));
        return values[index];
    }

    void writeBenchmarkCsv() {
        benchmarkWritten = true;
        if (inferenceSamples.size() != kBenchmarkSamples || gpuSamples.size() != kBenchmarkSamples) {
            std::cerr << "Benchmark sample shortfall: CPU=" << inferenceSamples.size() << ", GPU=" << gpuSamples.size() << "\n";
            return;
        }
        if (benchmarkPath.has_parent_path()) std::filesystem::create_directories(benchmarkPath.parent_path());
        std::ofstream stream(benchmarkPath);
        stream << "device,threads,vertices,upload_bytes,model_create_ms,samples,input_median_ms,input_p95_ms,inference_median_ms,inference_p95_ms,gpu_transform_median_ms,gpu_transform_p95_ms,total_step_median_ms,total_step_p95_ms,dropped_steps\n";
        stream << '"' << deviceProperties.deviceName << "\"," << threads << ',' << kVertexCount << ',' << kPointBytes << ','
               << std::fixed << std::setprecision(6) << runtime->creationMilliseconds() << ',' << kBenchmarkSamples << ','
               << percentile(inputSamples, 0.5) << ',' << percentile(inputSamples, 0.95) << ','
               << percentile(inferenceSamples, 0.5) << ',' << percentile(inferenceSamples, 0.95) << ','
               << percentile(gpuSamples, 0.5) << ',' << percentile(gpuSamples, 0.95) << ','
               << percentile(totalStepSamples, 0.5) << ',' << percentile(totalStepSamples, 0.95) << ',' << droppedSteps << '\n';
        std::cout << "Wrote benchmark: " << benchmarkPath << "\n";
    }

    void keyPressed(uint32_t keyCode) override {
        if (keyCode == KEY_P || keyCode == 0x50) simulationPaused = !simulationPaused;
        if (keyCode == 0x52) resetRequested = true;
    }

    void OnUpdateUIOverlay(vks::UIOverlay* overlay) override {
        if (!overlay->header("MLCloth CPU upload")) return;
        overlay->text("Animation frame: %u / %u", clipFrame, clipInfo.header.frameCount);
        overlay->text("Model: %d -> %d, PCA %d", modelInfo.driverFeatureLen, modelInfo.drivenFeatureLen, modelInfo.pcaDim);
        overlay->text("Points: %u, upload: %llu bytes", kVertexCount, static_cast<unsigned long long>(kPointBytes));
        overlay->text("CPU inference last: %.3f ms", lastInferenceMs);
        if (!recentInferenceSamples.empty()) overlay->text("CPU inference median/p95 (recent): %.3f / %.3f ms", percentile(recentInferenceSamples, 0.5), percentile(recentInferenceSamples, 0.95));
        overlay->text("GPU transform: %.4f ms", lastGpuTransformMs);
        overlay->text("Dropped simulation steps: %llu", static_cast<unsigned long long>(droppedSteps));
        overlay->text("Status: %s", errorStatus.c_str());
        overlay->text("P: pause  R: deterministic reset  Esc: exit");
        if (overlay->checkBox("Paused", &simulationPaused)) accumulatorSeconds = 0.0;
        if (overlay->button("Reset")) resetRequested = true;
    }
};

VULKAN_EXAMPLE_MAIN()
