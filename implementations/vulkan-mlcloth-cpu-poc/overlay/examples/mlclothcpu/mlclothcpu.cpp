/*
 * AILab/MNN CPU inference -> Vulkan upload -> compute transform -> point cloud.
 * This validation sample intentionally contains no XPBD or cloth topology.
 */
#include "vulkanexamplebase.h"
#include "mlcloth_formats.h"
#include "mlcloth_runtime.h"
#include "mlcloth_xpbd.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class VulkanExample : public VulkanExampleBase {
public:
    static constexpr uint32_t kVertexCount = mlcloth::MLClothSequenceState::kVertexCount;
    static constexpr uint32_t kBenchmarkWarmup = 200;
    static constexpr uint32_t kBenchmarkSamples = 1000;

    struct FrameResources {
        vks::Buffer upload;
        vks::Buffer points;
        vks::Buffer normals;
        vks::Buffer transformUniform;
        vks::Buffer cameraUniform;
        VkDescriptorSet computeSet{ VK_NULL_HANDLE };
        VkDescriptorSet normalSet{ VK_NULL_HANDLE };
        VkDescriptorSet graphicsSet{ VK_NULL_HANDLE };
        VkQueryPool queryPool{ VK_NULL_HANDLE };
        bool queryIssued{};
    };
    std::array<FrameResources, maxConcurrentFrames> frames{};

    // Topology is optional: without --mesh the sample is the original point-cloud
    // path, unchanged down to the verified coordinates, so adding the mesh cannot
    // regress the existing verify and benchmark numbers.
    struct MeshResources {
        vks::Buffer triangles;      // index buffer and storage buffer for the normal pass
        vks::Buffer csrOffsets;
        vks::Buffer csrIndices;
        vks::Buffer uniform;
    } meshResources;

    struct alignas(16) MeshUniform {
        uint32_t vertexCount{};
        uint32_t triangleCount{};
        uint32_t reserved0{};
        uint32_t reserved1{};
    } meshUniform;
    static_assert(sizeof(MeshUniform) == 16);

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
    VkDescriptorSetLayout normalSetLayout{ VK_NULL_HANDLE };
    VkDescriptorSetLayout graphicsSetLayout{ VK_NULL_HANDLE };
    VkPipelineLayout computePipelineLayout{ VK_NULL_HANDLE };
    VkPipelineLayout normalPipelineLayout{ VK_NULL_HANDLE };
    VkPipelineLayout graphicsPipelineLayout{ VK_NULL_HANDLE };
    VkPipeline computePipeline{ VK_NULL_HANDLE };
    VkPipeline normalPipeline{ VK_NULL_HANDLE };
    VkPipeline graphicsPipeline{ VK_NULL_HANDLE };
    VkPipeline clothPipeline{ VK_NULL_HANDLE };

    std::filesystem::path runtimeDirectory;
    std::filesystem::path modelPath;
    std::filesystem::path clipPath;
    std::filesystem::path meshPath;
    std::filesystem::path benchmarkPath{ "mlcloth_benchmark.csv" };
    int threads{ 1 };
    bool verifyMode{};
    bool benchmarkMode{};
    bool pointMode{};
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
    std::vector<uint8_t> meshBytes;

    // CPU XPBD post-constraint. The network prediction is both the guide and the initial
    // state; the solve runs in metres so the sibling PoC tuned constants transfer unchanged.
    bool xpbdEnabled{};
    mlcloth::XpbdConfig xpbdConfig{};
    mlcloth::XpbdSolver xpbdSolver;
    std::vector<uint32_t> xpbdPairs;
    uint32_t xpbdBendStart{};
    std::vector<float> xpbdTargetLength;
    std::vector<float> xpbdTargetArea;
    std::vector<float> guideLocalM;      // the prediction, metres, reference-bone local
    std::vector<float> solvedLocalM;
    std::vector<float> solvedLocalCm;    // what buildPointData consumes
    std::vector<float> previousLocalM;   // for the Verlet extrapolation
    std::vector<float> olderLocalM;
    std::vector<float> inertialLocalM;
    uint32_t solvedFrames{};

    // Runtime animation selection, following the sibling PoC's `hoodCollectMotions` /
    // `hoodLoadAnimation` split: the clip is swapped without touching the model, the runtime,
    // the topology or the constraint set, because those are shared by every motion of this
    // garment. That is the correctness argument and it is also the zero-per-asset-authoring
    // claim the comparison is meant to demonstrate -- if switching motions needed anything
    // re-authored, the claim would be false.
    std::filesystem::path clipDirectory;
    std::vector<std::filesystem::path> availableClipPaths;
    std::vector<std::string> availableClipNames;
    int32_t selectedClip{};
    // Frame decimation, which is the speed axis. The timestep is deliberately NOT scaled with
    // it: the point is to ask what happens when the body moves faster per solver step, and
    // scaling the timestep would cancel exactly that. Integer only, so a value reproduces a
    // published column rather than landing between two.
    int32_t frameStep{ 1 };
    bool holdLastFrame{};
    std::string clipSwitchStatus{ "OK" };
    std::string collisionPieces;
    std::vector<uint8_t> collisionMaskHost;

    // Side-by-side comparison. One inference feeds both branches -- the network output is the
    // same for pure and hybrid by definition -- so this costs one extra XPBD solve and nothing
    // else. Branch 0 is the network alone, branch 1 is the network plus constraints.
    bool compareMode{};
    float compareSpacingCm{ 90.0f };
    static constexpr uint32_t kBranchA = 0;   // pure network
    static constexpr uint32_t kBranchB = 1;   // network + XPBD
    uint32_t branchCount{ 1 };

    // Body collision geometry. Capsules are bone-local, so the clip's own 45 driver transforms
    // place them: no new per-frame data and no runtime skinning. They are also exactly the
    // geometry the penetration criterion uses, which is why they are what gets drawn rather
    // than a body mesh borrowed from the sibling PoC -- that asset lives in a different space
    // and a few centimetres of registration error would make the overlay worse than nothing.
    std::filesystem::path capsulePath;
    std::vector<uint8_t> capsuleBytes;
    mlcloth::CapsuleInfo capsuleInfo{};
    bool capsulesLoaded{};
    bool drawBody{ true };
    std::vector<mlcloth::Capsule> placedCapsulesM;      // reference-bone local, metres
    struct BodyMesh {
        vks::Buffer positions;
        vks::Buffer normals;
        vks::Buffer indices;
        uint32_t indexCount{};
        uint32_t vertexCount{};
    } body;
    // Template unit-sphere directions plus which cap each belongs to, so a capsule of any
    // radius and length is exact rather than a stretched sphere.
    std::vector<glm::vec4> capsuleTemplate;            // xyz direction, w cap sign
    std::vector<uint32_t> capsuleTemplateIndices;
    std::vector<glm::vec4> bodyPositionsHost;
    std::vector<glm::vec4> bodyNormalsHost;

    // Penetration against the capsules, for the network alone and for the network plus
    // constraints. Reported for both because the absolute number is not the interesting one:
    // the capsules are a ragdoll envelope larger than the skin, and ChaosCloth's own settled
    // T-pose reads 1.42% of vertices inside them at 6.7 cm deep. Only the comparison between
    // branches means anything, which is exactly what the side-by-side view is for.
    struct Penetration {
        uint32_t inside{};
        float deepestM{};
        float fraction{};
    };
    Penetration purePenetration{};
    Penetration solvedPenetration{};

    // Per surface component, because the whole-garment figure is dominated by whichever pieces
    // are *supposed* to sit inside the capsules. Measured on ChaosCloth's own solution for a run
    // clip -- the reference, correct by definition -- the four pieces of this garment read 69%,
    // 98%, 14% and 100% inside. The fitted bodice and the collar lie on the torso, and the torso
    // capsules are a ragdoll envelope wider than the skin, so "inside" is the right answer there.
    // Only the skirt's 14% is a number a solver should be trying to reduce.
    //
    // This matters more than a reporting detail: a whole-garment column would have shown
    // 67% -> 27% for the hybrid and read as a large win, when what it actually shows is the
    // constraints lifting a fitted bodice off the chest.
    std::vector<uint32_t> componentOfVertex;
    uint32_t componentCount{};
    std::vector<Penetration> purePerComponent;
    std::vector<Penetration> solvedPerComponent;
    double lastSolveMs{};
    mlcloth::ModelInfo modelInfo{};
    mlcloth::ClipInfo clipInfo{};
    mlcloth::MeshInfo meshInfo{};
    bool meshMode{};
    std::vector<float> localFu;
    std::vector<float> componentFu;
    std::vector<float> componentPositionsCm;
    std::unique_ptr<mlcloth::AILabRuntime> runtime;
    mlcloth::MLClothSequenceState sequence;
    std::vector<glm::vec4> latestLocalPoints;
    std::vector<glm::vec4> latestWorldPoints;
    uint64_t firstFrameHash{};
    // Hashed over the positions that reach the GPU, not the network raw output. Hashing the
    // raw output made the no-regression gate vacuous: it is unaffected by XPBD by
    // construction, so `--xpbd-iterations 0` reproduced it whether or not the solver worked,
    // and so did `--xpbd-iterations 8` while visibly moving vertices 2.5 cm.
    uint64_t firstSolvedHash{};

    std::vector<double> inferenceSamples;
    std::vector<double> recentInferenceSamples;
    std::vector<double> inputSamples;
    std::vector<double> totalStepSamples;
    std::vector<double> gpuSamples;
    uint64_t benchmarkCpuSteps{};
    uint64_t benchmarkGpuFrames{};

    VkDeviceSize branchPointBytes() const {
        return static_cast<VkDeviceSize>(kVertexCount) * branchCount * sizeof(glm::vec4);
    }

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
        meshPath = argumentValue("--mesh", "");
        // Where to look for sibling clips. Defaults to the directory the chosen clip lives in,
        // so a bake directory becomes a selectable set with no extra flag.
        clipDirectory = argumentValue("--clip-dir", "");
        benchmarkPath = argumentValue("--benchmark-output", "mlcloth_benchmark.csv");
        threads = std::stoi(argumentValue("--threads", "1"));
        requestedFrames = static_cast<uint32_t>(std::stoul(argumentValue("--frames", "0")));
        verifyMode = hasArgument("--verify");
        benchmarkMode = hasArgument("--benchmark");
        pointMode = hasArgument("--points");
        // Zero iterations is the no-regression arm: the solver returns the network output
        // byte for byte, so --xpbd --xpbd-iterations 0 must reproduce the pure-MNN hash.
        compareMode = hasArgument("--compare");
        compareSpacingCm = std::stof(argumentValue("--compare-spacing-cm", "90"));
        capsulePath = argumentValue("--capsules", "");
        // Which garment pieces get contacts, as a comma-separated list of largest-first piece
        // indices. Empty means all, which measurement shows is wrong for this garment; "2" is the
        // skirt, the one piece whose penetration is actually reducible.
        collisionPieces = argumentValue("--collision-pieces", "");
        drawBody = !hasArgument("--no-body");
        xpbdEnabled = hasArgument("--xpbd") || compareMode;
        xpbdConfig.iterations = std::stoi(argumentValue("--xpbd-iterations", "8"));
        xpbdConfig.timestep = std::stof(argumentValue("--xpbd-timestep", "0.0333333333"));
        xpbdConfig.relaxation = std::stof(argumentValue("--xpbd-relaxation", "1.0"));
        xpbdConfig.oneSided = !hasArgument("--xpbd-two-sided");
        // alpha-tilde = compliance / dt^2, so at 30 Hz a compliance of 1 already gives 900
        // against an inverse-mass sum near 90: anything much above 0.1 is inert here, and the
        // useful band is narrow. See SolverConfig::guide_compliance in the sibling PoC.
        xpbdConfig.stretchCompliance = std::stof(argumentValue("--xpbd-stretch-compliance", "0.0"));
        xpbdConfig.bendCompliance = std::stof(argumentValue("--xpbd-bend-compliance", "0.05"));
        xpbdConfig.areaFloor = std::stof(argumentValue("--xpbd-area-floor", "0.0"));
        xpbdConfig.areaCompliance = std::stof(argumentValue("--xpbd-area-compliance", "0.0"));
        // Negative disables the guide, which is the standard mode where the prediction is
        // only the initial state and the constraints have the last word.
        xpbdConfig.guideCompliance = std::stof(argumentValue("--xpbd-guide-compliance", "-1.0"));
        xpbdConfig.guideTrustRatio = std::stof(argumentValue("--xpbd-guide-trust", "0.0"));
        // Capsule placement per frame is not wired yet, so collision stays off in the
        // runtime even though the solver and the baked capsules both support it.
        // Collision is on whenever capsules are available: the point of the comparison is to
        // see whether the constraints keep the garment off the body, and a hybrid branch with
        // contacts disabled would be answering a different question.
        xpbdConfig.collision = !hasArgument("--no-collision");
        xpbdConfig.contactOffset = std::stof(argumentValue("--xpbd-contact-offset", "0.0"));
        if (runtimeDirectory.empty() || modelPath.empty() || clipPath.empty()) {
            throw std::runtime_error("--runtime-dir, --model and --clip are required");
        }
        meshMode = !meshPath.empty() && !pointMode;
        // Two branches only make sense with topology: the comparison is about whether the
        // constraints keep the surface off the body, which a point cloud cannot show.
        if (compareMode && !meshMode) throw std::runtime_error("--compare needs --mesh");
        branchCount = compareMode ? 2u : 1u;
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
            frame.normals.destroy();
            frame.transformUniform.destroy();
            frame.cameraUniform.destroy();
            if (frame.queryPool) vkDestroyQueryPool(device, frame.queryPool, nullptr);
        }
        meshResources.triangles.destroy();
        meshResources.csrOffsets.destroy();
        meshResources.csrIndices.destroy();
        meshResources.uniform.destroy();
        body.positions.destroy();
        body.normals.destroy();
        body.indices.destroy();
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipeline(device, normalPipeline, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipeline(device, clothPipeline, nullptr);
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, normalPipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, computeSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, normalSetLayout, nullptr);
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
        latestLocalPoints.resize(size_t(kVertexCount) * branchCount);
        latestWorldPoints.resize(size_t(kVertexCount) * branchCount);
        std::cout << "AILab model preloaded in " << runtime->creationMilliseconds() << " ms; clip="
                  << clipInfo.header.frameCount << " frames, drivers=" << clipInfo.header.driverCount << "\n";
        if (meshMode) {
            meshBytes = readFile(meshPath);
            // The model digest is the argument, not the mesh's own: it is what stops a
            // topology exported for another garment from loading against this model.
            if (!mlcloth::parse_mesh(meshBytes.data(), meshBytes.size(), modelHash, meshInfo, parseError)) {
                throw std::runtime_error("MLMSH001 validation failed: " + parseError);
            }
            if (meshInfo.vertices != kVertexCount) {
                throw std::runtime_error("Mesh vertex count does not match the inference output");
            }
            // The normal pass and the index buffer cover every branch, with the second copy
            // offset by one vertex block. Duplicating the topology rather than drawing the same
            // one twice keeps each branch's normals independent, which matters because the whole
            // point of the comparison is that the two surfaces differ.
            meshUniform.vertexCount = meshInfo.vertices * branchCount;
            meshUniform.triangleCount = meshInfo.triangles * branchCount;
            std::cout << "Mesh: " << meshInfo.vertices << " vertices, " << meshInfo.triangles
                      << " triangles, " << meshInfo.edges << " edges, " << meshInfo.boundaryLoops
                      << " boundary loops, " << meshInfo.pinnedVertices << " pinned\n";
            if (xpbdEnabled) buildXpbd();
            deriveComponents();
            loadCapsules(modelHash);
            applyCollisionPieces();
        }
        collectClips();
        if (availableClipNames.size() > 1) {
            std::cout << "Clips available for selection: " << availableClipNames.size() << "\n";
        }
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

    // Writes one vertex block per branch. Branch A is the network's own output and branch B is
    // the solved one; in single-branch mode there is only whichever the caller passed. The
    // lateral offset is applied in *reference-bone-local* coordinates, before the transform, so
    // it is a rigid sideways shift of the whole branch and not a change to any constraint.
    void buildPointData(uint32_t frame, const float* output) {
        glm::vec3 up = glm::normalize(rootUp(frame));
        glm::vec3 right = glm::normalize(rootRight(frame));
        glm::vec3 forward = glm::normalize(glm::cross(right, up));
        const glm::vec3 root = rootPosition(frame);
        glm::vec3 minimum(std::numeric_limits<float>::max());
        glm::vec3 maximum(std::numeric_limits<float>::lowest());
        for (uint32_t slot = 0; slot < kVertexCount * branchCount; ++slot) {
            const uint32_t branch = slot / kVertexCount;
            const uint32_t vertex = slot % kVertexCount;
            const float* source = (branchCount > 1 && branch == kBranchA) ? sequence.output().data() : output;
            const float lateral = branchCount > 1
                ? (branch == kBranchA ? -0.5f * compareSpacingCm : 0.5f * compareSpacingCm)
                : 0.0f;
            const glm::vec3 local(source[vertex * 3], source[vertex * 3 + 1] + lateral, source[vertex * 3 + 2]);
            latestLocalPoints[slot] = glm::vec4(local, 1.0f);
            const glm::vec3 component = root + forward * local.x + right * local.y + up * local.z;
            const glm::vec3 world(component.x, component.z, -component.y);
            const glm::vec3 metres = world * 0.01f;
            if (!std::isfinite(metres.x) || !std::isfinite(metres.y) || !std::isfinite(metres.z)) {
                throw std::runtime_error("CPU coordinate reference contains NaN or Inf");
            }
            latestWorldPoints[slot] = glm::vec4(metres, 1.0f);
            minimum = glm::min(minimum, metres);
            maximum = glm::max(maximum, metres);
        }
        const glm::vec3 extent = maximum - minimum;
        if (std::max({ extent.x, extent.y, extent.z }) <= 1.0e-6f) throw std::runtime_error("Predicted cloth AABB is degenerate");
        transformUniform.rootPositionAndCount = glm::vec4(root, 0.0f);
        const uint32_t transformedCount = kVertexCount * branchCount;
        std::memcpy(&transformUniform.rootPositionAndCount.w, &transformedCount, sizeof(transformedCount));
        transformUniform.rootUp = glm::vec4(up, 0.0f);
        transformUniform.rootRight = glm::vec4(right, 0.0f);
    }

    // Constraint targets come from the mesh own stored rest configuration, which for this
    // asset is ChaosCloth settled T-pose rather than an authored flat pattern -- already a
    // physically plausible configuration, so it is a defensible control arm rather than the
    // trap the sibling PoC hit, where skinning an authored rest into frame 0 left 8% of
    // edges past 1.5x rest and aiming at rest contracted them hard. Calibrating over a
    // trajectory instead is the next step and is what the S13 sweep is about; this is the
    // arm it gets measured against.
    // Every `.mldrv` beside the chosen clip. They are not validated here -- a clip baked
    // against another model is refused by `parse_clip`'s hash check at the moment it is
    // selected, and reporting that in the overlay is more useful than hiding the file.
    void collectClips() {
        availableClipPaths.clear();
        availableClipNames.clear();
        const auto root = clipDirectory.empty() ? clipPath.parent_path() : clipDirectory;
        std::error_code code;
        if (std::filesystem::is_directory(root, code)) {
            for (const auto& entry : std::filesystem::directory_iterator(root, code)) {
                if (!entry.is_regular_file() || entry.path().extension() != ".mldrv") continue;
                availableClipPaths.push_back(entry.path());
            }
        }
        std::sort(availableClipPaths.begin(), availableClipPaths.end());
        const auto found = std::find(availableClipPaths.begin(), availableClipPaths.end(), clipPath);
        if (found == availableClipPaths.end()) {
            availableClipPaths.insert(availableClipPaths.begin(), clipPath);
            selectedClip = 0;
        } else {
            selectedClip = static_cast<int32_t>(std::distance(availableClipPaths.begin(), found));
        }
        for (const auto& path : availableClipPaths) availableClipNames.push_back(path.stem().string());
    }

    // Swap the animation and nothing else. Returns false and leaves the current clip playing
    // when the new one is rejected, rather than tearing down a working state: a clip baked
    // against a different model is a real thing to encounter in a directory of bakes, and the
    // hash check in `parse_clip` is what catches it.
    bool loadClip(const std::filesystem::path& path) {
        std::vector<uint8_t> bytes;
        mlcloth::ClipInfo info{};
        std::string parseError;
        try {
            bytes = readFile(path);
        } catch (const std::exception& failure) {
            clipSwitchStatus = std::string("cannot read: ") + failure.what();
            return false;
        }
        const auto modelHash = mlcloth::sha256(modelBytes);
        const auto driverHash = mlcloth::sha256_driver_names(modelInfo.driverNames);
        if (!mlcloth::parse_clip(bytes.data(), bytes.size(), modelHash, driverHash, info, parseError)) {
            clipSwitchStatus = "rejected: " + parseError;
            return false;
        }
        clipBytes = std::move(bytes);
        clipInfo = info;
        clipPath = path;
        localFu.assign(clipInfo.localFu, clipInfo.localFu + clipInfo.header.localFloatCount);
        componentFu.assign(clipInfo.componentFu, clipInfo.componentFu + clipInfo.header.componentFloatCount);
        componentPositionsCm.assign(clipInfo.componentPosCm, clipInfo.componentPosCm + clipInfo.header.positionFloatCount);
        // A full reset, not just a frame rewind. The recurrent state, the solver's two-frame
        // history and the benchmark accumulators all belong to the old motion; carrying any of
        // them over would make a switched-to clip differ from the same clip launched directly.
        resetSequence();
        clipSwitchStatus = "OK";
        return true;
    }

    // One vertex block per capsule per branch, host-visible because it is rewritten every frame
    // as the body moves. 14 capsules x ~300 vertices is small enough that a staging round trip
    // would cost more than the upload it avoids.
    void createBodyResources() {
        if (!capsulesLoaded || !drawBody) return;
        const uint32_t perCapsule = static_cast<uint32_t>(capsuleTemplate.size());
        body.vertexCount = perCapsule * capsuleInfo.count * branchCount;
        body.indexCount = static_cast<uint32_t>(capsuleTemplateIndices.size()) * capsuleInfo.count * branchCount;
        bodyPositionsHost.assign(body.vertexCount, glm::vec4(0.0f));
        bodyNormalsHost.assign(body.vertexCount, glm::vec4(0.0f));

        std::vector<uint32_t> indices;
        indices.reserve(body.indexCount);
        for (uint32_t branch = 0; branch < branchCount; ++branch) {
            for (uint32_t capsule = 0; capsule < capsuleInfo.count; ++capsule) {
                const uint32_t base = (branch * capsuleInfo.count + capsule) * perCapsule;
                for (uint32_t index : capsuleTemplateIndices) indices.push_back(base + index);
            }
        }
        const VkDeviceSize vertexBytes = VkDeviceSize(body.vertexCount) * sizeof(glm::vec4);
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &body.positions, vertexBytes));
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &body.normals, vertexBytes));
        VK_CHECK_RESULT(body.positions.map());
        VK_CHECK_RESULT(body.normals.map());
        createDeviceLocalBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, body.indices,
            indices.data(), indices.size() * sizeof(uint32_t));
        std::cout << "Body: " << capsuleInfo.count << " capsules, " << body.indexCount / 3 << " triangles\n";
    }

    // Evaluates the capsule surface in world metres, from the placement the solver is using, so
    // what is on screen is the geometry the contacts were resolved against rather than a
    // decorative stand-in.
    void updateBodyGeometry(uint32_t frame) {
        if (!capsulesLoaded || !drawBody || body.vertexCount == 0) return;
        const glm::vec3 up = glm::normalize(rootUp(frame));
        const glm::vec3 right = glm::normalize(rootRight(frame));
        const glm::vec3 forward = glm::normalize(glm::cross(right, up));
        const glm::vec3 root = rootPosition(frame);
        const uint32_t perCapsule = static_cast<uint32_t>(capsuleTemplate.size());

        for (uint32_t branch = 0; branch < branchCount; ++branch) {
            const float lateral = branchCount > 1
                ? (branch == kBranchA ? -0.5f * compareSpacingCm : 0.5f * compareSpacingCm)
                : 0.0f;
            for (uint32_t index = 0; index < capsuleInfo.count; ++index) {
                const mlcloth::Capsule& capsule = placedCapsulesM[index];
                const glm::vec3 axis(capsule.axis[0], capsule.axis[1], capsule.axis[2]);
                // Any two directions orthogonal to the axis will do; the choice only rotates the
                // triangulation around the barrel. Picked away from the axis so the cross product
                // is well conditioned.
                const glm::vec3 seed = std::abs(axis.x) < 0.9f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
                const glm::vec3 u = glm::normalize(glm::cross(axis, seed));
                const glm::vec3 v = glm::cross(axis, u);
                const glm::vec3 centre(capsule.centre[0], capsule.centre[1], capsule.centre[2]);
                const uint32_t base = (branch * capsuleInfo.count + index) * perCapsule;
                for (uint32_t slot = 0; slot < perCapsule; ++slot) {
                    const glm::vec4& templated = capsuleTemplate[slot];
                    const glm::vec3 direction = u * templated.x + v * templated.y + axis * templated.z;
                    // Metres in the reference-bone-local frame, then the same local -> component
                    // -> world chain buildPointData uses, so body and cloth cannot disagree.
                    const glm::vec3 localM = centre + direction * capsule.radius + axis * (capsule.halfLength * templated.w);
                    glm::vec3 localCm = localM * 100.0f;
                    localCm.y += lateral;
                    const glm::vec3 component = root + forward * localCm.x + right * localCm.y + up * localCm.z;
                    const glm::vec3 world(component.x, component.z, -component.y);
                    bodyPositionsHost[base + slot] = glm::vec4(world * 0.01f, 1.0f);
                    const glm::vec3 componentNormal = forward * direction.x + right * direction.y + up * direction.z;
                    const glm::vec3 worldNormal(componentNormal.x, componentNormal.z, -componentNormal.y);
                    bodyNormalsHost[base + slot] = glm::vec4(glm::normalize(worldNormal), 0.0f);
                }
            }
        }
        std::memcpy(body.positions.mapped, bodyPositionsHost.data(), bodyPositionsHost.size() * sizeof(glm::vec4));
        std::memcpy(body.normals.mapped, bodyNormalsHost.data(), bodyNormalsHost.size() * sizeof(glm::vec4));
    }

    void loadCapsules(const mlcloth::Sha256Digest& modelHash) {
        auto path = capsulePath;
        if (path.empty()) {
            const auto fallback = meshPath.parent_path().parent_path() / "capsules" / "ch10032.mlcap";
            if (std::filesystem::exists(fallback)) path = fallback;
        }
        if (path.empty() || !std::filesystem::exists(path)) {
            std::cout << "Capsules: none (pass --capsules for body collision and the body overlay)\n";
            xpbdConfig.collision = false;
            return;
        }
        capsuleBytes = readFile(path);
        std::string parseError;
        if (!mlcloth::parse_capsules(capsuleBytes.data(), capsuleBytes.size(), modelHash, capsuleInfo, parseError)) {
            throw std::runtime_error("MLCAP001 validation failed: " + parseError);
        }
        capsulesLoaded = true;
        placedCapsulesM.resize(capsuleInfo.count);
        buildCapsuleTemplate();
        std::cout << "Capsules: " << capsuleInfo.count << " bodies on driver bones\n";
    }

    // A capsule of radius r and half-length h is two hemispheres of radius r centred at +-h
    // joined by a cylinder. Storing unit sphere directions with a cap sign, rather than a baked
    // mesh, keeps that exact for every radius and length: a scaled sphere would give the wrong
    // cap curvature and the wrong normals along the barrel.
    void buildCapsuleTemplate() {
        constexpr uint32_t latitudes = 8;      // per hemisphere
        constexpr uint32_t longitudes = 16;
        capsuleTemplate.clear();
        capsuleTemplateIndices.clear();
        // Rows run from the top pole to the bottom pole. The two equator rows share a direction
        // and differ only in cap sign, which is what forms the barrel.
        for (uint32_t cap = 0; cap < 2; ++cap) {
            const float sign = cap == 0 ? 1.0f : -1.0f;
            for (uint32_t lat = 0; lat <= latitudes; ++lat) {
                const float theta = (float(lat) / float(latitudes)) * (glm::pi<float>() * 0.5f);
                const float z = std::cos(theta) * sign;
                const float ring = std::sin(theta);
                for (uint32_t lon = 0; lon <= longitudes; ++lon) {
                    const float phi = (float(lon) / float(longitudes)) * glm::two_pi<float>();
                    capsuleTemplate.emplace_back(ring * std::cos(phi), ring * std::sin(phi), z, sign);
                }
            }
        }
        const uint32_t stride = longitudes + 1;
        const uint32_t rowsPerCap = latitudes + 1;
        for (uint32_t cap = 0; cap < 2; ++cap) {
            const uint32_t base = cap * rowsPerCap * stride;
            for (uint32_t lat = 0; lat < latitudes; ++lat) {
                for (uint32_t lon = 0; lon < longitudes; ++lon) {
                    const uint32_t a = base + lat * stride + lon;
                    const uint32_t b = a + 1;
                    const uint32_t c = a + stride;
                    const uint32_t d = c + 1;
                    // Wound so the top cap faces outward; the bottom cap is mirrored by its
                    // negative z, so its winding is reversed to keep the outward face outward.
                    if (cap == 0) {
                        capsuleTemplateIndices.insert(capsuleTemplateIndices.end(), { a, c, d, a, d, b });
                    } else {
                        capsuleTemplateIndices.insert(capsuleTemplateIndices.end(), { a, d, c, a, b, d });
                    }
                }
            }
        }
        // The barrel: the last row of the top cap to the last row of the bottom cap.
        const uint32_t topEquator = (rowsPerCap - 1) * stride;
        const uint32_t bottomEquator = rowsPerCap * stride + (rowsPerCap - 1) * stride;
        for (uint32_t lon = 0; lon < longitudes; ++lon) {
            const uint32_t a = topEquator + lon;
            const uint32_t b = a + 1;
            const uint32_t c = bottomEquator + lon;
            const uint32_t d = c + 1;
            capsuleTemplateIndices.insert(capsuleTemplateIndices.end(), { a, c, d, a, d, b });
        }
    }

    // Component-space basis of a driver bone, reconstructed from the clip's 6D rotation. The
    // stored pair is (q*Z, q*Y); a proper rotation's remaining column is their cross product,
    // so no extra data is needed. Same construction `buildPointData` uses for the root, which
    // is why the two agree by definition rather than by coincidence.
    void boneBasis(uint32_t frame, uint32_t bone, glm::vec3& ex, glm::vec3& ey, glm::vec3& ez) const {
        const size_t offset = (static_cast<size_t>(frame) * mlcloth::kDriverCount + bone) * 6;
        ez = glm::normalize(glm::vec3(componentFu[offset], componentFu[offset + 1], componentFu[offset + 2]));
        ey = glm::normalize(glm::vec3(componentFu[offset + 3], componentFu[offset + 4], componentFu[offset + 5]));
        ex = glm::normalize(glm::cross(ey, ez));
    }

    glm::vec3 bonePosition(uint32_t frame, uint32_t bone) const {
        const size_t offset = (static_cast<size_t>(frame) * mlcloth::kDriverCount + bone) * 3;
        return glm::vec3(componentPositionsCm[offset], componentPositionsCm[offset + 1], componentPositionsCm[offset + 2]);
    }

    // Places every capsule for this frame in reference-bone-local metres, which is the space the
    // solver and the cloth prediction share, so no registration is involved anywhere.
    void placeCapsules(uint32_t frame) {
        if (!capsulesLoaded) return;
        glm::vec3 rx, ry, rz;
        boneBasis(frame, mlcloth::kRootDriverIndex, rx, ry, rz);
        const glm::vec3 rootPos = bonePosition(frame, mlcloth::kRootDriverIndex);
        for (uint32_t index = 0; index < capsuleInfo.count; ++index) {
            const uint32_t bone = capsuleInfo.driverIndices[index];
            glm::vec3 bx, by, bz;
            boneBasis(frame, bone, bx, by, bz);
            const glm::vec3 bonePos = bonePosition(frame, bone);
            const glm::vec3 localCentre(capsuleInfo.centresCm[3 * index], capsuleInfo.centresCm[3 * index + 1], capsuleInfo.centresCm[3 * index + 2]);
            const glm::vec3 localAxis(capsuleInfo.axes[3 * index], capsuleInfo.axes[3 * index + 1], capsuleInfo.axes[3 * index + 2]);
            const glm::vec3 centreComponent = bonePos + bx * localCentre.x + by * localCentre.y + bz * localCentre.z;
            const glm::vec3 axisComponent = glm::normalize(bx * localAxis.x + by * localAxis.y + bz * localAxis.z);
            const glm::vec3 delta = centreComponent - rootPos;
            const glm::vec3 centreLocal(glm::dot(delta, rx), glm::dot(delta, ry), glm::dot(delta, rz));
            const glm::vec3 axisLocal(glm::dot(axisComponent, rx), glm::dot(axisComponent, ry), glm::dot(axisComponent, rz));
            mlcloth::Capsule& capsule = placedCapsulesM[index];
            for (int k = 0; k < 3; ++k) {
                capsule.centre[k] = centreLocal[k] * 0.01f;
                capsule.axis[k] = axisLocal[k];
            }
            capsule.radius = capsuleInfo.sizesCm[2 * index] * 0.01f;
            capsule.halfLength = capsuleInfo.sizesCm[2 * index + 1] * 0.01f;
        }
    }

    void buildXpbd() {
        const uint32_t vertices = meshInfo.vertices;
        xpbdPairs.assign(meshInfo.edgePairs, meshInfo.edgePairs + size_t(meshInfo.edges) * 2);
        xpbdBendStart = meshInfo.edges;
        const std::vector<uint32_t> bend =
            mlcloth::derive_bend_pairs(vertices, meshInfo.triangleIndices, meshInfo.triangles);
        xpbdPairs.insert(xpbdPairs.end(), bend.begin(), bend.end());

        const float* restCm = meshInfo.positionsCm;
        const uint32_t constraints = static_cast<uint32_t>(xpbdPairs.size() / 2);
        xpbdTargetLength.resize(constraints);
        for (uint32_t index = 0; index < constraints; ++index) {
            const uint32_t a = xpbdPairs[2 * index], b = xpbdPairs[2 * index + 1];
            const float dx = restCm[3 * a] - restCm[3 * b];
            const float dy = restCm[3 * a + 1] - restCm[3 * b + 1];
            const float dz = restCm[3 * a + 2] - restCm[3 * b + 2];
            xpbdTargetLength[index] = std::sqrt(dx * dx + dy * dy + dz * dz) * 0.01f;
        }
        xpbdTargetArea.resize(meshInfo.triangles);
        for (uint32_t triangle = 0; triangle < meshInfo.triangles; ++triangle) {
            const uint32_t a = meshInfo.triangleIndices[3 * triangle];
            const uint32_t b = meshInfo.triangleIndices[3 * triangle + 1];
            const uint32_t c = meshInfo.triangleIndices[3 * triangle + 2];
            const glm::vec3 pa(restCm[3 * a], restCm[3 * a + 1], restCm[3 * a + 2]);
            const glm::vec3 pb(restCm[3 * b], restCm[3 * b + 1], restCm[3 * b + 2]);
            const glm::vec3 pc(restCm[3 * c], restCm[3 * c + 1], restCm[3 * c + 2]);
            xpbdTargetArea[triangle] = 0.5f * glm::length(glm::cross(pb - pa, pc - pa)) * 1.0e-4f;
        }

        std::string err;
        if (!xpbdSolver.build(vertices, xpbdPairs.data(), constraints, xpbdBendStart,
                              meshInfo.triangleIndices, meshInfo.triangles,
                              meshInfo.vertexMassKg, meshInfo.pinMask,
                              xpbdTargetLength.data(), xpbdTargetArea.data(), err)) {
            throw std::runtime_error("XPBD solver build failed: " + err);
        }
        xpbdSolver.configure(xpbdConfig);
        const size_t values = size_t(vertices) * 3;
        guideLocalM.assign(values, 0.0f);
        solvedLocalM.assign(values, 0.0f);
        solvedLocalCm.assign(values, 0.0f);
        previousLocalM.assign(values, 0.0f);
        olderLocalM.assign(values, 0.0f);
        inertialLocalM.assign(values, 0.0f);
        solvedFrames = 0;
        std::cout << "XPBD: " << constraints << " constraints (" << xpbdBendStart << " stretch, "
                  << (constraints - xpbdBendStart) << " bend), " << meshInfo.triangles
                  << " area floors, " << xpbdConfig.iterations << " iterations\n";
    }

    // Returns the positions buildPointData should consume: the solved ones when XPBD is on,
    // the network own otherwise.
    // Connected components of the triangle graph, derived at load. The mesh container does not
    // carry them, and a union-find over 10k triangles costs nothing next to one inference.
    void deriveComponents() {
        const uint32_t vertices = meshInfo.vertices;
        std::vector<uint32_t> parent(vertices);
        for (uint32_t v = 0; v < vertices; ++v) parent[v] = v;
        const std::function<uint32_t(uint32_t)> find = [&](uint32_t x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };
        for (uint32_t triangle = 0; triangle < meshInfo.triangles; ++triangle) {
            const uint32_t a = meshInfo.triangleIndices[3 * triangle];
            const uint32_t b = meshInfo.triangleIndices[3 * triangle + 1];
            const uint32_t c = meshInfo.triangleIndices[3 * triangle + 2];
            for (auto pair : { std::make_pair(a, b), std::make_pair(b, c) }) {
                const uint32_t ra = find(pair.first), rb = find(pair.second);
                if (ra != rb) parent[ra] = rb;
            }
        }
        // Labelled largest-first, not first-encountered. The offline tools sort by size, and a
        // per-piece column that pairs the runtime's piece 2 with the tool's piece 2 has to mean
        // the same piece -- labelling by first vertex silently pairs the skirt against the
        // bodice, which is how the first version of this report came out misaligned.
        std::map<uint32_t, uint32_t> sizes;
        for (uint32_t v = 0; v < vertices; ++v) ++sizes[find(v)];
        std::vector<std::pair<uint32_t, uint32_t>> order(sizes.begin(), sizes.end());
        std::stable_sort(order.begin(), order.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        std::map<uint32_t, uint32_t> label;
        componentCount = 0;
        for (const auto& entry : order) label[entry.first] = componentCount++;
        componentOfVertex.assign(vertices, 0);
        for (uint32_t v = 0; v < vertices; ++v) componentOfVertex[v] = label[find(v)];
        for (const auto& entry : order) {
            std::cout << "  piece " << label[entry.first] << ": " << entry.second << " vertices\n";
        }
        purePerComponent.assign(componentCount, Penetration{});
        solvedPerComponent.assign(componentCount, Penetration{});
        std::cout << "Surface components: " << componentCount << "\n";
    }

    // Restricts contacts to the listed pieces. Left empty every vertex takes contacts, which is
    // reported rather than silently accepted because it is measurably the wrong default here.
    void applyCollisionPieces() {
        if (!xpbdEnabled || componentOfVertex.empty()) return;
        if (collisionPieces.empty()) {
            std::cout << "Collision applies to every piece, which measurement says is wrong here: "
                         "against ChaosCloth's own run this lifts the fitted pieces off the body. "
                         "Pass --collision-pieces 2 to restrict contacts to the skirt.\n";
            xpbdSolver.set_collision_mask(nullptr);
            return;
        }
        std::vector<uint32_t> wanted;
        std::stringstream stream(collisionPieces);
        std::string token;
        while (std::getline(stream, token, ',')) {
            if (token.empty()) continue;
            const uint32_t piece = static_cast<uint32_t>(std::stoul(token));
            if (piece >= componentCount) {
                throw std::runtime_error("--collision-pieces names piece " + token + " but the garment has "
                                         + std::to_string(componentCount));
            }
            wanted.push_back(piece);
        }
        collisionMaskHost.assign(meshInfo.vertices, 0);
        for (uint32_t v = 0; v < meshInfo.vertices; ++v) {
            for (uint32_t piece : wanted) {
                if (componentOfVertex[v] == piece) { collisionMaskHost[v] = 1; break; }
            }
        }
        const uint32_t enabled = static_cast<uint32_t>(std::count(collisionMaskHost.begin(), collisionMaskHost.end(), uint8_t(1)));
        xpbdSolver.set_collision_mask(collisionMaskHost.data());
        std::cout << "Collision restricted to piece(s) " << collisionPieces << ": " << enabled << " vertices\n";
    }

    Penetration measurePenetration(const float* positionsM, std::vector<Penetration>& perComponent) const {
        Penetration result{};
        for (Penetration& entry : perComponent) entry = Penetration{};
        if (!capsulesLoaded || placedCapsulesM.empty()) return result;
        std::vector<uint32_t> totals(perComponent.size(), 0);
        for (uint32_t vertex = 0; vertex < kVertexCount; ++vertex) {
            float worst = 0.0f;
            for (const mlcloth::Capsule& capsule : placedCapsulesM) {
                worst = std::min(worst, mlcloth::capsule_signed_distance(&positionsM[vertex * 3], capsule));
            }
            const uint32_t component = componentOfVertex.empty() ? 0u : componentOfVertex[vertex];
            if (component < totals.size()) ++totals[component];
            if (worst < 0.0f) {
                ++result.inside;
                result.deepestM = std::min(result.deepestM, worst);
                if (component < perComponent.size()) {
                    ++perComponent[component].inside;
                    perComponent[component].deepestM = std::min(perComponent[component].deepestM, worst);
                }
            }
        }
        for (size_t index = 0; index < perComponent.size(); ++index) {
            if (totals[index] != 0) perComponent[index].fraction = float(perComponent[index].inside) / float(totals[index]);
        }
        result.fraction = float(result.inside) / float(kVertexCount);
        return result;
    }

    const float* solveXpbd() {
        const auto& output = sequence.output();
        if (!xpbdEnabled || !meshMode) return output.data();
        // With no iterations there is nothing to solve, so the centimetre-to-metre round trip
        // is skipped rather than performed and undone. It is not the identity in float32
        // -- `x * 0.01f * 100.0f` differs from `x` by an ulp on most values -- and that alone
        // was enough to make `--xpbd-iterations 0` produce a different hash from the pure
        // network path, which is precisely the equality this gate exists to assert.
        if (xpbdConfig.iterations <= 0) {
            lastSolveMs = 0.0;
            return output.data();
        }
        const size_t values = size_t(kVertexCount) * 3;
        for (size_t index = 0; index < values; ++index) guideLocalM[index] = output[index] * 0.01f;

        // The Verlet extrapolation 2*x_now - x_previous that the guide trust gate measures
        // against. Before two solved frames exist there is no velocity to extrapolate, so the
        // gate is passed null rather than handed the prediction itself: that would read as
        // zero displacement and trust the network unconditionally on exactly the frame where
        // it is least justified.
        const float* inertial = nullptr;
        if (solvedFrames >= 2 && xpbdConfig.guideTrustRatio > 0.0f) {
            for (size_t index = 0; index < values; ++index) {
                inertialLocalM[index] = 2.0f * previousLocalM[index] - olderLocalM[index];
            }
            inertial = inertialLocalM.data();
        }

        const auto begin = std::chrono::steady_clock::now();
        xpbdSolver.step(guideLocalM.data(), inertial, placedCapsulesM, solvedLocalM.data());
        lastSolveMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();

        std::swap(olderLocalM, previousLocalM);
        std::copy(solvedLocalM.begin(), solvedLocalM.end(), previousLocalM.begin());
        if (solvedFrames < 2) ++solvedFrames;
        purePenetration = measurePenetration(guideLocalM.data(), purePerComponent);
        solvedPenetration = measurePenetration(solvedLocalM.data(), solvedPerComponent);
        for (size_t index = 0; index < values; ++index) solvedLocalCm[index] = solvedLocalM[index] * 100.0f;
        return solvedLocalCm.data();
    }

    static uint64_t fnv1a64(const float* values, size_t count) {
        constexpr uint64_t offset = 1469598103934665603ull;
        constexpr uint64_t prime = 1099511628211ull;
        uint64_t hash = offset;
        const auto* bytes = reinterpret_cast<const uint8_t*>(values);
        for (size_t index = 0; index < count * sizeof(float); ++index) {
            hash ^= bytes[index];
            hash *= prime;
        }
        return hash;
    }

    void simulateOneStep(bool collectBenchmark) {
        const uint32_t frame = clipFrame;
        const auto begin = std::chrono::steady_clock::now();
        lastInferenceMs = sequence.inferFrame(*runtime, frame, localFu, componentFu, componentPositionsCm);
        const auto afterInference = std::chrono::steady_clock::now();
        placeCapsules(frame);
        const float* solved = solveXpbd();
        if (simulationSteps == 0) firstSolvedHash = fnv1a64(solved, size_t(kVertexCount) * 3);
        buildPointData(frame, solved);
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
        // Decimation, so the body moves `frameStep` clip frames per solver step. Held at the
        // last frame instead of looping when asked, which is how a settle is watched: looping
        // restarts the recurrent state and hides whatever the garment was converging to.
        const uint32_t step = static_cast<uint32_t>(std::max(1, frameStep));
        if (holdLastFrame && clipFrame + step >= clipInfo.header.frameCount) {
            clipFrame = clipInfo.header.frameCount - 1;
        } else {
            clipFrame += step;
            if (clipFrame >= clipInfo.header.frameCount) {
                clipFrame = 0;
                sequence.reset();
                // Otherwise frame 0 of the second pass takes the *first* pass's last two frames
                // as its inertial reference, so a looped clip would not reproduce a fresh one.
                resetXpbdHistory();
            }
        }
    }

    // The solver keeps two frames of history for the Verlet extrapolation the guide trust gate
    // measures against, so it has to be reset alongside the recurrent network state. It was
    // not, and nothing caught it: the reset-determinism check below hashed the network raw
    // output, which the solver cannot influence, so a solve carrying state across a reset or a
    // clip loop would have gone unnoticed.
    void resetXpbdHistory() {
        std::fill(previousLocalM.begin(), previousLocalM.end(), 0.0f);
        std::fill(olderLocalM.begin(), olderLocalM.end(), 0.0f);
        std::fill(inertialLocalM.begin(), inertialLocalM.end(), 0.0f);
        solvedFrames = 0;
    }

    void resetSequence() {
        sequence.reset();
        resetXpbdHistory();
        clipFrame = 0;
        accumulatorSeconds = 0.0;
        simulationSteps = 0;
        firstFrameHash = 0;
        firstSolvedHash = 0;
        simulateOneStep(false);
    }

    // Plays the clip through and then one frame past the loop, asserting that the wrapped
    // frame reproduces the first. Both hashes are checked: the network's own output, and the
    // positions that reach the GPU. The second is the one that covers the solver, and adding it
    // is what turned this from a check of the recurrent state into a check of the whole step.
    void runCpuVerification() {
        resetXpbdHistory();
        sequence.reset();
        clipFrame = 0;
        simulationSteps = 0;
        uint64_t baselineFirstHash = 0;
        uint64_t baselineSolvedHash = 0;
        for (uint32_t frame = 0; frame < clipInfo.header.frameCount; ++frame) {
            simulateOneStep(false);
            if (frame == 0) {
                baselineFirstHash = sequence.outputHash64();
                baselineSolvedHash = firstSolvedHash;
            }
        }
        if (clipFrame != 0) throw std::runtime_error("Clip loop did not reset sequence state");
        simulationSteps = 0;
        simulateOneStep(false);
        if (sequence.outputHash64() != baselineFirstHash) throw std::runtime_error("CPU reset replay first-frame hash differs");
        if (firstSolvedHash != baselineSolvedHash) {
            throw std::runtime_error("CPU reset replay solved-position hash differs; solver state leaked across the clip loop");
        }
        std::cout << "CPU full-clip verification passed; network hash=0x" << std::hex << baselineFirstHash
                  << ", solved hash=0x" << baselineSolvedHash << std::dec << "\n";
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

    // Staged upload for the read-only topology. These are sampled once per vertex in
    // the normal pass, so keeping them device-local matters on a discrete part even
    // though they are small.
    void createDeviceLocalBuffer(VkBufferUsageFlags usage, vks::Buffer& target, const void* data, VkDeviceSize bytes) {
        vks::Buffer staging;
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &staging, bytes, const_cast<void*>(data)));
        VK_CHECK_RESULT(vulkanDevice->createBuffer(usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &target, bytes));
        vulkanDevice->copyBuffer(&staging, &target, queue);
        staging.destroy();
    }

    void prepareBuffers() {
        const VkQueueFamilyProperties& queueProperties = vulkanDevice->queueFamilyProperties[vulkanDevice->queueFamilyIndices.graphics];
        const bool timestampUsable = queueProperties.timestampValidBits > 0 && deviceProperties.limits.timestampPeriod > 0.0f;
        for (auto& frame : frames) {
            VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &frame.upload, branchPointBytes()));
            VK_CHECK_RESULT(frame.upload.map());
            VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &frame.points, branchPointBytes()));
            VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &frame.transformUniform, sizeof(TransformUniform)));
            VK_CHECK_RESULT(frame.transformUniform.map());
            VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &frame.cameraUniform, sizeof(CameraUniform)));
            VK_CHECK_RESULT(frame.cameraUniform.map());
            if (meshMode) {
                VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &frame.normals, branchPointBytes()));
            }
            if (timestampUsable) {
                VkQueryPoolCreateInfo queryInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
                queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
                queryInfo.queryCount = 2;
                VK_CHECK_RESULT(vkCreateQueryPool(device, &queryInfo, nullptr, &frame.queryPool));
            }
        }
        if (!meshMode) return;
        // The topology is replicated once per branch: the second copy indexes the second vertex
        // block, and its CSR offsets continue past the first block's triangles. Drawing the same
        // topology twice would be wrong in a way that is easy to miss -- the normal pass writes
        // per-vertex normals, so both branches would gather from whichever block was written
        // last and the comparison would show two surfaces lit by one set of normals.
        const uint32_t vertices = meshInfo.vertices;
        const uint32_t triangles = meshInfo.triangles;
        std::vector<uint32_t> indices(size_t(triangles) * 3 * branchCount);
        std::vector<uint32_t> csrOffsets(size_t(vertices) * branchCount + 1);
        std::vector<uint32_t> csrIndices(size_t(triangles) * 3 * branchCount);
        for (uint32_t branch = 0; branch < branchCount; ++branch) {
            const uint32_t vertexBase = branch * vertices;
            const uint32_t triangleBase = branch * triangles;
            for (uint32_t i = 0; i < triangles * 3; ++i) {
                indices[size_t(triangleBase) * 3 + i] = meshInfo.triangleIndices[i] + vertexBase;
                csrIndices[size_t(triangleBase) * 3 + i] = meshInfo.triangleCsrIndices[i] + triangleBase;
            }
            for (uint32_t v = 0; v < vertices; ++v) {
                csrOffsets[size_t(vertexBase) + v] = meshInfo.triangleCsrOffsets[v] + triangleBase * 3;
            }
        }
        csrOffsets.back() = triangles * 3 * branchCount;

        // One buffer serves as both the index buffer for the draw and the storage
        // buffer the normal pass reads, so the two can never drift apart.
        createDeviceLocalBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            meshResources.triangles, indices.data(), indices.size() * sizeof(uint32_t));
        createDeviceLocalBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, meshResources.csrOffsets,
            csrOffsets.data(), csrOffsets.size() * sizeof(uint32_t));
        createDeviceLocalBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, meshResources.csrIndices,
            csrIndices.data(), csrIndices.size() * sizeof(uint32_t));
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &meshResources.uniform, sizeof(MeshUniform), &meshUniform));
        createBodyResources();
    }

    void prepareDescriptors() {
        std::vector<VkDescriptorPoolSize> sizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxConcurrentFrames * 7),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxConcurrentFrames * 3),
        };
        VkDescriptorPoolCreateInfo poolInfo = vks::initializers::descriptorPoolCreateInfo(sizes, maxConcurrentFrames * 3);
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
        if (meshMode) {
            std::vector<VkDescriptorSetLayoutBinding> normalBindings = {
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5),
            };
            VkDescriptorSetLayoutCreateInfo normalLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(normalBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &normalLayoutInfo, nullptr, &normalSetLayout));
        }
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
            if (!meshMode) continue;
            allocate = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &normalSetLayout, 1);
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocate, &frame.normalSet));
            std::array<VkWriteDescriptorSet, 6> normalWrites = {
                vks::initializers::writeDescriptorSet(frame.normalSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &frame.points.descriptor),
                vks::initializers::writeDescriptorSet(frame.normalSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &meshResources.triangles.descriptor),
                vks::initializers::writeDescriptorSet(frame.normalSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &meshResources.csrOffsets.descriptor),
                vks::initializers::writeDescriptorSet(frame.normalSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &meshResources.csrIndices.descriptor),
                vks::initializers::writeDescriptorSet(frame.normalSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &frame.normals.descriptor),
                vks::initializers::writeDescriptorSet(frame.normalSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5, &meshResources.uniform.descriptor),
            };
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(normalWrites.size()), normalWrites.data(), 0, nullptr);
        }
    }

    void preparePipelines() {
        VkPipelineLayoutCreateInfo computeLayoutInfo = vks::initializers::pipelineLayoutCreateInfo(&computeSetLayout, 1);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &computeLayoutInfo, nullptr, &computePipelineLayout));
        VkComputePipelineCreateInfo computeInfo = vks::initializers::computePipelineCreateInfo(computePipelineLayout);
        computeInfo.stage = loadShader(getShadersPath() + "mlclothcpu/point_transform.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
        VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computeInfo, nullptr, &computePipeline));

        if (meshMode) {
            VkPipelineLayoutCreateInfo normalLayoutInfo = vks::initializers::pipelineLayoutCreateInfo(&normalSetLayout, 1);
            VK_CHECK_RESULT(vkCreatePipelineLayout(device, &normalLayoutInfo, nullptr, &normalPipelineLayout));
            VkComputePipelineCreateInfo normalInfo = vks::initializers::computePipelineCreateInfo(normalPipelineLayout);
            normalInfo.stage = loadShader(getShadersPath() + "mlclothcpu/cloth_normals.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
            VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &normalInfo, nullptr, &normalPipeline));
        }

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

        if (!meshMode) return;
        // Same layout, same render pass, same camera set: only the topology, the
        // vertex streams and the two stages differ. Culling stays off so the
        // fragment shader's back-face tint can expose inconsistent winding.
        VkPipelineInputAssemblyStateCreateInfo triangleAssembly = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
        std::array<VkPipelineShaderStageCreateInfo, 2> clothStages = {
            loadShader(getShadersPath() + "mlclothcpu/cloth.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader(getShadersPath() + "mlclothcpu/cloth.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
        };
        const std::array<VkVertexInputBindingDescription, 2> clothBindings = {
            vks::initializers::vertexInputBindingDescription(0, sizeof(glm::vec4), VK_VERTEX_INPUT_RATE_VERTEX),
            vks::initializers::vertexInputBindingDescription(1, sizeof(glm::vec4), VK_VERTEX_INPUT_RATE_VERTEX),
        };
        const std::array<VkVertexInputAttributeDescription, 2> clothAttributes = {
            vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0),
            vks::initializers::vertexInputAttributeDescription(1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0),
        };
        VkPipelineVertexInputStateCreateInfo clothVertexInput = vks::initializers::pipelineVertexInputStateCreateInfo();
        clothVertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(clothBindings.size());
        clothVertexInput.pVertexBindingDescriptions = clothBindings.data();
        clothVertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(clothAttributes.size());
        clothVertexInput.pVertexAttributeDescriptions = clothAttributes.data();
        pipeline.pVertexInputState = &clothVertexInput;
        pipeline.pInputAssemblyState = &triangleAssembly;
        pipeline.stageCount = static_cast<uint32_t>(clothStages.size());
        pipeline.pStages = clothStages.data();
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipeline, nullptr, &clothPipeline));
    }

    void prepare() override {
        try {
            loadModelClipAndRuntime();
            // body buffers need the device, so they are made after the mesh ones
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
        std::memcpy(frame.upload.mapped, latestLocalPoints.data(), static_cast<size_t>(branchPointBytes()));
        updateBodyGeometry(clipFrame == 0 ? 0 : clipFrame - 1);
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
        if (meshMode) {
            // The normal pass gathers neighbours, so it must not start until every
            // world position of this frame has been written.
            VkBufferMemoryBarrier positionBarrier = vks::initializers::bufferMemoryBarrier();
            positionBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            positionBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            positionBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            positionBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            positionBarrier.buffer = frame.points.buffer;
            positionBarrier.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, nullptr, 1, &positionBarrier, 0, nullptr);
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, normalPipeline);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, normalPipelineLayout, 0, 1, &frame.normalSet, 0, nullptr);
            vkCmdDispatch(command, (kVertexCount + 127) / 128, 1, 1);
        }
        if (frame.queryPool) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, frame.queryPool, 1);
        VkBufferMemoryBarrier pointBarrier = vks::initializers::bufferMemoryBarrier();
        pointBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        pointBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        pointBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pointBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pointBarrier.buffer = frame.points.buffer;
        pointBarrier.size = VK_WHOLE_SIZE;
        std::vector<VkBufferMemoryBarrier> vertexBarriers = { pointBarrier };
        if (meshMode) {
            VkBufferMemoryBarrier normalBarrier = pointBarrier;
            normalBarrier.buffer = frame.normals.buffer;
            vertexBarriers.push_back(normalBarrier);
        }
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0,
            0, nullptr, static_cast<uint32_t>(vertexBarriers.size()), vertexBarriers.data(), 0, nullptr);

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
        VkDeviceSize offset = 0;
        if (meshMode) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, clothPipeline);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &frame.graphicsSet, 0, nullptr);
            const std::array<VkBuffer, 2> streams = { frame.points.buffer, frame.normals.buffer };
            const std::array<VkDeviceSize, 2> offsets = { 0, 0 };
            vkCmdBindVertexBuffers(command, 0, static_cast<uint32_t>(streams.size()), streams.data(), offsets.data());
            vkCmdBindIndexBuffer(command, meshResources.triangles.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(command, meshInfo.triangles * 3 * branchCount, 1, 0, 0, 0);
            if (capsulesLoaded && drawBody && body.indexCount != 0) {
                // Same pipeline: it takes a position and a normal stream and a camera, which is
                // all the body needs. Culling is off there, so the capsules read correctly from
                // inside as well -- which is the view that shows a garment sunk into a limb.
                const std::array<VkBuffer, 2> bodyStreams = { body.positions.buffer, body.normals.buffer };
                const std::array<VkDeviceSize, 2> bodyOffsets = { 0, 0 };
                vkCmdBindVertexBuffers(command, 0, static_cast<uint32_t>(bodyStreams.size()), bodyStreams.data(), bodyOffsets.data());
                vkCmdBindIndexBuffer(command, body.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(command, body.indexCount, 1, 0, 0, 0);
            }
        } else {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &frame.graphicsSet, 0, nullptr);
            vkCmdBindVertexBuffers(command, 0, 1, &frame.points.buffer, &offset);
            vkCmdDraw(command, kVertexCount, 1, 0, 0);
        }
        drawUI(command);
        vkCmdEndRenderPass(command);
        VK_CHECK_RESULT(vkEndCommandBuffer(command));
        frame.queryIssued = frame.queryPool != VK_NULL_HANDLE;
    }

    void verifyGpuReadback(FrameResources& frame) {
        if (!verifyMode || gpuVerificationDone) return;
        vks::Buffer readback;
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &readback, branchPointBytes()));
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
        VkBufferCopy copy{ 0, 0, branchPointBytes() };
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
        // The first-frame hash is in here, not only on the console, so the no-regression gate
        // is scriptable: `--xpbd --xpbd-iterations 0` has to reproduce the pure-MNN hash
        // exactly, and a gate that can only be read by a human is a gate that stops running.
        report << std::setprecision(10) << "{\n  \"vertices\": " << kVertexCount
               << ",\n  \"max_abs_m\": " << maxError << ",\n  \"limit_m\": 1e-5,\n  \"passed\": "
               << (maxError <= 1.0e-5 ? "true" : "false")
               << ",\n  \"first_frame_hash\": \"0x" << std::hex << firstFrameHash << std::dec << "\""
               << ",\n  \"first_solved_hash\": \"0x" << std::hex << firstSolvedHash << std::dec << "\""
               << ",\n  \"clip\": \"" << clipPath.stem().string() << "\""
               << ",\n  \"clips_available\": " << availableClipNames.size()
               << ",\n  \"frame_step\": " << frameStep
               << ",\n  \"xpbd\": " << (xpbdEnabled ? "true" : "false")
               << ",\n  \"xpbd_iterations\": " << (xpbdEnabled ? xpbdConfig.iterations : 0)
               << ",\n  \"xpbd_solve_ms\": " << lastSolveMs
               << ",\n  \"xpbd_stretch_active\": " << xpbdSolver.lastStep().stretchActive
               << ",\n  \"xpbd_area_active\": " << xpbdSolver.lastStep().areaFloorsActive
               << ",\n  \"xpbd_max_correction_m\": " << xpbdSolver.lastStep().maxCorrectionM
               << ",\n  \"xpbd_contacts_resolved\": " << xpbdSolver.lastStep().contactsResolved
               << ",\n  \"capsules\": " << (capsulesLoaded ? capsuleInfo.count : 0u)
               << ",\n  \"compare\": " << (compareMode ? "true" : "false")
               << ",\n  \"penetration_pure_fraction\": " << purePenetration.fraction
               << ",\n  \"penetration_pure_deepest_m\": " << purePenetration.deepestM
               << ",\n  \"penetration_solved_fraction\": " << solvedPenetration.fraction
               << ",\n  \"penetration_solved_deepest_m\": " << solvedPenetration.deepestM
               << ",\n  \"penetration_per_component\": [";
        for (uint32_t component = 0; component < componentCount; ++component) {
            report << (component ? ", " : "") << "{\"pure\": " << purePerComponent[component].fraction
                   << ", \"solved\": " << solvedPerComponent[component].fraction << "}";
        }
        report << "]"
               << "\n}\n";
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
        stream << '"' << deviceProperties.deviceName << "\"," << threads << ',' << kVertexCount << ',' << branchPointBytes() << ','
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
        overlay->text("Points: %u, upload: %llu bytes", kVertexCount, static_cast<unsigned long long>(branchPointBytes()));
        if (meshMode) {
            overlay->text("Mesh: %u triangles, %u edges, %u pinned", meshInfo.triangles, meshInfo.edges, meshInfo.pinnedVertices);
        } else {
            overlay->text("Mesh: none (point cloud); pass --mesh for triangles");
        }
        overlay->text("CPU inference last: %.3f ms", lastInferenceMs);
        if (!recentInferenceSamples.empty()) overlay->text("CPU inference median/p95 (recent): %.3f / %.3f ms", percentile(recentInferenceSamples, 0.5), percentile(recentInferenceSamples, 0.95));
        overlay->text("GPU transform: %.4f ms", lastGpuTransformMs);
        overlay->text("Dropped simulation steps: %llu", static_cast<unsigned long long>(droppedSteps));
        overlay->text("Status: %s", errorStatus.c_str());
        if (compareMode) overlay->text("Left: network only    Right: network + XPBD");
        if (capsulesLoaded && drawBody) overlay->checkBox("Draw body capsules", &drawBody);
        overlay->text("P: pause  R: deterministic reset  Esc: exit");
        if (overlay->checkBox("Paused", &simulationPaused)) accumulatorSeconds = 0.0;
        if (overlay->button("Reset")) resetRequested = true;
        if (availableClipNames.size() > 1 && overlay->comboBox("Animation", &selectedClip, availableClipNames)) {
            const auto& chosen = availableClipPaths[static_cast<size_t>(selectedClip)];
            if (chosen != clipPath && !loadClip(chosen)) {
                // Put the selector back on the clip that is actually playing, so the overlay
                // never shows a name whose clip was refused.
                const auto found = std::find(availableClipPaths.begin(), availableClipPaths.end(), clipPath);
                selectedClip = found == availableClipPaths.end()
                    ? 0 : static_cast<int32_t>(std::distance(availableClipPaths.begin(), found));
            }
        }
        if (clipSwitchStatus != "OK") overlay->text("Clip: %s", clipSwitchStatus.c_str());
        // The interactive form of the probe's frame-scale axis. The timestep is not scaled.
        overlay->sliderInt("Playback speed (x)", &frameStep, 1, 4);
        overlay->checkBox("Hold last frame (see the settle)", &holdLastFrame);
        if (xpbdEnabled && meshMode) {
            if (capsulesLoaded) {
                overlay->text("Inside body: network %.2f%% (%.1f cm deep) -> hybrid %.2f%% (%.1f cm)",
                    purePenetration.fraction * 100.0f, -purePenetration.deepestM * 100.0f,
                    solvedPenetration.fraction * 100.0f, -solvedPenetration.deepestM * 100.0f);
                overlay->text("  ragdoll envelope, so a fitted piece belongs inside. ChaosCloth's own");
                overlay->text("  run reads 69 / 98 / 14 / 100%% per piece -- only the skirt's is reducible.");
                for (uint32_t component = 0; component < componentCount; ++component) {
                    overlay->text("  piece %u: network %.1f%% -> hybrid %.1f%%", component,
                        purePerComponent[component].fraction * 100.0f,
                        solvedPerComponent[component].fraction * 100.0f);
                }
            }
            overlay->text("XPBD: %.3f ms, %u stretch, %u area, max %.1f mm",
                lastSolveMs, xpbdSolver.lastStep().stretchActive, xpbdSolver.lastStep().areaFloorsActive,
                xpbdSolver.lastStep().maxCorrectionM * 1000.0f);
            if (overlay->sliderInt("XPBD iterations", &xpbdConfig.iterations, 0, 64)) xpbdSolver.configure(xpbdConfig);
            if (overlay->sliderFloat("Stretch compliance", &xpbdConfig.stretchCompliance, 0.0f, 0.2f)) xpbdSolver.configure(xpbdConfig);
            if (overlay->sliderFloat("Bend compliance", &xpbdConfig.bendCompliance, 0.0f, 0.5f)) xpbdSolver.configure(xpbdConfig);
            overlay->sliderFloat("Area floor (x target)", &xpbdConfig.areaFloor, 0.0f, 1.5f);
            overlay->sliderFloat("Guide compliance (<0 off)", &xpbdConfig.guideCompliance, -1.0f, 5.0f);
            overlay->sliderFloat("Guide trust ratio (0 off)", &xpbdConfig.guideTrustRatio, 0.0f, 4.0f);
        }
    }
};

VULKAN_EXAMPLE_MAIN()
