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
    // Ceiling on how far the measured rigid pin bind may sit from the network's own prediction
    // for the same pinned vertices, in metres. A rigid bind approximates smooth skinning and the
    // network is free to deviate from both, so centimetres are expected: over 15 clips spanning
    // run, sprint, dodge, turn, fall, attack and death the per-clip worst single vertex ranges
    // 6.3 to 14.5 cm and the per-frame mean is 1.1 to 1.7 cm. This sits 2.4x above that and well
    // under the scale of the failures it is for -- a bind taken in the wrong bone frame, or with
    // the up axis guessed, separates the two by a large fraction of the garment's 1.2 m extent.
    static constexpr float kPinBindGapLimitM = 0.35f;
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

    // Mirrors `CameraParams` in scene_camera.hlsli. The inverse is composed here because HLSL
    // has no matrix inverse and the sky pass has to turn a pixel back into a world ray through
    // exactly the transform the geometry went forward through -- anything reconstructed from the
    // parts would put the horizon slightly off the geometry under some cameras and not others.
    struct CameraUniform {
        glm::mat4 projection{};
        glm::mat4 view{};
        glm::mat4 inverseViewProjection{};
        glm::vec4 cameraPositionM{};     // world metres; .w unused
    } cameraUniform;
    static_assert(sizeof(CameraUniform) == 208);

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
    VkPipeline skyPipeline{ VK_NULL_HANDLE };
    // The backdrop is a reading aid, so it has an off switch: against a flat dark field it is
    // easier to say whether a dark patch of garment is shading or geometry, and a screenshot for
    // a document sometimes wants nothing behind the mesh.
    bool skyEnabled{ true };

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
    std::vector<uint32_t> xpbdPairs;
    uint32_t xpbdBendStart{};
    std::vector<float> xpbdTargetLength;
    std::vector<float> xpbdTargetArea;

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
    // The clip frame the most recent step ran, which is what the body overlay has to use.
    uint32_t renderFrame{};
    std::string clipSwitchStatus{ "OK" };
    std::string collisionPieces;
    std::vector<uint8_t> collisionMaskHost;

    // Side-by-side comparison of the three architectures, the same three the sibling PoC's
    // `--hood-compare` draws and the same tints, so a screenshot from either reads the same
    // way: A the network alone, B the constraints alone with no network at all, C both.
    // Everything else -- garment, topology, calibration, animation, frame, camera -- is
    // shared, so the only difference on screen is the solver.
    //
    // A and C share one inference by definition, so the mode costs one extra solve for C.
    // B costs a second solve and no inference, which is also the honest cost comparison:
    // the network is 2.558 ms of CPU that B does not spend, and `--xpbd-iterations-b` is
    // what that buys back in sweeps.
    bool compareMode{};
    float compareSpacingCm{ 90.0f };
    enum Branch : uint32_t { kBranchNetwork = 0, kBranchConstraints = 1, kBranchHybrid = 2 };
    static constexpr uint32_t kMaxBranches = 3;
    static constexpr const char* kBranchNames[kMaxBranches]{
        "A network only", "B constraints only", "C hybrid" };
    // Blue A / orange B / green C, byte for byte the sibling PoC's `hoodBranchTints`.
    static constexpr float kBranchTints[kMaxBranches][3]{
        { 0.12f, 0.42f, 0.88f }, { 0.95f, 0.55f, 0.15f }, { 0.20f, 0.75f, 0.35f } };
    uint32_t branchCount{ 1 };
    int branchIterationsB{ 15 };
    // Which architecture each drawn slot holds. In comparison mode that is the identity; in
    // single-branch mode the one slot is the hybrid when constraints are on and the network
    // otherwise, so every per-branch table below is indexed by *role* and never by slot.
    std::array<uint32_t, kMaxBranches> roleOfSlot{ { kBranchNetwork, kBranchConstraints, kBranchHybrid } };
    bool branchVisible[kMaxBranches]{ true, true, true };

    // Everything a solved branch owns. Two branches solve, and they cannot share history:
    // B's trajectory diverges from C's on the first frame and never returns, which is the
    // whole point of running it.
    struct Penetration {
        uint32_t inside{};
        float deepestM{};
        float fraction{};
    };
    struct BranchState {
        mlcloth::XpbdSolver solver;
        std::vector<float> startM;       // what the solve begins from, metres, bone-local
        std::vector<float> solvedM;
        std::vector<float> solvedCm;     // what buildPointData consumes
        std::vector<float> previousM;    // for the Verlet extrapolation
        std::vector<float> olderM;
        std::vector<float> inertialM;
        uint32_t solvedFrames{};
        double solveMs{};
        int iterations{ 8 };
        Penetration penetration{};
        std::vector<Penetration> perComponent;
    };
    std::array<BranchState, kMaxBranches> branches;
    // Positions each slot's role produced this step, centimetres, bone-local. Role A's entry
    // points straight at the network output with no copy and no unit round trip.
    std::array<const float*, kMaxBranches> roleCm{};
    // Agreement between the measured pin bind and the network, at the pinned vertices. See
    // `measurePinBindGap`.
    float pinBindGapMaxM{};
    float pinBindGapMeanM{};
    float worstPinBindGapM{};
    // Body checks, filled by the verification pass. See `verifyBodySkin`.
    float bodySkinGapM{};
    float garmentSkinP05Cm{};
    float garmentSkinP50Cm{};
    uint32_t garmentSkinMatched{};
    // Backdrop checks. See `verifyCameraRays`.
    float cameraRayNdcError{};
    float cameraRayWorstFacing{};

    // Body collision geometry. Capsules are bone-local, so the clip's own 45 driver transforms
    // place them: no new per-frame data and no runtime skinning. They are also exactly the
    // geometry the penetration criterion uses, which is why they remain what the numbers are
    // measured against -- but not what gets drawn by default. A ragdoll envelope is several
    // centimetres wider than the skin, so judging a garment by eye against it is misleading in
    // both directions, and `--body mesh` draws the character's own render mesh instead.
    std::filesystem::path capsulePath;
    std::vector<uint8_t> capsuleBytes;
    mlcloth::CapsuleInfo capsuleInfo{};
    bool capsulesLoaded{};
    enum class BodyDisplay { None, Capsules, Mesh };
    BodyDisplay bodyDisplay{ BodyDisplay::Mesh };
    std::vector<mlcloth::Capsule> placedCapsulesM;      // reference-bone local, metres

    // The character's render mesh, skinned on the GPU from the same 45 driver transforms that
    // place the capsules. 67,857 vertices at five influences is nothing next to a 2.6 ms
    // inference, and the 45 skin matrices are composed on the host once per frame.
    std::filesystem::path bodyMeshPath;
    std::vector<uint8_t> bodyMeshBytes;
    mlcloth::BodyInfo bodyInfo{};
    bool bodyMeshLoaded{};
    struct SkinnedBody {
        vks::Buffer restPositions, restNormals, boneIndices, boneWeights, indices;
        vks::Buffer positions, normals;      // world metres, written by body_skin.comp
        vks::Buffer skinMatrices, uniform;   // host visible, rewritten every frame
        uint32_t indexCount{};
    } skin;
    std::vector<float> skinMatricesHost;     // drivers * 12, row-major 3x4
    struct BodyUniform {
        uint32_t vertexCount{};
        uint32_t influences{};
        uint32_t reserved0{};
        uint32_t reserved1{};
    } bodyUniform{};
    VkDescriptorSetLayout skinSetLayout{ VK_NULL_HANDLE };
    VkDescriptorSet skinSet{ VK_NULL_HANDLE };
    VkPipelineLayout skinPipelineLayout{ VK_NULL_HANDLE };
    VkPipeline skinPipeline{ VK_NULL_HANDLE };
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

    // Penetration against the capsules, per branch. Reported per branch because the absolute
    // number is not the interesting one: the capsules are a ragdoll envelope larger than the
    // skin, and ChaosCloth's own settled T-pose reads 1.42% of vertices inside them at 6.7 cm
    // deep. Only the comparison between branches means anything, which is exactly what the
    // side-by-side view is for.
    //
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
        // The gradient sky over the gridded floor. On by default because without it there is no
        // cue for how high a hem sits or whether the side-by-side branches stand at the same
        // height; off for a screenshot that wants nothing behind the mesh.
        skyEnabled = !hasArgument("--no-sky");
        // Zero iterations is the no-regression arm: the solver returns the network output
        // byte for byte, so --xpbd --xpbd-iterations 0 must reproduce the pure-MNN hash.
        compareMode = hasArgument("--compare");
        compareSpacingCm = std::stof(argumentValue("--compare-spacing-cm", "90"));
        // The speed axis, reachable from a script and not only from the overlay slider: the
        // body advances this many clip frames per solver step and the timestep is deliberately
        // NOT scaled with it, because the question is what happens when the body moves faster
        // per step and scaling the timestep would cancel exactly that.
        frameStep = std::max(1, std::stoi(argumentValue("--frame-step", "1")));
        holdLastFrame = hasArgument("--hold-last-frame");
        capsulePath = argumentValue("--capsules", "");
        bodyMeshPath = argumentValue("--body-mesh", "");
        // What stands in for the character. `mesh` is the skinned render body, `capsules` the
        // collision envelope, `none` neither. The default is the mesh when its asset is there,
        // because the envelope is not the surface the garment is meant to lie on.
        {
            const std::string mode = argumentValue("--body", hasArgument("--no-body") ? "none" : "mesh");
            if (mode == "none") bodyDisplay = BodyDisplay::None;
            else if (mode == "capsules") bodyDisplay = BodyDisplay::Capsules;
            else if (mode == "mesh") bodyDisplay = BodyDisplay::Mesh;
            else throw std::runtime_error("--body must be mesh, capsules or none");
        }
        // Which garment pieces get contacts, as a comma-separated list of largest-first piece
        // indices. Empty means all, which measurement shows is wrong for this garment; "2" is the
        // skirt, the one piece whose penetration is actually reducible.
        collisionPieces = argumentValue("--collision-pieces", "");
        xpbdEnabled = hasArgument("--xpbd") || compareMode;
        xpbdConfig.iterations = std::stoi(argumentValue("--xpbd-iterations", "8"));
        // Equal CPU budget, not equal iterations. Measured on this machine at 5,294 vertices:
        // C costs 2.558 ms of MNN inference plus 2.57 ms for 8 iterations, so 5.13 ms; B runs at
        // 0.341 ms per iteration and buys 15 of them for the same money. Comparing at equal
        // iterations instead would hand B a third of C's compute and then report the result as
        // an architecture difference. Re-derive it from `-Benchmark` and the per-branch solve_ms
        // in `mlcloth_verify.json` if this moves to another machine; it is a measurement.
        branchIterationsB = std::stoi(argumentValue("--xpbd-iterations-b", "15"));
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
        // Three branches only make sense with topology: the comparison is about whether the
        // constraints keep the surface off the body, which a point cloud cannot show.
        if (compareMode && !meshMode) throw std::runtime_error("--compare needs --mesh");
        branchCount = compareMode ? kMaxBranches : 1u;
        // A point cloud has no topology, so there is no solve to draw even when `--xpbd` was
        // asked for; the one slot is then the network's own output.
        if (!compareMode) roleOfSlot[0] = (xpbdEnabled && meshMode) ? kBranchHybrid : kBranchNetwork;
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
        skin.restPositions.destroy();
        skin.restNormals.destroy();
        skin.boneIndices.destroy();
        skin.boneWeights.destroy();
        skin.indices.destroy();
        skin.positions.destroy();
        skin.normals.destroy();
        skin.skinMatrices.destroy();
        skin.uniform.destroy();
        if (skinPipeline) vkDestroyPipeline(device, skinPipeline, nullptr);
        if (skinPipelineLayout) vkDestroyPipelineLayout(device, skinPipelineLayout, nullptr);
        if (skinSetLayout) vkDestroyDescriptorSetLayout(device, skinSetLayout, nullptr);
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipeline(device, normalPipeline, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipeline(device, clothPipeline, nullptr);
        vkDestroyPipeline(device, skyPipeline, nullptr);
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
            // The constraints-only branch has no network to take its pin target from, so it
            // needs the measured bind. Refused up front rather than quietly falling back to the
            // rest pose: a static anchor would make B lose for a reason that is not the solver.
            if (compareMode && !meshInfo.pinDriverIndices) {
                throw std::runtime_error(
                    "--compare needs the pin bind sections, which this mesh does not carry. Re-bake "
                    "with reference clips: pwsh ./bake_cloth_topology.ps1");
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
            loadBodyMesh(modelHash);
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

    // Where a drawn slot stands, in reference-bone-local centimetres. Symmetric about zero, so
    // one branch alone is centred and three stand left / middle / right.
    float branchOffsetCm(uint32_t slot) const {
        if (branchCount <= 1) return 0.0f;
        return (static_cast<float>(slot) - 0.5f * static_cast<float>(branchCount - 1)) * compareSpacingCm;
    }

    // A branch's hue. The single-branch path gets A's blue, so its picture is unchanged.
    glm::vec4 tintOf(uint32_t role) const {
        const uint32_t index = branchCount > 1 ? role : kBranchNetwork;
        return glm::vec4(kBranchTints[index][0], kBranchTints[index][1], kBranchTints[index][2], 1.0f);
    }

    // What B's solve actually costs relative to C's whole step, inference included. The
    // comparison rests on equal CPU budget rather than equal iterations, and `--xpbd-iterations-b`
    // is a *measurement* made in one configuration -- so the achieved ratio is reported rather
    // than assumed. Zero when there is nothing to compare.
    double compareBudgetRatio() const {
        if (!compareMode) return 0.0;
        const double hybrid = lastInferenceMs + branches[kBranchHybrid].solveMs;
        if (hybrid <= 0.0) return 0.0;
        return branches[kBranchConstraints].solveMs / hybrid;
    }

    // Mirrors `Instance` in cloth.vert / cloth.frag; keep the three in step.
    struct InstancePush {
        glm::vec4 offset{};
        glm::vec4 tint{};
    };

    void pushInstance(VkCommandBuffer command, const glm::vec3& offset, const glm::vec4& tint) const {
        const InstancePush push{ glm::vec4(offset, 0.0f), tint };
        vkCmdPushConstants(command, graphicsPipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    }

    // Where a drawn slot's body copy stands, in world metres. The cloth's own offset is baked
    // into its positions in reference-bone-local coordinates, so this is the same displacement
    // expressed the only way a single skinned body can use it: the local Y direction for the
    // frame, mapped through the same component-to-world swap, scaled by the lateral distance.
    glm::vec3 branchWorldOffsetM(uint32_t slot, uint32_t frame) const {
        const float lateral = branchOffsetCm(slot);
        if (lateral == 0.0f) return glm::vec3(0.0f);
        const glm::vec3 right = glm::normalize(rootRight(frame));
        return glm::vec3(right.x, right.z, -right.y) * (lateral * 0.01f);
    }

    // Writes one vertex block per drawn slot from that slot's own role. The lateral offset is
    // applied in *reference-bone-local* coordinates, before the transform, so it is a rigid
    // sideways shift of the whole branch and not a change to any constraint.
    void buildPointData(uint32_t frame) {
        glm::vec3 up = glm::normalize(rootUp(frame));
        glm::vec3 right = glm::normalize(rootRight(frame));
        glm::vec3 forward = glm::normalize(glm::cross(right, up));
        const glm::vec3 root = rootPosition(frame);
        glm::vec3 minimum(std::numeric_limits<float>::max());
        glm::vec3 maximum(std::numeric_limits<float>::lowest());
        for (uint32_t slot = 0; slot < branchCount; ++slot) {
            const float* source = roleCm[roleOfSlot[slot]];
            if (source == nullptr) throw std::runtime_error("A drawn branch produced no positions");
            const float lateral = branchOffsetCm(slot);
            for (uint32_t vertex = 0; vertex < kVertexCount; ++vertex) {
                const uint32_t index = slot * kVertexCount + vertex;
                const glm::vec3 local(source[vertex * 3], source[vertex * 3 + 1] + lateral, source[vertex * 3 + 2]);
                latestLocalPoints[index] = glm::vec4(local, 1.0f);
                const glm::vec3 component = root + forward * local.x + right * local.y + up * local.z;
                const glm::vec3 world(component.x, component.z, -component.y);
                const glm::vec3 metres = world * 0.01f;
                if (!std::isfinite(metres.x) || !std::isfinite(metres.y) || !std::isfinite(metres.z)) {
                    throw std::runtime_error("CPU coordinate reference contains NaN or Inf");
                }
                latestWorldPoints[index] = glm::vec4(metres, 1.0f);
                minimum = glm::min(minimum, metres);
                maximum = glm::max(maximum, metres);
            }
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
    // One skinned copy of the body, shared by every comparison branch: the branch offset is a
    // rigid world translation the vertex shader applies per draw, so skinning it three times
    // would produce three identical buffers.
    void createSkinnedBodyResources() {
        // Built whenever the asset loaded, for the same reason the capsules are: the overlay
        // switches between the two forms, and allocating on that switch would mean creating
        // buffers while a frame is in flight. The dispatch, not the allocation, is what the
        // display mode gates.
        if (!bodyMeshLoaded || bodyDisplay == BodyDisplay::None) return;
        const VkDeviceSize vertexBytes = VkDeviceSize(bodyInfo.vertices) * sizeof(glm::vec4);
        // The rest arrays are float3 in the asset and float4 in the shader, because a
        // StructuredBuffer<float3> has a stride the host and the compiler can disagree about.
        std::vector<glm::vec4> restPositions(bodyInfo.vertices), restNormals(bodyInfo.vertices);
        for (uint32_t v = 0; v < bodyInfo.vertices; ++v) {
            const float* p = bodyInfo.restPositionsCm + size_t(v) * 3;
            const float* n = bodyInfo.restNormals + size_t(v) * 3;
            restPositions[v] = glm::vec4(p[0], p[1], p[2], 1.0f);
            restNormals[v] = glm::vec4(n[0], n[1], n[2], 0.0f);
        }
        const size_t influenceCount = size_t(bodyInfo.vertices) * bodyInfo.influences;
        createDeviceLocalBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, skin.restPositions,
            restPositions.data(), restPositions.size() * sizeof(glm::vec4));
        createDeviceLocalBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, skin.restNormals,
            restNormals.data(), restNormals.size() * sizeof(glm::vec4));
        createDeviceLocalBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, skin.boneIndices,
            bodyInfo.boneIndices, influenceCount * sizeof(uint32_t));
        createDeviceLocalBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, skin.boneWeights,
            bodyInfo.boneWeights, influenceCount * sizeof(float));
        createDeviceLocalBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, skin.indices,
            bodyInfo.triangleIndices, size_t(bodyInfo.triangles) * 3 * sizeof(uint32_t));
        skin.indexCount = bodyInfo.triangles * 3;
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &skin.positions, vertexBytes));
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &skin.normals, vertexBytes));
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &skin.skinMatrices, VkDeviceSize(skinMatricesHost.size() * sizeof(float))));
        VK_CHECK_RESULT(skin.skinMatrices.map());
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &skin.uniform, sizeof(BodyUniform), &bodyUniform));
        std::cout << "Body: skinned mesh, " << bodyInfo.triangles << " triangles from "
                  << bodyInfo.drivers << " driver bones\n";
    }

    void createBodyResources() {
        createSkinnedBodyResources();
        // Built whenever the capsules loaded, not only when they are the display mode, so the
        // overlay can switch between the two forms without recreating buffers mid-frame. It is
        // 14 capsules of ~300 vertices; the memory is not worth a conditional.
        if (!capsulesLoaded || bodyDisplay == BodyDisplay::None) return;
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
        if (!capsulesLoaded || bodyDisplay != BodyDisplay::Capsules || body.vertexCount == 0) return;
        const glm::vec3 up = glm::normalize(rootUp(frame));
        const glm::vec3 right = glm::normalize(rootRight(frame));
        const glm::vec3 forward = glm::normalize(glm::cross(right, up));
        const glm::vec3 root = rootPosition(frame);
        const uint32_t perCapsule = static_cast<uint32_t>(capsuleTemplate.size());

        for (uint32_t branch = 0; branch < branchCount; ++branch) {
            const float lateral = branchOffsetCm(branch);
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

    // The character's own render mesh, skinnable from this clip and nothing else. Optional:
    // without it `--body mesh` falls back to the capsules and says so, because a PoC checkout
    // without the sibling's character export should still run.
    void loadBodyMesh(const mlcloth::Sha256Digest& modelHash) {
        auto path = bodyMeshPath;
        if (path.empty()) {
            const auto fallback = meshPath.parent_path().parent_path() / "body" / "ch10032_body.mlbody";
            if (std::filesystem::exists(fallback)) path = fallback;
        }
        if (path.empty() || !std::filesystem::exists(path)) {
            if (bodyDisplay == BodyDisplay::Mesh) {
                std::cout << "Body mesh: none, falling back to the capsules. Bake it with "
                             "tools/bake_mlcloth_body.py, or pass --body capsules to stop asking.\n";
                bodyDisplay = BodyDisplay::Capsules;
            }
            return;
        }
        bodyMeshBytes = readFile(path);
        std::string parseError;
        if (!mlcloth::parse_body(bodyMeshBytes.data(), bodyMeshBytes.size(), modelHash, bodyInfo, parseError)) {
            throw std::runtime_error("MLBDY001 validation failed: " + parseError);
        }
        if (bodyInfo.drivers != mlcloth::kDriverCount) {
            throw std::runtime_error("Body driver count does not match the clip");
        }
        bodyMeshLoaded = true;
        skinMatricesHost.assign(size_t(bodyInfo.drivers) * 12, 0.0f);
        bodyUniform.vertexCount = bodyInfo.vertices;
        bodyUniform.influences = bodyInfo.influences;
        std::cout << "Body mesh: " << bodyInfo.vertices << " vertices, " << bodyInfo.triangles
                  << " triangles, " << bodyInfo.influences << " influences";
        if (bodyInfo.foldedBones != 0) {
            std::cout << ", " << bodyInfo.foldedBones
                      << " bone(s) folded onto a driven parent (see the bake report)";
        }
        std::cout << "\n";
    }

    // `pose * inverseBind` per driver slot, in component centimetres. 45 matrices, so the host
    // is the right place for it: the alternative is uploading 45 quaternions and duplicating
    // this composition in the shader, where it would run once per vertex instead of once.
    void updateSkinMatrices(uint32_t frame) {
        if (!bodyMeshLoaded) return;
        for (uint32_t slot = 0; slot < bodyInfo.drivers; ++slot) {
            glm::vec3 ex, ey, ez;
            boneBasis(frame, slot, ex, ey, ez);
            const glm::vec3 translation = bonePosition(frame, slot);
            const float* inverse = bodyInfo.inverseBind + size_t(slot) * 12;
            // Rows of the pose rotation are the basis vectors' components, so row r of
            // `pose * inverse` is pose_row_r dotted with each column of `inverse`.
            const glm::vec3 poseRow[3] = {
                glm::vec3(ex.x, ey.x, ez.x),
                glm::vec3(ex.y, ey.y, ez.y),
                glm::vec3(ex.z, ey.z, ez.z),
            };
            for (uint32_t row = 0; row < 3; ++row) {
                for (uint32_t column = 0; column < 3; ++column) {
                    skinMatricesHost[size_t(slot) * 12 + row * 4 + column] =
                        poseRow[row].x * inverse[0 * 4 + column] +
                        poseRow[row].y * inverse[1 * 4 + column] +
                        poseRow[row].z * inverse[2 * 4 + column];
                }
                skinMatricesHost[size_t(slot) * 12 + row * 4 + 3] =
                    poseRow[row].x * inverse[0 * 4 + 3] +
                    poseRow[row].y * inverse[1 * 4 + 3] +
                    poseRow[row].z * inverse[2 * 4 + 3] +
                    translation[static_cast<int>(row)];
            }
        }
        if (skin.skinMatrices.mapped) {
            std::memcpy(skin.skinMatrices.mapped, skinMatricesHost.data(), skinMatricesHost.size() * sizeof(float));
        }
    }

    // Skinning one vertex on the host, from the same matrices the compute pass reads. Used by
    // the verification pass as the oracle for the GPU skin and by the registration gate, and
    // deliberately not by the render path: at 67,857 vertices it is milliseconds, which is the
    // whole reason the skinning is a compute pass.
    glm::vec3 skinVertexWorldM(uint32_t vertex) const {
        glm::vec3 component(0.0f);
        for (uint32_t k = 0; k < bodyInfo.influences; ++k) {
            const size_t entry = size_t(vertex) * bodyInfo.influences + k;
            const float weight = bodyInfo.boneWeights[entry];
            if (weight <= 0.0f) continue;
            const float* m = skinMatricesHost.data() + size_t(bodyInfo.boneIndices[entry]) * 12;
            const float* rest = bodyInfo.restPositionsCm + size_t(vertex) * 3;
            for (uint32_t row = 0; row < 3; ++row) {
                component[static_cast<int>(row)] += weight *
                    (m[row * 4 + 0] * rest[0] + m[row * 4 + 1] * rest[1] + m[row * 4 + 2] * rest[2] + m[row * 4 + 3]);
            }
        }
        return glm::vec3(component.x, component.z, -component.y) * 0.01f;
    }

    void loadCapsules(const mlcloth::Sha256Digest& modelHash) {        auto path = capsulePath;
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

    // Which architectures actually run a solve. Role A never does, by definition.
    bool roleSolves(uint32_t role) const { return role != kBranchNetwork; }
    // Whether a role occupies a drawn slot. In single-branch mode two of the three do not.
    bool roleDrawn(uint32_t role) const {
        for (uint32_t slot = 0; slot < branchCount; ++slot) if (roleOfSlot[slot] == role) return true;
        return false;
    }
    // Role B is the one that has no network anywhere in it.
    bool roleUsesNetwork(uint32_t role) const { return role != kBranchConstraints; }
    int roleIterations(uint32_t role) const {
        return role == kBranchConstraints ? branchIterationsB : xpbdConfig.iterations;
    }
    // The slot whose positions the no-regression hash and the single-branch path read: the
    // hybrid in comparison mode, the one configured role otherwise.
    uint32_t primaryRole() const { return roleOfSlot[branchCount - 1]; }

    // A branch's own configuration. Only two things differ from the shared one, and both are
    // forced rather than exposed: B has no network, so a guide target and a trust gate that
    // measures distance to one are not things it can have.
    mlcloth::XpbdConfig roleConfig(uint32_t role) const {
        mlcloth::XpbdConfig config = xpbdConfig;
        config.iterations = roleIterations(role);
        if (!roleUsesNetwork(role)) {
            config.guideCompliance = -1.0f;
            config.guideTrustRatio = 0.0f;
        }
        return config;
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

        // One solver per solving role, built from the same tables. They cannot share one: the
        // warm-start multipliers and the two-frame history belong to a trajectory, and B's
        // trajectory leaves C's on the first frame.
        const size_t values = size_t(vertices) * 3;
        // Every role allocates, not only the drawn ones. Role A is measured in metres against
        // the same capsules whether or not it is on screen, so its converted copy is needed in
        // single-branch mode too -- where it is not a slot, and where writing into an
        // unallocated vector is what this loop used to do.
        for (uint32_t role = 0; role < kMaxBranches; ++role) {
            BranchState& state = branches[role];
            state.startM.assign(values, 0.0f);
            state.solvedM.assign(values, 0.0f);
            state.solvedCm.assign(values, 0.0f);
            state.previousM.assign(values, 0.0f);
            state.olderM.assign(values, 0.0f);
            state.inertialM.assign(values, 0.0f);
            state.solvedFrames = 0;
            if (!roleSolves(role) || !roleDrawn(role)) continue;
            std::string err;
            if (!state.solver.build(vertices, xpbdPairs.data(), constraints, xpbdBendStart,
                                    meshInfo.triangleIndices, meshInfo.triangles,
                                    meshInfo.vertexMassKg, meshInfo.pinMask,
                                    xpbdTargetLength.data(), xpbdTargetArea.data(), err)) {
                throw std::runtime_error("XPBD solver build failed: " + err);
            }
            state.iterations = roleIterations(role);
            state.solver.configure(roleConfig(role));
        }
        std::cout << "XPBD: " << constraints << " constraints (" << xpbdBendStart << " stretch, "
                  << (constraints - xpbdBendStart) << " bend), " << meshInfo.triangles
                  << " area floors, " << xpbdConfig.iterations << " iterations\n";
        if (compareMode) {
            std::cout << "  branch B runs " << branchIterationsB
                      << " iterations and no inference, which is the equal-CPU-budget arm\n";
        }
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
        for (BranchState& state : branches) state.perComponent.assign(componentCount, Penetration{});
        std::cout << "Surface components: " << componentCount << "\n";
    }

    // Restricts contacts to the listed pieces. Left empty every vertex takes contacts, which is
    // reported rather than silently accepted because it is measurably the wrong default here.
    void applyCollisionPieces() {
        if (!xpbdEnabled || componentOfVertex.empty()) return;
        // Applied to every solving branch, or the comparison would be between two solvers with
        // different contact sets and would say nothing about the architectures.
        auto pushMask = [&](const uint8_t* mask) {
            for (uint32_t slot = 0; slot < branchCount; ++slot) {
                const uint32_t role = roleOfSlot[slot];
                if (roleSolves(role)) branches[role].solver.set_collision_mask(mask);
            }
        };
        if (collisionPieces.empty()) {
            std::cout << "Collision applies to every piece, which measurement says is wrong here: "
                         "against ChaosCloth's own run this lifts the fitted pieces off the body. "
                         "Pass --collision-pieces 2 to restrict contacts to the skirt.\n";
            pushMask(nullptr);
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
        pushMask(collisionMaskHost.data());
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

    // Gravity in reference-bone-local metres for this frame. Component +Z is up -- the export
    // drives a PoseableMeshComponent and the reference transform's own translation is the pelvis
    // height in that space -- so world down is rotated into the root's frame rather than assumed
    // to be an axis of it. In this asset the root's local X is what points up, so a hardcoded
    // -Z would have pushed the garment sideways.
    glm::vec3 localGravity(uint32_t frame) const {
        glm::vec3 ex, ey, ez;
        boneBasis(frame, mlcloth::kRootDriverIndex, ex, ey, ez);
        const glm::vec3 componentGravity(0.0f, 0.0f, -9.81f);
        return glm::vec3(glm::dot(componentGravity, ex), glm::dot(componentGravity, ey),
                         glm::dot(componentGravity, ez));
    }

    // Where the pinned vertices of a network-free branch belong this frame: each rides the
    // driver bone the bake measured it against, at the mean offset measured in that bone's own
    // frame. Same arithmetic as `placeCapsules`, which is the point -- the anchor and the
    // collision geometry are placed by one construction, so they cannot drift apart.
    void applyPinTargets(uint32_t frame, float* positionsM) const {
        glm::vec3 rx, ry, rz;
        boneBasis(frame, mlcloth::kRootDriverIndex, rx, ry, rz);
        const glm::vec3 rootPos = bonePosition(frame, mlcloth::kRootDriverIndex);
        for (uint32_t vertex = 0; vertex < kVertexCount; ++vertex) {
            const uint32_t driver = meshInfo.pinDriverIndices[vertex];
            if (driver == mlcloth::kNoDriver) continue;
            glm::vec3 bx, by, bz;
            boneBasis(frame, driver, bx, by, bz);
            const glm::vec3 offset(meshInfo.pinLocalCm[3 * vertex], meshInfo.pinLocalCm[3 * vertex + 1],
                                   meshInfo.pinLocalCm[3 * vertex + 2]);
            const glm::vec3 component = bonePosition(frame, driver) + bx * offset.x + by * offset.y + bz * offset.z;
            const glm::vec3 delta = component - rootPos;
            positionsM[3 * vertex + 0] = glm::dot(delta, rx) * 0.01f;
            positionsM[3 * vertex + 1] = glm::dot(delta, ry) * 0.01f;
            positionsM[3 * vertex + 2] = glm::dot(delta, rz) * 0.01f;
        }
    }

    // The state branch B's solve starts from: the Verlet extrapolation of its own last two
    // frames plus one step of gravity, with the pins put on the body. No network appears
    // anywhere in it, which is what makes it an answer to "would constraints alone do".
    //
    // Its first frame is seeded from the network instead, and takes no gravity. The branch
    // needs *some* initial configuration and the network's frame-0 prediction is the one both
    // other branches also start from, so seeding from it is what keeps the three comparable;
    // the sibling PoC's `BallisticGravity` reasons the same way about its settle step, where a
    // nominal dt of 1/3 s would have displaced 1.09 m before a single constraint ran.
    void buildBallisticStart(uint32_t frame, BranchState& state) {
        const size_t values = size_t(kVertexCount) * 3;
        const auto& output = sequence.output();
        if (state.solvedFrames == 0) {
            for (size_t index = 0; index < values; ++index) state.startM[index] = output[index] * 0.01f;
        } else {
            const glm::vec3 gravity = localGravity(frame);
            const float dt = xpbdConfig.timestep;
            const float scale = dt * dt;
            const bool haveVelocity = state.solvedFrames >= 2;
            for (uint32_t vertex = 0; vertex < kVertexCount; ++vertex) {
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    const size_t index = size_t(vertex) * 3 + axis;
                    const float inertia = haveVelocity
                        ? 2.0f * state.previousM[index] - state.olderM[index]
                        : state.previousM[index];
                    state.startM[index] = inertia + gravity[static_cast<int>(axis)] * scale;
                }
            }
        }
        applyPinTargets(frame, state.startM.data());
    }

    // One step for every drawn branch, leaving each role's centimetre positions in `roleCm`.
    void solveBranches(uint32_t frame) {
        const auto& output = sequence.output();
        roleCm.fill(nullptr);
        roleCm[kBranchNetwork] = output.data();
        if (!xpbdEnabled || !meshMode) return;
        const size_t values = size_t(kVertexCount) * 3;

        // Role A first, in metres: it is measured against the same capsule placement and in the
        // same units as the others, so the comparison is not confounded by either, and the bind
        // check below needs it before B builds its start state.
        BranchState& network = branches[kBranchNetwork];
        for (size_t index = 0; index < values; ++index) network.startM[index] = output[index] * 0.01f;
        network.penetration = measurePenetration(network.startM.data(), network.perComponent);

        for (uint32_t slot = 0; slot < branchCount; ++slot) {
            const uint32_t role = roleOfSlot[slot];
            if (!roleSolves(role)) continue;
            BranchState& state = branches[role];
            // With no iterations there is nothing to solve, so for the guided branch the
            // centimetre-to-metre round trip is skipped rather than performed and undone. It is
            // not the identity in float32 -- `x * 0.01f * 100.0f` differs from `x` by an ulp on
            // most values -- and that alone was enough to make `--xpbd-iterations 0` produce a
            // different hash from the pure network path, which is precisely the equality this
            // gate exists to assert. Branch B has no such equality to preserve: its start state
            // is not the network's output.
            if (roleUsesNetwork(role) && state.iterations <= 0) {
                state.solveMs = 0.0;
                roleCm[role] = output.data();
                continue;
            }

            const float* inertial = nullptr;
            if (roleUsesNetwork(role)) {
                for (size_t index = 0; index < values; ++index) state.startM[index] = output[index] * 0.01f;
                // The Verlet extrapolation 2*x_now - x_previous that the guide trust gate
                // measures against. Before two solved frames exist there is no velocity to
                // extrapolate, so the gate is passed null rather than handed the prediction
                // itself: that would read as zero displacement and trust the network
                // unconditionally on exactly the frame where it is least justified.
                if (state.solvedFrames >= 2 && xpbdConfig.guideTrustRatio > 0.0f) {
                    for (size_t index = 0; index < values; ++index) {
                        state.inertialM[index] = 2.0f * state.previousM[index] - state.olderM[index];
                    }
                    inertial = state.inertialM.data();
                }
            } else {
                buildBallisticStart(frame, state);
                measurePinBindGap(state);
            }

            const auto begin = std::chrono::steady_clock::now();
            state.solver.step(state.startM.data(), inertial, placedCapsulesM, state.solvedM.data());
            state.solveMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();

            std::swap(state.olderM, state.previousM);
            std::copy(state.solvedM.begin(), state.solvedM.end(), state.previousM.begin());
            if (state.solvedFrames < 2) ++state.solvedFrames;
            state.penetration = measurePenetration(state.solvedM.data(), state.perComponent);
            for (size_t index = 0; index < values; ++index) state.solvedCm[index] = state.solvedM[index] * 100.0f;
            roleCm[role] = state.solvedCm.data();
        }
    }

    // How far the measured rigid bind sits from where the network puts the same pinned vertices.
    // This is the one check on the bind that does not require looking at the screen: the network
    // and the bind are independent descriptions of where the garment is sewn to the body, so a
    // bind taken in the wrong bone frame, or with the up axis guessed, separates them by a large
    // fraction of the garment rather than by the few centimetres a rigid approximation of smooth
    // skinning costs. Reported every frame; the verification gate reads the worst.
    void measurePinBindGap(const BranchState& state) {
        pinBindGapMaxM = 0.0f;
        pinBindGapMeanM = 0.0f;
        if (meshInfo.pinDriverIndices == nullptr) return;
        const float* reference = branches[kBranchNetwork].startM.data();
        double total = 0.0;
        uint32_t counted = 0;
        for (uint32_t vertex = 0; vertex < kVertexCount; ++vertex) {
            if (meshInfo.pinDriverIndices[vertex] == mlcloth::kNoDriver) continue;
            float squared = 0.0f;
            for (uint32_t axis = 0; axis < 3; ++axis) {
                const float delta = state.startM[3 * vertex + axis] - reference[3 * vertex + axis];
                squared += delta * delta;
            }
            const float distance = std::sqrt(squared);
            pinBindGapMaxM = std::max(pinBindGapMaxM, distance);
            total += double(distance);
            ++counted;
        }
        if (counted != 0) pinBindGapMeanM = float(total / double(counted));
        worstPinBindGapM = std::max(worstPinBindGapM, pinBindGapMaxM);
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
        renderFrame = frame;
        const auto begin = std::chrono::steady_clock::now();
        lastInferenceMs = sequence.inferFrame(*runtime, frame, localFu, componentFu, componentPositionsCm);
        const auto afterInference = std::chrono::steady_clock::now();
        placeCapsules(frame);
        solveBranches(frame);
        buildPointData(frame);
        // Over every drawn branch, not just the primary one: with three branches a solver whose
        // state leaked across a reset would otherwise hide in whichever branch the hash skipped.
        // Comparable only between runs with the same branch count, which is the same caveat the
        // upload size already carries.
        if (simulationSteps == 0) {
            firstSolvedHash = fnv1a64(&latestLocalPoints[0].x, latestLocalPoints.size() * 4);
        }
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
        for (BranchState& state : branches) {
            std::fill(state.previousM.begin(), state.previousM.end(), 0.0f);
            std::fill(state.olderM.begin(), state.olderM.end(), 0.0f);
            std::fill(state.inertialM.begin(), state.inertialM.end(), 0.0f);
            state.solvedFrames = 0;
        }
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
        // Drives the clip at one frame per step with looping on, whatever the playback settings
        // are: this is a check that the *loop* resets every piece of state, and under decimation
        // or hold-last-frame the clip never lands back on frame 0, so the check below would fire
        // on the settings rather than on a leak. Restored afterwards so the interactive session
        // still honours what was asked for.
        const int32_t requestedStep = frameStep;
        const bool requestedHold = holdLastFrame;
        frameStep = 1;
        holdLastFrame = false;
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
        frameStep = requestedStep;
        holdLastFrame = requestedHold;
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
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxConcurrentFrames * 7 + 7),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxConcurrentFrames * 3 + 1),
        };
        VkDescriptorPoolCreateInfo poolInfo = vks::initializers::descriptorPoolCreateInfo(sizes, maxConcurrentFrames * 3 + 1);
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
        std::vector<VkDescriptorSetLayoutBinding> computeBindings = {
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
        };
        VkDescriptorSetLayoutCreateInfo computeLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(computeBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &computeLayoutInfo, nullptr, &computeSetLayout));
        // Visible to the fragment stage as well now: the sky pass unprojects a pixel through the
        // inverse view-projection, and both surface stages need the eye position for a specular.
        const auto graphicsBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
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
        // One set, not one per frame in flight: every buffer it names is either read-only or
        // rewritten before the dispatch that reads it, and the body is skinned once per frame
        // rather than once per branch.
        if (bodyMeshLoaded && bodyDisplay != BodyDisplay::None) {
            std::vector<VkDescriptorSetLayoutBinding> skinBindings;
            for (uint32_t binding = 0; binding < 7; ++binding) {
                skinBindings.push_back(vks::initializers::descriptorSetLayoutBinding(
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, binding));
            }
            skinBindings.push_back(vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 7));
            VkDescriptorSetLayoutCreateInfo skinLayoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(skinBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &skinLayoutInfo, nullptr, &skinSetLayout));
            VkDescriptorSetAllocateInfo skinAllocate = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &skinSetLayout, 1);
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &skinAllocate, &skinSet));
            std::array<VkWriteDescriptorSet, 8> skinWrites = {
                vks::initializers::writeDescriptorSet(skinSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &skin.restPositions.descriptor),
                vks::initializers::writeDescriptorSet(skinSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &skin.restNormals.descriptor),
                vks::initializers::writeDescriptorSet(skinSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &skin.boneIndices.descriptor),
                vks::initializers::writeDescriptorSet(skinSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &skin.boneWeights.descriptor),
                vks::initializers::writeDescriptorSet(skinSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &skin.skinMatrices.descriptor),
                vks::initializers::writeDescriptorSet(skinSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, &skin.positions.descriptor),
                vks::initializers::writeDescriptorSet(skinSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6, &skin.normals.descriptor),
                vks::initializers::writeDescriptorSet(skinSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 7, &skin.uniform.descriptor),
            };
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(skinWrites.size()), skinWrites.data(), 0, nullptr);
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
        if (skinSetLayout != VK_NULL_HANDLE) {
            VkPipelineLayoutCreateInfo skinLayoutInfo = vks::initializers::pipelineLayoutCreateInfo(&skinSetLayout, 1);
            VK_CHECK_RESULT(vkCreatePipelineLayout(device, &skinLayoutInfo, nullptr, &skinPipelineLayout));
            VkComputePipelineCreateInfo skinInfo = vks::initializers::computePipelineCreateInfo(skinPipelineLayout);
            skinInfo.stage = loadShader(getShadersPath() + "mlclothcpu/body_skin.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
            VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &skinInfo, nullptr, &skinPipeline));
        }

        VkPipelineLayoutCreateInfo graphicsLayoutInfo = vks::initializers::pipelineLayoutCreateInfo(&graphicsSetLayout, 1);
        // A rigid world offset and a branch tint. The offset is read by the vertex stage and the
        // tint by the fragment stage; both are declared on the layout the point pipeline shares,
        // because an unused range costs nothing and one layout keeps the pipelines swappable
        // behind the same descriptor set.
        const VkPushConstantRange tintRange{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(InstancePush) };
        graphicsLayoutInfo.pushConstantRangeCount = 1;
        graphicsLayoutInfo.pPushConstantRanges = &tintRange;
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

        // The backdrop, in both the mesh and the point-cloud path. No vertex input, no depth
        // test and no depth write: it is drawn first and covers the viewport, so the geometry
        // that follows overwrites it wherever the geometry exists -- including where the geometry
        // is *below* the floor, which a depth-writing ground plane would hide.
        {
            VkPipelineInputAssemblyStateCreateInfo skyAssembly = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
            VkPipelineDepthStencilStateCreateInfo skyDepth = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS);
            VkPipelineVertexInputStateCreateInfo skyVertexInput = vks::initializers::pipelineVertexInputStateCreateInfo();
            std::array<VkPipelineShaderStageCreateInfo, 2> skyStages = {
                loadShader(getShadersPath() + "mlclothcpu/sky.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
                loadShader(getShadersPath() + "mlclothcpu/sky.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
            };
            VkGraphicsPipelineCreateInfo skyPipelineInfo = pipeline;
            skyPipelineInfo.pVertexInputState = &skyVertexInput;
            skyPipelineInfo.pInputAssemblyState = &skyAssembly;
            skyPipelineInfo.pDepthStencilState = &skyDepth;
            skyPipelineInfo.stageCount = static_cast<uint32_t>(skyStages.size());
            skyPipelineInfo.pStages = skyStages.data();
            VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &skyPipelineInfo, nullptr, &skyPipeline));
        }

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
        // The frame the step actually ran, not `clipFrame - 1`: with decimation the previous
        // clip frame is not the one the contacts were resolved against, and the body on screen
        // has to be the body the solver saw or the penetration it shows is not the penetration
        // it measured.
        updateBodyGeometry(renderFrame);
        updateSkinMatrices(renderFrame);
        std::memcpy(frame.transformUniform.mapped, &transformUniform, sizeof(transformUniform));
        cameraUniform.projection = camera.matrices.perspective;
        cameraUniform.view = camera.matrices.view;
        // The eye is read out of the view matrix rather than taken from `camera.position`: with
        // the lookat control that member is the orbit offset, not where the camera stands, and
        // the specular and the sky's ray origin both need the latter.
        const glm::mat4 inverseView = glm::inverse(cameraUniform.view);
        cameraUniform.cameraPositionM = glm::vec4(glm::vec3(inverseView[3]), 1.0f);
        cameraUniform.inverseViewProjection = glm::inverse(cameraUniform.projection * cameraUniform.view);
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
        // Every branch's block, not one branch's worth. This was `kVertexCount` alone, so in
        // comparison mode the second block was never transformed and the second garment was
        // drawn from whatever was in device-local memory. The shaders were already correct --
        // both bound the count from their uniform -- so nothing caught it but the picture.
        vkCmdDispatch(command, (kVertexCount * branchCount + 127) / 128, 1, 1);
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
            vkCmdDispatch(command, (kVertexCount * branchCount + 127) / 128, 1, 1);
        }
        // The body's skin. Independent of the cloth passes -- different buffers, no shared
        // writes -- so it needs no barrier against them, only the one before the draw. Skipped
        // when the capsules are on screen instead, which is what makes the switch free.
        if (skinPipeline != VK_NULL_HANDLE && bodyDisplay == BodyDisplay::Mesh) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, skinPipeline);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, skinPipelineLayout, 0, 1, &skinSet, 0, nullptr);
            vkCmdDispatch(command, (bodyInfo.vertices + 127) / 128, 1, 1);
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
        if (skinPipeline != VK_NULL_HANDLE && bodyDisplay == BodyDisplay::Mesh) {
            VkBufferMemoryBarrier skinBarrier = pointBarrier;
            skinBarrier.buffer = skin.positions.buffer;
            vertexBarriers.push_back(skinBarrier);
            skinBarrier.buffer = skin.normals.buffer;
            vertexBarriers.push_back(skinBarrier);
        }
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0,
            0, nullptr, static_cast<uint32_t>(vertexBarriers.size()), vertexBarriers.data(), 0, nullptr);

        VkClearValue clear[2]{};
        // Only ever seen with the backdrop off; the sky pass covers every pixel otherwise.
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
        // First, so everything else lands on top of it. One triangle, one descriptor set, no
        // vertex buffer; it is outside the timestamped compute range and costs nothing that any
        // reported number covers.
        if (skyEnabled) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &frame.graphicsSet, 0, nullptr);
            vkCmdDraw(command, 3, 1, 0, 0);
        }
        VkDeviceSize offset = 0;
        if (meshMode) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, clothPipeline);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &frame.graphicsSet, 0, nullptr);
            const std::array<VkBuffer, 2> streams = { frame.points.buffer, frame.normals.buffer };
            const std::array<VkDeviceSize, 2> offsets = { 0, 0 };
            vkCmdBindVertexBuffers(command, 0, static_cast<uint32_t>(streams.size()), streams.data(), offsets.data());
            vkCmdBindIndexBuffer(command, meshResources.triangles.buffer, 0, VK_INDEX_TYPE_UINT32);
            // One draw per slot, so each carries its own hue and a hidden branch is simply not
            // drawn. The vertex blocks and the solve are unaffected by visibility: hiding a
            // branch changes the picture and nothing else, which is what makes the checkboxes
            // safe to use while reading numbers off the same frame.
            for (uint32_t slot = 0; slot < branchCount; ++slot) {
                if (!branchVisible[roleOfSlot[slot]]) continue;
                pushInstance(command, glm::vec3(0.0f), tintOf(roleOfSlot[slot]));
                vkCmdDrawIndexed(command, meshInfo.triangles * 3, 1, slot * meshInfo.triangles * 3, 0, 0);
            }
            // The body, drawn in whichever form was asked for. Culling is off on this pipeline,
            // so a surface reads correctly from inside as well -- which is the view that shows a
            // garment sunk into a limb.
            if (bodyMeshLoaded && bodyDisplay == BodyDisplay::Mesh && skin.indexCount != 0) {
                const std::array<VkBuffer, 2> skinStreams = { skin.positions.buffer, skin.normals.buffer };
                const std::array<VkDeviceSize, 2> skinOffsets = { 0, 0 };
                vkCmdBindVertexBuffers(command, 0, static_cast<uint32_t>(skinStreams.size()), skinStreams.data(), skinOffsets.data());
                vkCmdBindIndexBuffer(command, skin.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                // Neutral grey, and one draw per slot from one skinned copy: the body is
                // identical in every branch by construction, and giving it a branch hue would
                // suggest otherwise. Darker and a shade warmer than a true grey on purpose: it
                // keeps the body under the garment in value, and it is what separates a lit
                // shoulder from the cool sky behind it, which no amount of brightness would.
                for (uint32_t slot = 0; slot < branchCount; ++slot) {
                    if (!branchVisible[roleOfSlot[slot]]) continue;
                    pushInstance(command, branchWorldOffsetM(slot, renderFrame), glm::vec4(0.30f, 0.29f, 0.275f, 1.0f));
                    vkCmdDrawIndexed(command, skin.indexCount, 1, 0, 0, 0);
                }
            } else if (capsulesLoaded && bodyDisplay == BodyDisplay::Capsules && body.indexCount != 0) {
                const std::array<VkBuffer, 2> bodyStreams = { body.positions.buffer, body.normals.buffer };
                const std::array<VkDeviceSize, 2> bodyOffsets = { 0, 0 };
                vkCmdBindVertexBuffers(command, 0, static_cast<uint32_t>(bodyStreams.size()), bodyStreams.data(), bodyOffsets.data());
                vkCmdBindIndexBuffer(command, body.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
                // The capsule copies already carry their own lateral offset in their positions.
                const uint32_t perSlot = body.indexCount / branchCount;
                for (uint32_t slot = 0; slot < branchCount; ++slot) {
                    if (!branchVisible[roleOfSlot[slot]]) continue;
                    pushInstance(command, glm::vec3(0.0f), glm::vec4(0.26f, 0.27f, 0.30f, 1.0f));
                    vkCmdDrawIndexed(command, perSlot, 1, slot * perSlot, 0, 0);
                }
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

    // Two things about the body that a picture cannot settle.
    //
    // The first is whether the compute skin agrees with the same arithmetic on the host. The
    // shaders in this PoC have twice been correct while the dispatch around them was not, so
    // the oracle is worth its 67,857 host-side vertices once.
    //
    // The second is registration, and it is the one that decides whether the body is worth
    // drawing at all. The character mesh, its skeleton and the driver clip come from three
    // different exports; if the bind pose, the coordinate mapping or the bone order were wrong
    // the body would still be a body, just not this character's body in this pose. The check is
    // that the *garment* lands on the skin -- specifically the network branch's prediction, which
    // is the arm no constraint and no capsule touches, so the number does not move when the
    // solver is retuned. What makes it evidence is that the network was trained without ever
    // seeing the body mesh, the skeleton or the bind pose, so the two sides of the comparison
    // share nothing but the character. Measured over run, sprint, lunge and death poses the
    // nearest-skin distance is p05 0.4 cm and p50 2.2-2.7 cm, and the sibling PoC independently
    // measured its own cloth-to-body distance at a 2.41 cm median. A wrong chain does not produce
    // a stable 2.4 cm across poses.
    void verifyBodySkin(const glm::vec4* gpuSkinned) {
        bodySkinGapM = 0.0f;
        if (!bodyMeshLoaded || gpuSkinned == nullptr) return;
        for (uint32_t vertex = 0; vertex < bodyInfo.vertices; ++vertex) {
            const glm::vec3 host = skinVertexWorldM(vertex);
            for (uint32_t axis = 0; axis < 3; ++axis) {
                bodySkinGapM = std::max(bodySkinGapM, std::abs(gpuSkinned[vertex][axis] - host[axis]));
            }
        }
    }

    // The one thing the backdrop pass does that no vertex buffer constrains: it turns a pixel
    // back into a world ray through `inverseViewProjection`. A wrong sign or a wrong depth
    // convention there does not fail -- it draws the sky where the floor belongs, or puts the
    // whole backdrop behind the camera, and both are easy to mistake for a camera problem. So
    // the reconstruction is checked against the forward transform the geometry uses: at the four
    // screen corners and the centre, the ray has to point into the scene, and re-projecting a
    // point along it has to return the pixel it came from.
    //
    // The round trip is the part that catches a convention error rather than an arithmetic one:
    // feeding the NDC y with the wrong sign inverts it here, while every length and angle stays
    // perfectly plausible.
    //
    // The limit is loose on purpose. Unprojecting at the far plane is badly conditioned -- a
    // hyperbolic depth range with near 0.01 and far 1000 puts almost the whole NDC z range in the
    // first few centimetres, so the far plane is where single precision has least to say -- and
    // it measures 6e-5 NDC, about a sixteenth of a pixel. Any convention error is off by order 1,
    // three orders clear of this, so tightening it would only make the gate a float-precision
    // tripwire that fails when someone changes the far plane.
    static constexpr float kCameraRayNdcLimit = 1.0e-3f;
    void verifyCameraRays() {
        const glm::mat4 viewProjection = cameraUniform.projection * cameraUniform.view;
        const glm::mat4 inverseView = glm::inverse(cameraUniform.view);
        const glm::vec3 eye(cameraUniform.cameraPositionM);
        const glm::vec3 forward = -glm::normalize(glm::vec3(inverseView[2]));
        const std::array<glm::vec2, 5> corners{ { { 0.0f, 0.0f }, { -1.0f, -1.0f }, { 1.0f, -1.0f },
                                                  { -1.0f, 1.0f }, { 1.0f, 1.0f } } };
        cameraRayNdcError = 0.0f;
        cameraRayWorstFacing = 1.0f;
        for (const glm::vec2& ndc : corners) {
            // `farPoint`, not `far`: windef.h still defines that as an empty macro.
            const glm::vec4 farPoint = cameraUniform.inverseViewProjection * glm::vec4(ndc, 1.0f, 1.0f);
            const glm::vec3 direction = glm::normalize(glm::vec3(farPoint) / farPoint.w - eye);
            cameraRayWorstFacing = std::min(cameraRayWorstFacing, glm::dot(direction, forward));
            // Any positive distance does: the returned NDC x and y do not depend on where along
            // the ray the point sits.
            const glm::vec4 back = viewProjection * glm::vec4(eye + direction * 3.0f, 1.0f);
            cameraRayNdcError = std::max(cameraRayNdcError, glm::length(glm::vec2(back) / back.w - ndc));
        }
        if (cameraRayNdcError > kCameraRayNdcLimit || cameraRayWorstFacing <= 0.0f) {
            std::ostringstream message;
            message << "Backdrop ray reconstruction disagrees with the forward transform: NDC round"
                    << " trip " << cameraRayNdcError << " (limit " << kCameraRayNdcLimit
                    << "), worst facing " << cameraRayWorstFacing << " (must be positive)";
            vks::tools::exitFatal(message.str(), -1);
        }
    }

    void measureGarmentToSkin() {
        garmentSkinP05Cm = 0.0f;
        garmentSkinP50Cm = 0.0f;
        garmentSkinMatched = 0;
        if (!bodyMeshLoaded) return;
        // A uniform grid over the skin, so this is one pass rather than 5,294 x 67,857. The
        // cell is the search radius: a garment vertex further than that from any skin vertex is
        // not counted, which is correct for a hem that is genuinely off the body.
        constexpr float kCellM = 0.08f;
        struct Key { int x, y, z; };
        std::map<std::tuple<int, int, int>, std::vector<uint32_t>> grid;
        std::vector<glm::vec3> skinWorld(bodyInfo.vertices);
        for (uint32_t vertex = 0; vertex < bodyInfo.vertices; ++vertex) {
            skinWorld[vertex] = skinVertexWorldM(vertex);
            const glm::vec3& p = skinWorld[vertex];
            grid[{ int(std::floor(p.x / kCellM)), int(std::floor(p.y / kCellM)), int(std::floor(p.z / kCellM)) }]
                .push_back(vertex);
        }
        std::vector<float> distances;
        distances.reserve(kVertexCount);
        // Measured against the network branch where it is drawn, and with its side-by-side
        // displacement taken back off. The branches stand 90 cm apart, so comparing an offset
        // branch against one un-offset body finds nothing within any sane radius -- which is
        // exactly what the first version of this reported.
        uint32_t measureSlot = 0;
        for (uint32_t slot = 0; slot < branchCount; ++slot) {
            if (roleOfSlot[slot] == kBranchNetwork) { measureSlot = slot; break; }
        }
        const glm::vec3 lateral = branchWorldOffsetM(measureSlot, renderFrame);
        for (uint32_t vertex = 0; vertex < kVertexCount; ++vertex) {
            const glm::vec3 p = glm::vec3(latestWorldPoints[measureSlot * kVertexCount + vertex]) - lateral;
            const int cx = int(std::floor(p.x / kCellM)), cy = int(std::floor(p.y / kCellM)), cz = int(std::floor(p.z / kCellM));
            float best = std::numeric_limits<float>::max();
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        const auto found = grid.find({ cx + dx, cy + dy, cz + dz });
                        if (found == grid.end()) continue;
                        for (uint32_t candidate : found->second) {
                            const glm::vec3 delta = skinWorld[candidate] - p;
                            best = std::min(best, glm::dot(delta, delta));
                        }
                    }
                }
            }
            if (best < std::numeric_limits<float>::max()) distances.push_back(std::sqrt(best));
        }
        garmentSkinMatched = static_cast<uint32_t>(distances.size());
        if (distances.empty()) return;
        std::sort(distances.begin(), distances.end());
        const auto pick = [&](double fraction) {
            const size_t index = std::min(distances.size() - 1,
                static_cast<size_t>(fraction * double(distances.size())));
            return distances[index] * 100.0f;
        };
        garmentSkinP05Cm = pick(0.05);
        garmentSkinP50Cm = pick(0.50);
    }

    // Copies the skinned body back and hands it to the host oracle. Separate from the cloth
    // readback because it is a different buffer and a different size, and because the body is
    // optional -- there is nothing to compare when it was not loaded.
    void verifySkinnedBody() {
        bodySkinGapM = 0.0f;
        // Only when the skin pass actually ran this frame: with the capsules on screen the
        // output buffer exists but holds nothing, and comparing that against the host oracle
        // would fail the gate for the wrong reason.
        if (!bodyMeshLoaded || bodyDisplay != BodyDisplay::Mesh) return;
        if (skin.positions.buffer == VK_NULL_HANDLE) return;
        const VkDeviceSize bytes = VkDeviceSize(bodyInfo.vertices) * sizeof(glm::vec4);
        vks::Buffer readback;
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &readback, bytes));
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
        barrier.buffer = skin.positions.buffer;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        VkBufferCopy copy{ 0, 0, bytes };
        vkCmdCopyBuffer(command, skin.positions.buffer, readback.buffer, 1, &copy);
        VK_CHECK_RESULT(vkEndCommandBuffer(command));
        VkSubmitInfo submit = vks::initializers::submitInfo();
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        VK_CHECK_RESULT(vkQueueWaitIdle(queue));
        VK_CHECK_RESULT(readback.map());
        verifyBodySkin(static_cast<const glm::vec4*>(readback.mapped));
        readback.unmap();
        readback.destroy();
        vkFreeCommandBuffers(device, cmdPool, 1, &command);
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
        // Every branch's block. Checking only the first would have left the comparison mode's
        // extra blocks outside the one gate that reads what the GPU actually produced -- which
        // is exactly where the untransformed second block hid.
        for (uint32_t index = 0; index < kVertexCount * branchCount; ++index) {
            for (uint32_t axis = 0; axis < 3; ++axis) {
                maxError = std::max(maxError, std::abs(static_cast<double>(actual[index][axis]) - latestWorldPoints[index][axis]));
            }
        }
        readback.unmap();
        readback.destroy();
        vkFreeCommandBuffers(device, cmdPool, 1, &command);
        verifySkinnedBody();
        verifyCameraRays();
        measureGarmentToSkin();
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
               << ",\n  \"xpbd_iterations_b\": " << (compareMode ? branchIterationsB : 0)
               // The knobs the solver was actually configured with, not just how many times it
               // ran. Without them a dropped flag is invisible: `run.ps1 -Compare` forwarded
               // none of these for a while, and the only symptom was a result that did not
               // match the command that produced it.
               << ",\n  \"xpbd_stretch_compliance\": " << xpbdConfig.stretchCompliance
               << ",\n  \"xpbd_bend_compliance\": " << xpbdConfig.bendCompliance
               << ",\n  \"xpbd_area_floor\": " << xpbdConfig.areaFloor
               << ",\n  \"xpbd_area_compliance\": " << xpbdConfig.areaCompliance
               << ",\n  \"xpbd_guide_compliance\": " << xpbdConfig.guideCompliance
               << ",\n  \"xpbd_guide_trust_ratio\": " << xpbdConfig.guideTrustRatio
               << ",\n  \"xpbd_one_sided\": " << (xpbdConfig.oneSided ? "true" : "false")
               // 1.0 means B's solve costs what C's inference plus solve costs. Away from 1.0 the
               // two branches are not being given the same money, whatever the iteration counts
               // say, and the difference on screen is partly a budget difference.
               << ",\n  \"compare_budget_ratio\": " << compareBudgetRatio()
               << ",\n  \"xpbd_solve_ms\": " << branches[primaryRole()].solveMs
               << ",\n  \"xpbd_stretch_active\": " << branches[primaryRole()].solver.lastStep().stretchActive
               << ",\n  \"xpbd_area_active\": " << branches[primaryRole()].solver.lastStep().areaFloorsActive
               << ",\n  \"xpbd_max_correction_m\": " << branches[primaryRole()].solver.lastStep().maxCorrectionM
               << ",\n  \"xpbd_contacts_resolved\": " << branches[primaryRole()].solver.lastStep().contactsResolved
               << ",\n  \"capsules\": " << (capsulesLoaded ? capsuleInfo.count : 0u)
               << ",\n  \"compare\": " << (compareMode ? "true" : "false")
               << ",\n  \"pin_bind\": " << (meshMode && meshInfo.pinDriverIndices ? "true" : "false")
               // Which frame the per-branch numbers below were taken at. This runs once, on the
               // first rendered frame, so they are one frame of one clip and not a clip summary
               // -- branch B is still one step from its network seed there and has not yet
               // diverged. Read them as a smoke test; read the side-by-side view for the result.
               << ",\n  \"measured_at_frame\": " << clipFrame
               << ",\n  \"pin_bind_gap_mean_m\": " << pinBindGapMeanM
               << ",\n  \"pin_bind_gap_max_m\": " << pinBindGapMaxM
               << ",\n  \"pin_bind_gap_worst_m\": " << worstPinBindGapM
               << ",\n  \"pin_bind_gap_limit_m\": " << kPinBindGapLimitM
               << ",\n  \"body\": \"" << (bodyDisplay == BodyDisplay::Mesh && bodyMeshLoaded ? "mesh"
                                        : bodyDisplay == BodyDisplay::Capsules ? "capsules" : "none") << "\""
               << ",\n  \"body_vertices\": " << (bodyMeshLoaded ? bodyInfo.vertices : 0u)
               << ",\n  \"body_folded_bones\": " << (bodyMeshLoaded ? bodyInfo.foldedBones : 0u)
               << ",\n  \"body_skin_gpu_vs_cpu_m\": " << bodySkinGapM
               << ",\n  \"garment_to_skin_p05_cm\": " << garmentSkinP05Cm
               << ",\n  \"garment_to_skin_p50_cm\": " << garmentSkinP50Cm
               << ",\n  \"garment_to_skin_matched\": " << garmentSkinMatched
               << ",\n  \"sky\": " << (skyEnabled ? "true" : "false")
               << ",\n  \"camera_ray_ndc_round_trip\": " << cameraRayNdcError
               << ",\n  \"camera_ray_worst_facing\": " << cameraRayWorstFacing
               // Legacy pair, kept because it is what the earlier two-branch reports carried:
               // the network alone against the hybrid.
               << ",\n  \"penetration_pure_fraction\": " << branches[kBranchNetwork].penetration.fraction
               << ",\n  \"penetration_pure_deepest_m\": " << branches[kBranchNetwork].penetration.deepestM
               << ",\n  \"penetration_solved_fraction\": " << branches[kBranchHybrid].penetration.fraction
               << ",\n  \"penetration_solved_deepest_m\": " << branches[kBranchHybrid].penetration.deepestM
               << ",\n  \"branches\": [";
        for (uint32_t slot = 0; slot < branchCount; ++slot) {
            const uint32_t role = roleOfSlot[slot];
            const BranchState& state = branches[role];
            report << (slot ? ",\n    " : "\n    ")
                   << "{\"name\": \"" << kBranchNames[role] << "\""
                   << ", \"network\": " << (roleUsesNetwork(role) ? "true" : "false")
                   << ", \"iterations\": " << (roleSolves(role) ? state.iterations : 0)
                   << ", \"solve_ms\": " << state.solveMs
                   << ", \"contacts_resolved\": " << state.solver.lastStep().contactsResolved
                   << ", \"penetration_fraction\": " << state.penetration.fraction
                   << ", \"penetration_deepest_m\": " << state.penetration.deepestM
                   << ", \"per_component\": [";
            for (uint32_t component = 0; component < componentCount; ++component) {
                report << (component ? ", " : "") << state.perComponent[component].fraction;
            }
            report << "]}";
        }
        report << "\n  ]"
               << "\n}\n";
        gpuVerificationDone = true;
        std::cout << "GPU coordinate verification max_abs=" << maxError << " m\n";
        if (maxError > 1.0e-5) vks::tools::exitFatal("GPU coordinate verification exceeded 1e-5 m", -1);
        // Only meaningful once a network-free branch has actually built a start state.
        if (worstPinBindGapM > 0.0f) {
            std::cout << "Pin bind agreement with the network: worst " << worstPinBindGapM
                      << " m over the clip so far, limit " << kPinBindGapLimitM << " m\n";
            if (worstPinBindGapM > kPinBindGapLimitM) {
                vks::tools::exitFatal("The measured pin bind disagrees with the network by more than "
                                      "a rigid approximation of skinning can explain; the bind is "
                                      "probably in the wrong bone frame", -1);
            }
        }
        std::cout << "Backdrop ray round trip: " << cameraRayNdcError << " NDC, worst facing "
                  << cameraRayWorstFacing << "\n";
        if (bodyMeshLoaded) {
            std::cout << "Body skin GPU vs CPU max_abs=" << bodySkinGapM << " m; garment to skin p05="
                      << garmentSkinP05Cm << " cm p50=" << garmentSkinP50Cm << " cm over "
                      << garmentSkinMatched << " of " << kVertexCount << " vertices\n";
            if (bodySkinGapM > 1.0e-5f) {
                vks::tools::exitFatal("The compute skin disagrees with the host skin by more than 1e-5 m", -1);
            }
            // A three-way band, not a threshold: too small means the garment is buried in the
            // skin, too large means the body is not where the garment is. Measured 2.2-2.7 cm
            // across run, sprint, lunge and death; the sibling PoC measured 2.41 cm on its own
            // cloth. Outside 0.2..8 cm the registration is what is wrong, not the garment.
            if (garmentSkinMatched * 2 < kVertexCount) {
                vks::tools::exitFatal("Fewer than half the garment vertices have any skin vertex within 8 cm; "
                                      "the body is not registered to the garment", -1);
            }
            if (garmentSkinP50Cm < 0.2f || garmentSkinP50Cm > 8.0f) {
                vks::tools::exitFatal("The garment's median distance to the skin is outside 0.2..8 cm, which no "
                                      "pose explains; the bind pose, the coordinate mapping or the bone order is wrong", -1);
            }
        }
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
        if (compareMode) {
            overlay->text("Blue A network only | orange B constraints only | green C hybrid");
            overlay->checkBox("Show A (network only)", &branchVisible[kBranchNetwork]);
            overlay->checkBox("Show B (constraints only)", &branchVisible[kBranchConstraints]);
            overlay->checkBox("Show C (hybrid)", &branchVisible[kBranchHybrid]);
            overlay->sliderFloat("Branch spacing (cm)", &compareSpacingCm, 40.0f, 200.0f);
        }
        // Switching what stands in for the character, live. Only between forms that were
        // actually loaded: the buffers for the other one may not exist.
        if (bodyMeshLoaded || capsulesLoaded) {
            int32_t display = static_cast<int32_t>(bodyDisplay);
            std::vector<std::string> options = { "none" };
            std::vector<BodyDisplay> values = { BodyDisplay::None };
            if (capsulesLoaded && body.indexCount != 0) { options.push_back("capsules"); values.push_back(BodyDisplay::Capsules); }
            if (bodyMeshLoaded && skin.indexCount != 0) { options.push_back("skinned mesh"); values.push_back(BodyDisplay::Mesh); }
            for (size_t i = 0; i < values.size(); ++i) if (values[i] == bodyDisplay) display = static_cast<int32_t>(i);
            if (options.size() > 1 && overlay->comboBox("Body", &display, options)) {
                bodyDisplay = values[static_cast<size_t>(display)];
                    }
            if (bodyMeshLoaded && bodyInfo.foldedBones != 0) {
                overlay->text("  %u bone(s) not driven by the model, folded onto a parent:", bodyInfo.foldedBones);
                overlay->text("  the breast patch is ~2 cm off, so read chest clearance off the numbers.");
            }
        }
        overlay->text("P: pause  R: deterministic reset  Esc: exit");
        // Turning the backdrop off is the honest way to check a suspicion about a dark patch: the
        // lighting is the same either way, so what changes is only what it is being read against.
        overlay->checkBox("Sky and floor", &skyEnabled);
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
                overlay->text("Inside the capsules, per branch. The capsules are a ragdoll");
                overlay->text("  envelope, so a fitted piece belongs inside. ChaosCloth's own");
                overlay->text("  run reads 69 / 98 / 14 / 100%% per piece -- only the skirt's is reducible.");
                for (uint32_t slot = 0; slot < branchCount; ++slot) {
                    const BranchState& state = branches[roleOfSlot[slot]];
                    overlay->text("  %s: %.2f%% (%.1f cm deep)", kBranchNames[roleOfSlot[slot]],
                        state.penetration.fraction * 100.0f, -state.penetration.deepestM * 100.0f);
                    for (uint32_t component = 0; component < componentCount; ++component) {
                        overlay->text("      piece %u: %.1f%%", component,
                            state.perComponent[component].fraction * 100.0f);
                    }
                }
            }
            for (uint32_t slot = 0; slot < branchCount; ++slot) {
                const uint32_t role = roleOfSlot[slot];
                if (!roleSolves(role)) continue;
                const BranchState& state = branches[role];
                overlay->text("%s: %.3f ms, %u stretch, %u area, max %.1f mm", kBranchNames[role],
                    state.solveMs, state.solver.lastStep().stretchActive,
                    state.solver.lastStep().areaFloorsActive,
                    state.solver.lastStep().maxCorrectionM * 1000.0f);
            }
            // Whether the two architectures actually cost the same right now. The default 15
            // iterations for B was measured in one configuration, and it does not survive
            // another: turning the area floor on costs the solver about 40%, which lands on B
            // fifteen times and on C eight. An unequal comparison reported as an architecture
            // difference is the failure this line exists to prevent, so it is shown rather than
            // silently corrected -- correcting it per frame would make the picture wobble.
            if (compareMode) {
                const double ratio = compareBudgetRatio();
                overlay->text("Equal-budget check: B costs %.2fx C%s", ratio,
                    (ratio < 0.85 || ratio > 1.15) ? "  <-- retune B iterations" : "");
            }
            if (overlay->sliderInt(compareMode ? "C iterations" : "XPBD iterations", &xpbdConfig.iterations, 0, 64)) {
                reconfigureBranches();
            }
            // B's own count, because the comparison is at equal CPU cost rather than equal
            // iterations: B skips the 2.558 ms inference and spends it here.
            if (compareMode && overlay->sliderInt("B iterations (no network)", &branchIterationsB, 0, 128)) {
                reconfigureBranches();
            }
            if (overlay->sliderFloat("Stretch compliance", &xpbdConfig.stretchCompliance, 0.0f, 0.2f)) reconfigureBranches();
            if (overlay->sliderFloat("Bend compliance", &xpbdConfig.bendCompliance, 0.0f, 0.5f)) reconfigureBranches();
            // These three also have to be pushed. They were not, and dragging them did nothing:
            // the solver reads its own `config_`, which only `configure` writes, so the slider
            // moved the runtime's copy and the solve kept the value from startup.
            if (overlay->sliderFloat("Area floor (x target)", &xpbdConfig.areaFloor, 0.0f, 1.5f)) reconfigureBranches();
            if (overlay->sliderFloat("Guide compliance (<0 off)", &xpbdConfig.guideCompliance, -1.0f, 5.0f)) reconfigureBranches();
            if (overlay->sliderFloat("Guide trust ratio (0 off)", &xpbdConfig.guideTrustRatio, 0.0f, 4.0f)) reconfigureBranches();
        }
    }

    // Pushes the shared configuration into every solving branch, each with its own iteration
    // count and, for the network-free branch, its guide forced off.
    void reconfigureBranches() {
        for (uint32_t slot = 0; slot < branchCount; ++slot) {
            const uint32_t role = roleOfSlot[slot];
            if (!roleSolves(role)) continue;
            branches[role].iterations = roleIterations(role);
            branches[role].solver.configure(roleConfig(role));
        }
    }
};

VULKAN_EXAMPLE_MAIN()
