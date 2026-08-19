	struct HoodPlain3 { float x, y, z; };
	static_assert(sizeof(HoodPlain3) == 12);
	struct HoodPlainU3 { uint32_t x, y, z; };
	static_assert(sizeof(HoodPlainU3) == 12);
	struct HoodSkinParams {
		uint32_t frameIndex{}, nextFrameIndex{}, boneCount{}, characterCount{};
		uint32_t proxyCount{}, clothCount{}, resetState{}, renderFrameIndex{};
	};
	struct HoodFeatureParams {
		uint32_t clothCount{}, proxyCount{}, triangleCount{}, meshEdgeCount{};
		uint32_t embeddingOffset{}, firstStep{}, reserved0{}, reserved1{};
		float timestep{}, collisionRadius{ 0.03f }, material0{}, material1{};
		float material2{}, pad0{}, pad1{}, pad2{};
	};
	struct HoodPostFeatureParams {
		uint32_t clothCount{}, proxyCount{}, triangleCount{}, meshEdgeCount{};
		uint32_t coarse0Count{}, coarse1Count{}, embeddingOffset{}, levelEmbeddingOffset{};
		float timestep{}, collisionRadius{ 0.03f }, material0{}, material1{};
		float material2{}, pad0{}, pad1{}, pad2{};
	};
	// The decoder's MLP id sits after every processor block, so it moves with the block count
	// (3 + blocks * 3). It used to be a literal in each integrate shader, which silently made
	// a student with any depth other than the one the literal was written for decode with a
	// processor MLP instead. Pass it in.
	struct HoodIntegrateParams { uint32_t clothCount{}, firstStep{}, collisionProjection{}, decoderMlp{}; };
	// Push constants for hood_xpbd.comp. Mirrors `struct Push` there; keep the two in step.
	struct HoodXpbdPush {
		uint32_t clothCount{}, constraintCount{}, slotWidth{}, flags{};
		float timestep{}, relaxation{}, contactOffset{}, stretchCompliance{}, bendCompliance{};
	};
	static_assert(sizeof(HoodXpbdPush) == 36);
	static constexpr uint32_t hoodXpbdOneSidedFlag = 1u;
	static constexpr uint32_t hoodXpbdCollisionFlag = 2u;
	struct HoodToyParams {
		uint32_t clothCount{};
		float timestep{ 1.0f / 30.0f }, maxSpeed{ 8.0f }, maxAcceleration{ 30.0f };
		glm::vec4 externalAcceleration{ 0.0f, -9.8f, 0.0f, 0.0f };
	};
	static_assert(sizeof(HoodToyParams) == 32);
	struct HoodGraphicsUniform {
		glm::mat4 projection{};
		glm::mat4 modelview{};
		glm::vec4 lightPos{ -2.0f, 4.0f, -2.0f, 1.0f };
		glm::vec4 rootPosition{};
	};

	bool realSceneMode{ false };
	bool hoodGridScene{ false };
	bool hoodPaused{ false };
	bool hoodRequestReset{ true };
	bool hoodFirstStep{ true };
	bool hoodSimulateFrame{ false };
	bool hoodCollisionProjection{ false };
	// Unstructured Jacobi XPBD on top of the network's prediction (plans/gnn/gnn-xpbd-v2.md S2').
	// Off unless a .vxpbd asset is present and --hood-xpbd asked for it. The defaults are the
	// configuration gate G0 measured as best: 128 Jacobi iterations, one-sided (resist stretch but
	// never compression, worth 0.11 of score), rigid constraints. Compliance is a runtime knob
	// rather than a baked constant because gate G0 established the usable magnitude is around 1e-2,
	// seven orders above the 0..1e-6 range that sweep had covered, and that range is untested.
	bool hoodXpbdRequested{ false };
	bool hoodXpbdAvailable{ false };
	bool hoodXpbdEnabled{ false };
	bool hoodXpbdOneSided{ true };
	// Contacts are the nearest-proxy half-plane projection, folded into the sweep. On by default
	// because that is the configuration gate G0 measured; note this is the ONLY collision handling
	// when XPBD is on, since hood_integrate.comp's own projection is disabled in that case.
	bool hoodXpbdCollision{ true };
	int32_t hoodXpbdIterations{ 128 };
	float hoodXpbdStretchCompliance{ 0.0f };
	float hoodXpbdBendCompliance{ 0.0f };
	float hoodXpbdRelaxation{ 1.0f };
	float hoodXpbdContactOffset{ 0.005f };
	uint32_t hoodXpbdConstraintCount{};
	uint32_t hoodXpbdSlotWidth{};
	uint32_t hoodXpbdStretchCount{};
	std::filesystem::path hoodXpbdPath;
	bool hoodVerifyMode{ false };
	bool hoodVerifyWritten{ false };
	bool hoodStaticBenchmarkMode{ false };
	enum HoodSolver : int32_t { HoodFine15 = 0, HoodToy2L = 1, HoodTinyStudent = 2, HoodPostCvpr = 3 };
	int32_t hoodSolver{ HoodFine15 };
	std::filesystem::path hoodAssetRoot;
	std::filesystem::path hoodModelPath;
	std::filesystem::path hoodToyModelPath;
	std::filesystem::path hoodGoldenPath;
	std::filesystem::path hoodVerifyOutput;
	std::filesystem::path hoodStaticBenchmarkOutput{ "hood_static_timing.csv" };
	std::filesystem::path hoodStabilityOutput;
	std::string hoodMotion{ "ch10032_sprint" };
	float hoodAccumulator{};
	uint32_t hoodFrame{};
	uint32_t hoodCompletedSteps{}, hoodPauseAfterSteps{};
	uint32_t hoodComputeFrame{};
	uint32_t hoodNextFrame{};
	uint32_t hoodRenderFrame{};
	uint32_t hoodFrameCount{}, hoodBoneCount{}, hoodFps{};
	uint32_t hoodCharacterCount{}, hoodCharacterIndexCount{}, hoodProxyCount{};
	uint32_t hoodClothCount{}, hoodClothIndexCount{}, hoodTriangleCount{}, hoodMeshEdgeCount{};
	std::array<uint32_t, 2> hoodCoarseEdgeCounts{};
	std::vector<glm::vec3> hoodRootPositions;
	std::vector<glm::vec3> hoodRestPositionsCpu;
	std::vector<uint32_t> hoodPinMaskCpu, hoodMeshSendersCpu, hoodMeshReceiversCpu;
	std::vector<HoodPlainU3> hoodTrianglesCpu;
	std::vector<HoodPlain3> hoodGoldenPositions;
	std::vector<HoodPlain3> hoodGoldenAcceleration;
	std::vector<uint32_t> hoodGoldenWorld;
	std::vector<double> hoodVerifyStepMaximums;
	std::vector<double> hoodVerifyStepMeans;
	uint32_t hoodGoldenSteps{};
	uint32_t hoodVerifyStep{};
	double hoodVerifyMaximum{};
	double hoodVerifyMeanSum{};
	double hoodVerifyAccelerationMaximum{};
	double hoodVerifyAccelerationMean{};
	uint32_t hoodVerifyWorldMismatches{};
	uint64_t hoodVerifyValueCount{};
	static constexpr uint32_t hoodProcessorBlocks = 15;
	// Cloth-to-body world edges exist only inside this radius. hood_world_nearest.comp seeds
	// its per-lane minimum with it and the feature pass normalises against the same cutoff, so
	// the two must never disagree -- keep it in one place.
	static constexpr float hoodCollisionRadius = 0.03f;
	static constexpr uint32_t hoodTimestampCount = 9 + hoodProcessorBlocks * 2;
	// The integrate stamp keeps the index it had before the XPBD stage existed and XPBD's is
	// appended last, so every column in the existing timing CSVs still means what it meant.
	static constexpr uint32_t hoodIntegrateTimestamp = 8 + hoodProcessorBlocks * 2 - 1;
	static constexpr uint32_t hoodXpbdTimestamp = hoodTimestampCount - 1;
	uint32_t hoodEmbeddingOffset{};
	uint32_t hoodDecoderMlpId{};
	uint32_t hoodLevelEmbeddingOffset{ vhood::noTensor };
	std::array<std::array<uint32_t, 4>, hoodProcessorBlocks> hoodPostEdgeMlpIds{};
	std::array<uint32_t, hoodProcessorBlocks> hoodPostNodeMlpIds{}, hoodPostEdgeMasks{}, hoodPostActiveLevels{};
	uint32_t hoodStaticBenchmarkWarmup{ 5 };
	uint32_t hoodStaticBenchmarkTarget{ 20 };
	uint32_t hoodStaticBenchmarkDiscarded{};
	uint32_t hoodLatentSize{ 128 };
	uint32_t hoodActiveProcessorBlocks{ 15 };
	// "64x4" for the student shipped so far. Derived from the loaded checkpoint so a retrained
	// architecture labels itself correctly in titles and result files without a code change.
	std::string hoodStudentLabel() const { return std::to_string(hoodLatentSize) + "x" + std::to_string(hoodActiveProcessorBlocks); }

	struct HoodBuffers {
		vks::Buffer skinMatrices, characterRestPosition, characterRestNormal, characterBoneIndices, characterBoneWeights;
		vks::Buffer characterPosition, characterNormal, characterUv, characterIndices;
		vks::Buffer proxyRestPosition, proxyRestNormal, proxyBoneIndices, proxyBoneWeights;
		vks::Buffer proxyPosition, proxyNormal, proxyTarget;
		vks::Buffer clothRestPosition, clothBoneIndices, clothBoneWeights, pinTarget, pinMask, mass;
		vks::Buffer clothPosition, clothPrevious, effectivePosition, acceleration, clothTriangles, clothIndices;
		vks::Buffer meshSenders, meshReceivers, csrOffsets, worldObstacle, activeProxy;
		vks::Buffer clothTriangleOffsets, clothTriangleIndices;
		vks::Buffer worldReverseCloth, worldReverseBegin, worldReverseCount;
		vks::Buffer nodeFeatures, meshFeatures, worldDirectFeatures, worldInverseFeatures;
		vks::Buffer vertexLevel;
		std::array<vks::Buffer, 2> coarseSenders, coarseReceivers, coarseOffsets, coarseFeatures;
		vks::Buffer weights, mlpTable, normalizers, toyWeights, toyHidden;
		// XPBD constraint set from the .vxpbd asset, plus the two buffers the sweep needs at
		// runtime. `xpbdLambda` is per (vertex, slot), not per constraint: the fused sweep has both
		// endpoints of a constraint evaluate the same multiplier update, so each keeps its own copy.
		// `xpbdScratch` is the Jacobi ping-pong target -- every vertex has to read the same
		// positions, so updating clothPosition in place would silently become a nondeterministic
		// partial Gauss-Seidel.
		vks::Buffer xpbdPairs, xpbdTargetLength, xpbdWeightSum, xpbdKind;
		vks::Buffer xpbdSlots, xpbdSigns, xpbdIncident, xpbdInverseMass, xpbdMinEdge;
		vks::Buffer xpbdLambda, xpbdScratch;
		std::array<vks::Buffer, 2> nodeLatent, meshLatent, worldDirectLatent, worldInverseLatent;
		std::array<std::array<vks::Buffer, 2>, 2> coarseLatent;
	} hoodBuffers;

	struct HoodLayouts {
		VkDescriptorPool pool{ VK_NULL_HANDLE };
		VkDescriptorSetLayout skinLayout{}, featuresLayout{}, encodeLayout{}, edgeLayout{}, nodeLayout{}, integrateLayout{}, toyLayout{}, graphicsLayout{};
		VkPipelineLayout skinPipeline{}, featuresPipeline{}, encodePipeline{}, edgePipeline{}, nodePipeline{}, integratePipeline{}, toyPipeline{}, graphicsPipeline{};
		VkDescriptorSetLayout worldNearestLayout{}, worldReverseLayout{};
		VkPipelineLayout worldNearestPipeline{}, worldReversePipeline{};
		VkDescriptorSet worldNearestSet{}, worldReverseSet{};
		VkDescriptorSetLayout xpbdLayout{};
		VkPipelineLayout xpbdPipeline{};
		// Two sets: [0] reads clothPosition and writes the scratch buffer, [1] the other way round.
		std::array<VkDescriptorSet, 2> xpbdSets{};
		VkPipeline xpbd{};
		std::array<VkDescriptorSet, maxConcurrentFrames> skinSets{}, featureSets{}, integrateSets{}, graphicsSets{};
		std::array<VkDescriptorSet, maxConcurrentFrames> toySets{};
		VkDescriptorSet encodeSet{};
		std::array<VkDescriptorSet, 2> edgeSets{}, nodeSets{};
		VkPipeline skin{}, features{}, encode{}, edge{}, node{}, integrate{}, toyLayer0{}, toyLayer1{};
		VkPipeline worldNearest{}, worldReverse{};
		VkPipeline sky{}, character{}, cloth{};
		std::array<vks::Buffer, maxConcurrentFrames> skinUniforms, featureUniforms, integrateUniforms, toyUniforms, graphicsUniforms;
		std::array<VkQueryPool, maxConcurrentFrames> queryPools{};
		std::array<bool, maxConcurrentFrames> queryWritten{}, querySimulated{};
	} hood;

	struct HoodTiming {
		double skin{}, features{};
		std::array<double, 4> encoders{};
		std::array<double, hoodProcessorBlocks> edgeBlocks{}, nodeBlocks{};
		double integrate{}, xpbd{}, total{};
		double encodeTotal() const { return encoders[0] + encoders[1] + encoders[2] + encoders[3]; }
		double processorTotal() const {
			double value = 0.0;
			for (uint32_t block = 0; block < hoodProcessorBlocks; ++block) value += edgeBlocks[block] + nodeBlocks[block];
			return value;
		}
	} hoodTiming;
	std::vector<HoodTiming> hoodStaticBenchmarkSamples;

	template <typename T>
	static std::vector<glm::vec4> hoodExpandVec3(std::span<const T> input, float w)
	{
		std::vector<glm::vec4> output;
		output.reserve(input.size());
		for (const auto& value : input) output.emplace_back(value.x, value.y, value.z, w);
		return output;
	}

	void hoodUpload(vks::Buffer& destination, VkBufferUsageFlags usage, const void* data, VkDeviceSize bytes)
	{
		if (!bytes) throw std::runtime_error("Cannot create an empty HOOD runtime buffer");
		vks::Buffer staging;
		VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging, bytes, const_cast<void*>(data)));
		VK_CHECK_RESULT(vulkanDevice->createBuffer(usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &destination, bytes));
		VkCommandBuffer command = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copy{ .size = bytes };
		vkCmdCopyBuffer(command, staging.buffer, destination.buffer, 1, &copy);
		vulkanDevice->flushCommandBuffer(command, queue, true);
		staging.destroy();
	}

	void hoodEmpty(vks::Buffer& destination, VkBufferUsageFlags usage, VkDeviceSize bytes)
	{
		VK_CHECK_RESULT(vulkanDevice->createBuffer(usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &destination, bytes));
	}

	template <typename T>
	void hoodUploadVector(vks::Buffer& destination, VkBufferUsageFlags usage, const std::vector<T>& values)
	{
		hoodUpload(destination, usage, values.data(), static_cast<VkDeviceSize>(values.size()) * sizeof(T));
	}

	VkDescriptorSetLayout hoodMakeLayout(const std::vector<VkDescriptorType>& types)
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		for (uint32_t index = 0; index < types.size(); ++index) bindings.push_back(
			vks::initializers::descriptorSetLayoutBinding(types[index], types[index] == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ? VK_SHADER_STAGE_ALL : VK_SHADER_STAGE_COMPUTE_BIT, index));
		VkDescriptorSetLayout result{};
		auto info = vks::initializers::descriptorSetLayoutCreateInfo(bindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &info, nullptr, &result));
		return result;
	}

	VkPipelineLayout hoodMakePipelineLayout(VkDescriptorSetLayout layout, uint32_t pushBytes = 0)
	{
		auto info = vks::initializers::pipelineLayoutCreateInfo(&layout, 1);
		VkPushConstantRange range{};
		if (pushBytes) {
			range = vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, pushBytes, 0);
			info.pushConstantRangeCount = 1;
			info.pPushConstantRanges = &range;
		}
		VkPipelineLayout result{};
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &info, nullptr, &result));
		return result;
	}

	VkDescriptorSet hoodAllocateSet(VkDescriptorSetLayout layout)
	{
		VkDescriptorSet set{};
		auto info = vks::initializers::descriptorSetAllocateInfo(hood.pool, &layout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &info, &set));
		return set;
	}

	void hoodWriteSet(VkDescriptorSet set, const std::vector<std::pair<VkDescriptorType, vks::Buffer*>>& buffers)
	{
		std::vector<VkWriteDescriptorSet> writes;
		for (uint32_t binding = 0; binding < buffers.size(); ++binding)
			writes.push_back(vks::initializers::writeDescriptorSet(set, buffers[binding].first, binding, &buffers[binding].second->descriptor));
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}

	void hoodLoadAssets()
	{
		if (hoodAssetRoot.empty()) throw std::runtime_error("HOOD requires --asset-root <baked scene directory>");
		if (hoodModelPath.empty()) hoodModelPath = hoodAssetRoot.parent_path().parent_path() / "hood_data" /
			(hoodSolver == HoodPostCvpr ? "postcvpr.vhood" : (hoodSolver == HoodTinyStudent ? "tinyhood64x4.vhood" : "fine15.vhood"));
		const std::string assetStem = hoodGridScene ? "hood_grid64" : "ch10032";
		const auto character = vhood::loadSectioned(hoodAssetRoot / (assetStem + ".vchar"), "VCHAR001", 1);
		const auto animation = vhood::loadSectioned(hoodAssetRoot / (hoodMotion + ".vanim"), "VANIM001", 1);
		const auto cloth = vhood::loadSectioned(hoodAssetRoot / (assetStem + (hoodGridScene ? ".vcloth2" : "_lower.vcloth2")), "VCLTH002", 2);
		vhood::SectionedAsset hierarchy;
		if (hoodSolver == HoodPostCvpr)
			hierarchy = vhood::loadSectioned(hoodAssetRoot / (assetStem + ".postcvpr.vhier"), "VPHIER01", 1);
		const auto modelAsset = vhood::loadTensorAsset(hoodModelPath);
		const auto model = hoodSolver == HoodPostCvpr ? vhood::buildPostCvprGpuModel(modelAsset)
			: (hoodSolver == HoodTinyStudent ? vhood::buildTinyGpuModel(modelAsset) : vhood::buildGpuModel(modelAsset));
		// The student's width and depth come from the checkpoint, so a retrained architecture
		// only needs a new .vhood -- no code change here. Its width selects a shader variant,
		// so it has to be one the build produced.
		if (hoodSolver == HoodTinyStudent) {
			const auto architecture = vhood::inferTinyArchitecture(modelAsset);
			hoodLatentSize = architecture.latent;
			hoodActiveProcessorBlocks = architecture.blocks;
		} else {
			hoodLatentSize = 128u;
			hoodActiveProcessorBlocks = hoodProcessorBlocks;
		}
		hoodPostEdgeMlpIds = model.postEdgeMlpIds;
		hoodPostNodeMlpIds = model.postNodeMlpIds;
		hoodPostEdgeMasks = model.postEdgeMasks;
		hoodPostActiveLevels = model.postActiveLevels;
		if (hoodToyModelPath.empty()) hoodToyModelPath = std::filesystem::path(getShadersPath()) / "gnncloth" / "model.bin";
		const auto toyModel = vgnn::loadModel(hoodToyModelPath);
		if (hoodVerifyMode) {
			if (hoodGoldenPath.empty()) hoodGoldenPath = hoodAssetRoot /
				(hoodSolver == HoodPostCvpr ? "postcvpr_rollout.vhgold" : (hoodSolver == HoodTinyStudent ? "tinyhood64x4_rollout.vhgold" : "fine15_rollout.vhgold"));
			const auto golden = vhood::loadSectioned(hoodGoldenPath, "VHGOLD01", 1);
			const auto goldenInfo = golden.require("info", 4, 4).as<uint32_t>();
			hoodGoldenSteps = std::min(10u, goldenInfo[0]);
			if (goldenInfo[1] != cloth.require("positions", 12).count || !hoodGoldenSteps) throw std::runtime_error("HOOD golden dimensions are invalid");
			const auto positions = golden.require("rollout_pos", 12).as<HoodPlain3>(12);
			hoodGoldenPositions.assign(positions.begin(), positions.begin() + static_cast<size_t>(hoodGoldenSteps) * goldenInfo[1]);
			const auto acceleration = golden.require("first_accel", 12, goldenInfo[1]).as<HoodPlain3>(12);
			hoodGoldenAcceleration.assign(acceleration.begin(), acceleration.end());
			const auto world = golden.require("world_to", 4, goldenInfo[1]).as<uint32_t>(4);
			hoodGoldenWorld.assign(world.begin(), world.end());
		}

		const auto info = character.require("info", 4, 6).as<uint32_t>();
		hoodCharacterCount = info[0];
		hoodCharacterIndexCount = info[1] * 3;
		hoodBoneCount = info[2];
		hoodProxyCount = info[3];
		const auto animationInfo = animation.require("info", 4, 4).as<uint32_t>();
		hoodFrameCount = animationInfo[0];
		if (animationInfo[1] != hoodBoneCount || animationInfo[2] != 30) throw std::runtime_error("VANIM bone count/FPS does not match CH10032 runtime");
		hoodFps = animationInfo[2];
		hoodClothCount = cloth.require("positions", 12).count;
		hoodTriangleCount = cloth.require("triangles", 12).count;
		hoodClothIndexCount = hoodTriangleCount * 3;
		hoodMeshEdgeCount = cloth.require("csr_neighbors", 4).count;
		// The XPBD constraint set is a separate asset rather than new .vcloth2 sections. Its target
		// lengths are calibrated against a teacher rollout, so they may have to be per motion,
		// while .vcloth2 is shared by every motion of a garment -- and adding sections there would
		// change its payload hash, which the existing goldens are pinned to. Missing is not an
		// error: without it the runtime is exactly what it was before.
		vhood::SectionedAsset xpbd;
		if (hoodXpbdPath.empty()) hoodXpbdPath = hoodAssetRoot / (hoodMotion + ".vxpbd");
		hoodXpbdAvailable = std::filesystem::exists(hoodXpbdPath);
		if (hoodXpbdAvailable) {
			xpbd = vhood::loadSectioned(hoodXpbdPath, "VXPBD001", 1);
			const auto xpbdInfo = xpbd.require("info", 4, 4).as<uint32_t>();
			hoodXpbdConstraintCount = xpbdInfo[0];
			hoodXpbdSlotWidth = xpbdInfo[2];
			hoodXpbdStretchCount = xpbdInfo[3];
			if (xpbdInfo[1] != hoodClothCount || !hoodXpbdConstraintCount || !hoodXpbdSlotWidth
				|| hoodXpbdStretchCount > hoodXpbdConstraintCount)
				throw std::runtime_error("VXPBD dimensions do not match the cloth asset");
		}
		hoodXpbdEnabled = hoodXpbdRequested && hoodXpbdAvailable;
		if (hoodSolver == HoodPostCvpr) {
			const auto hierarchyInfo = hierarchy.require("info", 4, 6).as<uint32_t>();
			if (hierarchyInfo[0] != hoodClothCount || hierarchyInfo[5] != 3) throw std::runtime_error("PostCVPR hierarchy dimensions are invalid");
			hoodCoarseEdgeCounts = { hierarchyInfo[2], hierarchyInfo[3] };
		}

		const auto characterPosition = hoodExpandVec3(character.require("render_pos", 12, hoodCharacterCount).as<HoodPlain3>(12), 1.0f);
		const auto characterNormal = hoodExpandVec3(character.require("render_nrm", 12, hoodCharacterCount).as<HoodPlain3>(12), 0.0f);
		const auto proxyPosition = hoodExpandVec3(character.require("proxy_pos", 12, hoodProxyCount).as<HoodPlain3>(12), 1.0f);
		const auto proxyNormal = hoodExpandVec3(character.require("proxy_nrm", 12, hoodProxyCount).as<HoodPlain3>(12), 0.0f);
		const auto clothPosition = hoodExpandVec3(cloth.require("positions", 12, hoodClothCount).as<HoodPlain3>(12), 1.0f);
		hoodRestPositionsCpu.reserve(clothPosition.size());
		for (const auto& position : clothPosition) hoodRestPositionsCpu.emplace_back(position);
		const auto pins = cloth.require("pin_mask", 4, hoodClothCount).as<uint32_t>(4);
		hoodPinMaskCpu.assign(pins.begin(), pins.end());
		const auto triangles = cloth.require("triangles", 12, hoodTriangleCount).as<HoodPlainU3>(12);
		hoodTrianglesCpu.assign(triangles.begin(), triangles.end());
		const auto roots = animation.require("root_pos", 12, hoodFrameCount).as<HoodPlain3>(12);
		hoodRootPositions.reserve(roots.size());
		for (const auto& root : roots) hoodRootPositions.emplace_back(root.x, root.y, root.z);

		auto uploadView = [&](vks::Buffer& buffer, VkBufferUsageFlags usage, const vhood::ByteView& view) { hoodUpload(buffer, usage, view.bytes.data(), view.bytes.size()); };
		const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		hoodUploadVector(hoodBuffers.characterRestPosition, storage, characterPosition);
		hoodUploadVector(hoodBuffers.characterRestNormal, storage, characterNormal);
		uploadView(hoodBuffers.characterBoneIndices, storage, character.require("bone_idx", 48, hoodCharacterCount));
		uploadView(hoodBuffers.characterBoneWeights, storage, character.require("bone_weight", 48, hoodCharacterCount));
		uploadView(hoodBuffers.characterUv, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, character.require("render_uv", 8, hoodCharacterCount));
		uploadView(hoodBuffers.characterIndices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, character.require("render_tri", 12));
		hoodUploadVector(hoodBuffers.proxyRestPosition, storage, proxyPosition);
		hoodUploadVector(hoodBuffers.proxyRestNormal, storage, proxyNormal);
		uploadView(hoodBuffers.proxyBoneIndices, storage, character.require("proxy_bone_idx", 48, hoodProxyCount));
		uploadView(hoodBuffers.proxyBoneWeights, storage, character.require("proxy_weight", 48, hoodProxyCount));
		hoodUploadVector(hoodBuffers.clothRestPosition, storage, clothPosition);
		uploadView(hoodBuffers.clothBoneIndices, storage, cloth.require("bone_idx", 48, hoodClothCount));
		uploadView(hoodBuffers.clothBoneWeights, storage, cloth.require("bone_weight", 48, hoodClothCount));
		uploadView(hoodBuffers.pinMask, storage, cloth.require("pin_mask", 4, hoodClothCount));
		uploadView(hoodBuffers.mass, storage, cloth.require("mass", 4, hoodClothCount));
		uploadView(hoodBuffers.clothTriangles, storage, cloth.require("triangles", 12, hoodTriangleCount));
		uploadView(hoodBuffers.clothIndices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, cloth.require("triangles", 12, hoodTriangleCount));
		uploadView(hoodBuffers.skinMatrices, storage, animation.require("skin_matrices", 48, hoodFrameCount * hoodBoneCount));

		const auto offsets = cloth.require("csr_offsets", 4, hoodClothCount + 1).as<uint32_t>();
		const auto senders = cloth.require("csr_neighbors", 4, hoodMeshEdgeCount).as<uint32_t>();
		std::vector<uint32_t> meshSenders(senders.begin(), senders.end()), meshReceivers;
		meshReceivers.reserve(hoodMeshEdgeCount);
		for (uint32_t receiver = 0; receiver < hoodClothCount; ++receiver)
			for (uint32_t edge = offsets[receiver]; edge < offsets[receiver + 1]; ++edge) meshReceivers.push_back(receiver);
		hoodMeshSendersCpu = meshSenders;
		hoodMeshReceiversCpu = meshReceivers;
		hoodUploadVector(hoodBuffers.meshSenders, storage, meshSenders);
		hoodUploadVector(hoodBuffers.meshReceivers, storage, meshReceivers);
		uploadView(hoodBuffers.csrOffsets, storage, cloth.require("csr_offsets", 4, hoodClothCount + 1));

		// clothNormal used to walk every triangle in the mesh for every vertex, which is
		// 1377 x 2570 iterations on CH10032 spread over only 11 workgroups. The topology is
		// static, so derive the vertex -> incident-triangle CSR once here instead. Nothing is
		// read from the asset that was not already there, so no format change or rebake.
		// Each vertex's list stays in ascending triangle order, and a triangle that names the
		// same vertex twice is stored once -- exactly what the old `ids.x != vertex && ...`
		// test did -- so the cross-product accumulation order is unchanged bit for bit.
		{
			std::vector<uint32_t> triangleCounts(hoodClothCount + 1, 0);
			auto forEachIncidentVertex = [&](const HoodPlainU3& triangle, auto&& visit) {
				const uint32_t ids[3]{ triangle.x, triangle.y, triangle.z };
				for (uint32_t corner = 0; corner < 3; ++corner) {
					if (ids[corner] >= hoodClothCount) throw std::runtime_error("Cloth triangle references a vertex outside the cloth");
					bool duplicate = false;
					for (uint32_t earlier = 0; earlier < corner; ++earlier) duplicate = duplicate || ids[earlier] == ids[corner];
					if (!duplicate) visit(ids[corner]);
				}
			};
			for (const auto& triangle : hoodTrianglesCpu) forEachIncidentVertex(triangle, [&](uint32_t vertex) { ++triangleCounts[vertex]; });
			std::vector<uint32_t> triangleOffsets(hoodClothCount + 1, 0);
			for (uint32_t vertex = 0; vertex < hoodClothCount; ++vertex) triangleOffsets[vertex + 1] = triangleOffsets[vertex] + triangleCounts[vertex];
			std::vector<uint32_t> triangleIndices(triangleOffsets[hoodClothCount], 0);
			std::vector<uint32_t> cursor(triangleOffsets.begin(), triangleOffsets.end() - 1);
			for (uint32_t triangle = 0; triangle < hoodTriangleCount; ++triangle)
				forEachIncidentVertex(hoodTrianglesCpu[triangle], [&](uint32_t vertex) { triangleIndices[cursor[vertex]++] = triangle; });
			hoodUploadVector(hoodBuffers.clothTriangleOffsets, storage, triangleOffsets);
			// A cloth vertex with no incident triangle would leave the buffer empty and
			// hoodUpload rejects zero bytes; a mesh like that would also break clothNormal.
			if (triangleIndices.empty()) throw std::runtime_error("Cloth mesh has no vertex-triangle incidence");
			hoodUploadVector(hoodBuffers.clothTriangleIndices, storage, triangleIndices);
		}
		if (hoodSolver == HoodPostCvpr) {
			uploadView(hoodBuffers.vertexLevel, storage, hierarchy.require("vertex_level", 4, hoodClothCount));
			for (uint32_t level = 0; level < 2; ++level) {
				const std::string prefix = "c" + std::to_string(level) + "_";
				uploadView(hoodBuffers.coarseSenders[level], storage, hierarchy.require(prefix + "senders", 4, hoodCoarseEdgeCounts[level]));
				uploadView(hoodBuffers.coarseReceivers[level], storage, hierarchy.require(prefix + "receivers", 4, hoodCoarseEdgeCounts[level]));
				uploadView(hoodBuffers.coarseOffsets[level], storage, hierarchy.require(prefix + "offsets", 4, hoodClothCount + 1));
			}
		}
		hoodUploadVector(hoodBuffers.weights, storage, model.weights);
		hoodUploadVector(hoodBuffers.toyWeights, storage, toyModel.payload);
		std::vector<glm::uvec4> table;
		table.reserve(model.mlps.size() * 3);
		for (const auto& mlp : model.mlps) {
			table.emplace_back(mlp.w0, mlp.b0, mlp.w1, mlp.b1);
			table.emplace_back(mlp.w2, mlp.b2, mlp.layerNormWeight, mlp.layerNormBias);
			table.emplace_back(mlp.inputDimension, mlp.outputDimension, mlp.hasLayerNorm, 0);
		}
		hoodUploadVector(hoodBuffers.mlpTable, storage, table);
		hoodUploadVector(hoodBuffers.normalizers, storage, model.normalizers);
		if (hoodXpbdAvailable) {
			uploadView(hoodBuffers.xpbdPairs, storage, xpbd.require("pairs", 8, hoodXpbdConstraintCount));
			uploadView(hoodBuffers.xpbdTargetLength, storage, xpbd.require("target_len", 4, hoodXpbdConstraintCount));
			uploadView(hoodBuffers.xpbdWeightSum, storage, xpbd.require("weight_sum", 4, hoodXpbdConstraintCount));
			uploadView(hoodBuffers.xpbdKind, storage, xpbd.require("kind", 4, hoodXpbdConstraintCount));
			const uint32_t slotCount = hoodClothCount * hoodXpbdSlotWidth;
			uploadView(hoodBuffers.xpbdSlots, storage, xpbd.require("slots", 4, slotCount));
			uploadView(hoodBuffers.xpbdSigns, storage, xpbd.require("signs", 4, slotCount));
			uploadView(hoodBuffers.xpbdIncident, storage, xpbd.require("incident", 4, hoodClothCount));
			uploadView(hoodBuffers.xpbdInverseMass, storage, xpbd.require("inverse_mass", 4, hoodClothCount));
			// Baked for the per-vertex trust region in gnn-xpbd-v2.md section 7.1, which the first
			// kernel does not implement. Uploaded anyway so enabling it later needs no re-bake.
			uploadView(hoodBuffers.xpbdMinEdge, storage, xpbd.require("min_edge", 4, hoodClothCount));
			hoodEmpty(hoodBuffers.xpbdLambda, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, slotCount * sizeof(float));
			hoodEmpty(hoodBuffers.xpbdScratch, storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hoodClothCount * sizeof(glm::vec4));
		}

		hoodEmpty(hoodBuffers.characterPosition, storage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, hoodCharacterCount * sizeof(glm::vec4));
		hoodEmpty(hoodBuffers.characterNormal, storage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, hoodCharacterCount * sizeof(glm::vec4));
		hoodEmpty(hoodBuffers.proxyPosition, storage, hoodProxyCount * sizeof(glm::vec4));
		hoodEmpty(hoodBuffers.proxyNormal, storage, hoodProxyCount * sizeof(glm::vec4));
		hoodEmpty(hoodBuffers.proxyTarget, storage, hoodProxyCount * sizeof(glm::vec4));
		hoodEmpty(hoodBuffers.pinTarget, storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hoodClothCount * sizeof(glm::vec4));
		// The XPBD sweep ping-pongs into xpbdScratch, so clothPosition may need a copy back, and on
		// the settle step the corrected position also has to replace clothPrevious. Both are
		// transfers, hence the extra usage bits.
		hoodEmpty(hoodBuffers.clothPosition, storage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, hoodClothCount * sizeof(glm::vec4));
		hoodEmpty(hoodBuffers.clothPrevious, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, hoodClothCount * sizeof(glm::vec4));
		hoodEmpty(hoodBuffers.effectivePosition, storage, hoodClothCount * sizeof(glm::vec4));
		hoodEmpty(hoodBuffers.acceleration, storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hoodClothCount * sizeof(glm::vec4));
		hoodEmpty(hoodBuffers.toyHidden, storage, static_cast<VkDeviceSize>(hoodClothCount) * 4 * sizeof(glm::vec4));
		hoodEmpty(hoodBuffers.worldObstacle, storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hoodClothCount * sizeof(uint32_t));
		hoodEmpty(hoodBuffers.activeProxy, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, hoodProxyCount * sizeof(uint32_t));
		// Proxy -> cloth transpose of worldObstacle. There is at most one world edge per cloth
		// vertex, so every entry fits in clothCount slots.
		hoodEmpty(hoodBuffers.worldReverseCloth, storage, hoodClothCount * sizeof(uint32_t));
		hoodEmpty(hoodBuffers.worldReverseBegin, storage, hoodProxyCount * sizeof(uint32_t));
		hoodEmpty(hoodBuffers.worldReverseCount, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, hoodProxyCount * sizeof(uint32_t));
		const VkBufferUsageFlags debugStorage = storage | (hoodVerifyMode ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0);
		const uint32_t nodeFeatureDimension = hoodSolver == HoodPostCvpr ? 24u : 20u;
		hoodEmpty(hoodBuffers.nodeFeatures, debugStorage, static_cast<VkDeviceSize>(hoodClothCount + hoodProxyCount) * nodeFeatureDimension * sizeof(float));
		hoodEmpty(hoodBuffers.meshFeatures, debugStorage, static_cast<VkDeviceSize>(hoodMeshEdgeCount) * 12 * sizeof(float));
		hoodEmpty(hoodBuffers.worldDirectFeatures, debugStorage, static_cast<VkDeviceSize>(hoodClothCount) * 9 * sizeof(float));
		hoodEmpty(hoodBuffers.worldInverseFeatures, debugStorage, static_cast<VkDeviceSize>(hoodClothCount) * 9 * sizeof(float));
		if (hoodSolver == HoodPostCvpr) for (uint32_t level = 0; level < 2; ++level)
			hoodEmpty(hoodBuffers.coarseFeatures[level], debugStorage, static_cast<VkDeviceSize>(hoodCoarseEdgeCounts[level]) * 12 * sizeof(float));
		for (uint32_t ping = 0; ping < 2; ++ping) {
			hoodEmpty(hoodBuffers.nodeLatent[ping], debugStorage, static_cast<VkDeviceSize>(hoodClothCount + hoodProxyCount) * hoodLatentSize * sizeof(float));
			hoodEmpty(hoodBuffers.meshLatent[ping], storage, static_cast<VkDeviceSize>(hoodMeshEdgeCount) * hoodLatentSize * sizeof(float));
			hoodEmpty(hoodBuffers.worldDirectLatent[ping], storage, static_cast<VkDeviceSize>(hoodClothCount) * hoodLatentSize * sizeof(float));
			hoodEmpty(hoodBuffers.worldInverseLatent[ping], storage, static_cast<VkDeviceSize>(hoodClothCount) * hoodLatentSize * sizeof(float));
			if (hoodSolver == HoodPostCvpr) for (uint32_t level = 0; level < 2; ++level)
				hoodEmpty(hoodBuffers.coarseLatent[level][ping], storage, static_cast<VkDeviceSize>(hoodCoarseEdgeCounts[level]) * hoodLatentSize * sizeof(float));
		}
		hoodEmbeddingOffset = model.embeddingOffset;
		hoodDecoderMlpId = model.decoderMlpId;
		hoodLevelEmbeddingOffset = model.vertexLevelEmbeddingOffset;
	}

	void hoodPrepareDescriptors()
	{
		const std::vector<VkDescriptorPoolSize> sizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 400),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16),
		};
		auto poolInfo = vks::initializers::descriptorPoolCreateInfo(sizes, 32);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &hood.pool));
		auto storageTypes = [](uint32_t count) { return std::vector<VkDescriptorType>(count, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); };
		auto skinTypes = storageTypes(22); skinTypes[21] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		// The two vertex-triangle CSR buffers are appended last so every existing binding
		// index in the feature shaders keeps its number.
		auto featureTypes = storageTypes(hoodSolver == HoodPostCvpr ? 31u : 24u);
		featureTypes[hoodSolver == HoodPostCvpr ? 28u : 20u] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		auto integrateTypes = storageTypes(14); integrateTypes[13] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		auto toyTypes = storageTypes(10); toyTypes[9] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		hood.skinLayout = hoodMakeLayout(skinTypes); hood.featuresLayout = hoodMakeLayout(featureTypes);
		hood.encodeLayout = hoodMakeLayout(storageTypes(hoodSolver == HoodPostCvpr ? 14u : 10u));
		hood.edgeLayout = hoodMakeLayout(storageTypes(hoodSolver == HoodPostCvpr ? 21u : 12u));
		hood.nodeLayout = hoodMakeLayout(storageTypes(hoodSolver == HoodPostCvpr ? 23u : 16u)); hood.integrateLayout = hoodMakeLayout(integrateTypes);
		hood.toyLayout = hoodMakeLayout(toyTypes);
		// Nearest-proxy search: one workgroup per cloth vertex, shared by every non-toy solver.
		// Its three scalars come in as push constants so it does not have to care whether the
		// solver uses HoodFeatureParams or HoodPostFeatureParams.
		hood.worldNearestLayout = hoodMakeLayout(storageTypes(6));
		hood.worldReverseLayout = hoodMakeLayout(storageTypes(4));
		// All fourteen XPBD bindings are storage buffers; its scalars are push constants so the
		// per-iteration state (nothing) and the per-step state (timestep, compliance) do not need a
		// per-frame uniform buffer.
		if (hoodXpbdAvailable) hood.xpbdLayout = hoodMakeLayout(storageTypes(14));
		hood.graphicsLayout = hoodMakeLayout({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER });
		hood.skinPipeline = hoodMakePipelineLayout(hood.skinLayout); hood.featuresPipeline = hoodMakePipelineLayout(hood.featuresLayout);
		hood.encodePipeline = hoodMakePipelineLayout(hood.encodeLayout, 16); hood.edgePipeline = hoodMakePipelineLayout(hood.edgeLayout, hoodSolver == HoodPostCvpr ? 20u : 16u);
		hood.nodePipeline = hoodMakePipelineLayout(hood.nodeLayout, hoodSolver == HoodPostCvpr ? 20u : 16u); hood.integratePipeline = hoodMakePipelineLayout(hood.integrateLayout);
		hood.toyPipeline = hoodMakePipelineLayout(hood.toyLayout);
		hood.worldNearestPipeline = hoodMakePipelineLayout(hood.worldNearestLayout, 12);
		hood.worldReversePipeline = hoodMakePipelineLayout(hood.worldReverseLayout, 4);
		if (hoodXpbdAvailable) hood.xpbdPipeline = hoodMakePipelineLayout(hood.xpbdLayout, sizeof(HoodXpbdPush));
		hood.graphicsPipeline = hoodMakePipelineLayout(hood.graphicsLayout);

		for (uint32_t frame = 0; frame < maxConcurrentFrames; ++frame) {
			for (auto* buffer : { &hood.skinUniforms[frame], &hood.featureUniforms[frame], &hood.integrateUniforms[frame], &hood.toyUniforms[frame], &hood.graphicsUniforms[frame] }) {
				const VkDeviceSize size = buffer == &hood.skinUniforms[frame] ? sizeof(HoodSkinParams) : buffer == &hood.featureUniforms[frame]
					? (hoodSolver == HoodPostCvpr ? sizeof(HoodPostFeatureParams) : sizeof(HoodFeatureParams))
					: buffer == &hood.integrateUniforms[frame] ? sizeof(HoodIntegrateParams) : buffer == &hood.toyUniforms[frame] ? sizeof(HoodToyParams) : sizeof(HoodGraphicsUniform);
				VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, size));
				VK_CHECK_RESULT(buffer->map());
			}
			hood.skinSets[frame] = hoodAllocateSet(hood.skinLayout);
			hoodWriteSet(hood.skinSets[frame], {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.skinMatrices},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.characterRestPosition},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.characterRestNormal},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.characterBoneIndices},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.characterBoneWeights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.characterPosition},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.characterNormal},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyRestPosition},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyRestNormal},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyBoneIndices},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyBoneWeights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyPosition},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyNormal},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyTarget},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothRestPosition},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothBoneIndices},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothBoneWeights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinTarget},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPosition},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPrevious},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinMask},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,&hood.skinUniforms[frame]} });
			hood.featureSets[frame] = hoodAllocateSet(hood.featuresLayout);
			if (hoodSolver == HoodPostCvpr) hoodWriteSet(hood.featureSets[frame], {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPosition},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPrevious},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinTarget},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinMask},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothTriangles},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshSenders},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshReceivers},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothRestPosition},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.mass},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyPosition},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyTarget},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyNormal},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldObstacle},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeFeatures},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshFeatures},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectFeatures},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseFeatures},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.weights},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.normalizers},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.effectivePosition},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.activeProxy},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.vertexLevel},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseSenders[0]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseReceivers[0]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseSenders[1]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseReceivers[1]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseFeatures[0]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseFeatures[1]},
				{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,&hood.featureUniforms[frame]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothTriangleOffsets},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothTriangleIndices} });
			else hoodWriteSet(hood.featureSets[frame], {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPosition},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPrevious},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinTarget},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinMask},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothTriangles},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshSenders},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshReceivers},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothRestPosition},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.mass},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyPosition},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyTarget},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyNormal},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldObstacle},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeFeatures},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshFeatures},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectFeatures},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseFeatures},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.weights},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.normalizers},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.effectivePosition},
				{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,&hood.featureUniforms[frame]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.activeProxy},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothTriangleOffsets},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothTriangleIndices} });
			hood.integrateSets[frame] = hoodAllocateSet(hood.integrateLayout);
			hoodWriteSet(hood.integrateSets[frame], {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.weights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.mlpTable},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.normalizers},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeLatent[hoodActiveProcessorBlocks & 1u]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPosition},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPrevious},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.effectivePosition},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinTarget},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinMask},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.acceleration},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldObstacle},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyTarget},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyNormal},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,&hood.integrateUniforms[frame]} });
			hood.toySets[frame] = hoodAllocateSet(hood.toyLayout);
			hoodWriteSet(hood.toySets[frame], {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPosition},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPrevious},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinTarget},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinMask},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.csrOffsets},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshSenders},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.toyWeights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.toyHidden},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.acceleration},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,&hood.toyUniforms[frame]} });
			hood.graphicsSets[frame] = hoodAllocateSet(hood.graphicsLayout);
			hoodWriteSet(hood.graphicsSets[frame], { {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,&hood.graphicsUniforms[frame]} });
		}

		hood.encodeSet = hoodAllocateSet(hood.encodeLayout);
		hood.worldNearestSet = hoodAllocateSet(hood.worldNearestLayout);
		hoodWriteSet(hood.worldNearestSet, {
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.clothPosition},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinTarget},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.pinMask},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyPosition},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldObstacle},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.activeProxy} });
		hood.worldReverseSet = hoodAllocateSet(hood.worldReverseLayout);
		hoodWriteSet(hood.worldReverseSet, {
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldObstacle},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldReverseCloth},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldReverseBegin},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldReverseCount} });
		if (hoodXpbdAvailable) for (uint32_t ping = 0; ping < 2; ++ping) {
			// ping 0 reads clothPosition and writes the scratch buffer; ping 1 is the reverse.
			auto* read = ping == 0 ? &hoodBuffers.clothPosition : &hoodBuffers.xpbdScratch;
			auto* write = ping == 0 ? &hoodBuffers.xpbdScratch : &hoodBuffers.clothPosition;
			hood.xpbdSets[ping] = hoodAllocateSet(hood.xpbdLayout);
			hoodWriteSet(hood.xpbdSets[ping], {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.xpbdPairs},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.xpbdTargetLength},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.xpbdWeightSum},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.xpbdKind},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.xpbdSlots},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.xpbdSigns},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.xpbdIncident},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.xpbdInverseMass},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.xpbdLambda},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,read},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,write},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldObstacle},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyTarget},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.proxyNormal} });
		}
		if (hoodSolver == HoodPostCvpr) hoodWriteSet(hood.encodeSet, {
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.weights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.mlpTable},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeFeatures},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshFeatures},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseFeatures[0]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseFeatures[1]},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectFeatures},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseFeatures},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeLatent[0]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshLatent[0]},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseLatent[0][0]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseLatent[1][0]},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectLatent[0]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseLatent[0]} });
		else hoodWriteSet(hood.encodeSet, {
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.weights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.mlpTable},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeFeatures},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshFeatures},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectFeatures},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseFeatures},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeLatent[0]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshLatent[0]},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectLatent[0]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseLatent[0]} });
		for (uint32_t ping = 0; ping < 2; ++ping) {
			const uint32_t pong = 1 - ping;
			hood.edgeSets[ping] = hoodAllocateSet(hood.edgeLayout);
			if (hoodSolver == HoodPostCvpr) hoodWriteSet(hood.edgeSets[ping], {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.weights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.mlpTable},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeLatent[ping]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshLatent[ping]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseLatent[0][ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseLatent[1][ping]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseLatent[ping]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseLatent[0][pong]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseLatent[1][pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectLatent[pong]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshSenders},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshReceivers},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseSenders[0]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseReceivers[0]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseSenders[1]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseReceivers[1]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldObstacle},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.vertexLevel} });
			else hoodWriteSet(hood.edgeSets[ping], {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.weights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.mlpTable},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshLatent[ping]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseLatent[ping]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshLatent[pong]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseLatent[pong]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshSenders},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshReceivers},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldObstacle} });
				hood.nodeSets[ping] = hoodAllocateSet(hood.nodeLayout);
			if (hoodSolver == HoodPostCvpr) hoodWriteSet(hood.nodeSets[ping], {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.weights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.mlpTable},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseLatent[0][ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseLatent[0][pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseLatent[1][ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseLatent[1][pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.csrOffsets},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseOffsets[0]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.coarseOffsets[1]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldObstacle},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.vertexLevel},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.activeProxy},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldReverseCloth},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldReverseBegin},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldReverseCount} });
			else hoodWriteSet(hood.nodeSets[ping], {
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.weights},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.mlpTable},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.nodeLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.meshLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldDirectLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseLatent[ping]},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldInverseLatent[pong]},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.csrOffsets},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldObstacle},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.activeProxy},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldReverseCloth},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldReverseBegin},
				{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,&hoodBuffers.worldReverseCount} });
		}
	}

	void hoodPreparePipelines()
	{
		auto computePipeline = [&](const char* shader, VkPipelineLayout layout, VkPipeline& target) {
			auto info = vks::initializers::computePipelineCreateInfo(layout, 0);
			info.stage = loadShader(getShadersPath() + std::string("gnncloth/") + shader, VK_SHADER_STAGE_COMPUTE_BIT);
			VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &info, nullptr, &target));
		};
		computePipeline("hood_skin.comp.spv", hood.skinPipeline, hood.skin);
		computePipeline("hood_world_nearest.comp.spv", hood.worldNearestPipeline, hood.worldNearest);
		computePipeline("hood_world_reverse.comp.spv", hood.worldReversePipeline, hood.worldReverse);
		if (hoodXpbdAvailable) computePipeline("hood_xpbd.comp.spv", hood.xpbdPipeline, hood.xpbd);
		computePipeline(hoodSolver == HoodPostCvpr ? "postcvpr_features.comp.spv" : "hood_features.comp.spv", hood.featuresPipeline, hood.features);
		// A student's latent width is its workgroup size, so each width is a separate SPIR-V
		// module built by tools/compile_shaders.py. 64 keeps the unprefixed names it has always
		// had; narrower widths get a prefix.
		const std::string tiny = hoodLatentSize == 64 ? "tinyhood_" : ("tiny" + std::to_string(hoodLatentSize) + "_tinyhood_");
		auto studentShader = [&](const char* postCvpr, const std::string& stem, const char* teacher) {
			return hoodSolver == HoodPostCvpr ? std::string(postCvpr)
				: (hoodSolver == HoodTinyStudent ? tiny + stem : std::string(teacher));
		};
		computePipeline(studentShader("postcvpr_encode.comp.spv", "encode.comp.spv", "hood_encode.comp.spv").c_str(), hood.encodePipeline, hood.encode);
		computePipeline(studentShader("postcvpr_edge_update.comp.spv", "edge_update.comp.spv", "hood_edge_update.comp.spv").c_str(), hood.edgePipeline, hood.edge);
		computePipeline(studentShader("postcvpr_node_update.comp.spv", "node_update.comp.spv", "hood_node_update.comp.spv").c_str(), hood.nodePipeline, hood.node);
		computePipeline(studentShader("postcvpr_integrate.comp.spv", "integrate.comp.spv", "hood_integrate.comp.spv").c_str(), hood.integratePipeline, hood.integrate);
		computePipeline("hood_toy_layer0.comp.spv", hood.toyPipeline, hood.toyLayer0);
		computePipeline("hood_toy_layer1.comp.spv", hood.toyPipeline, hood.toyLayer1);

		VkPipelineInputAssemblyStateCreateInfo assembly = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo raster = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachment = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo blend = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachment);
		VkPipelineDepthStencilStateCreateInfo depth = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewport = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisample = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> states{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamic = vks::initializers::pipelineDynamicStateCreateInfo(states);
		VkPipelineVertexInputStateCreateInfo vertexInput = vks::initializers::pipelineVertexInputStateCreateInfo();
		auto pipeline = vks::initializers::pipelineCreateInfo(hood.graphicsPipeline, renderPass);
		pipeline.pInputAssemblyState = &assembly; pipeline.pRasterizationState = &raster; pipeline.pColorBlendState = &blend;
		pipeline.pMultisampleState = &multisample; pipeline.pViewportState = &viewport; pipeline.pDepthStencilState = &depth; pipeline.pDynamicState = &dynamic;
		pipeline.pVertexInputState = &vertexInput; pipeline.stageCount = 2;

		std::array<VkPipelineShaderStageCreateInfo, 2> shaders{
			loadShader(getShadersPath() + "gnncloth/hood_character.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "gnncloth/hood_character.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT) };
		std::array<VkVertexInputBindingDescription, 3> charBindings{
			vks::initializers::vertexInputBindingDescription(0, 16, VK_VERTEX_INPUT_RATE_VERTEX),
			vks::initializers::vertexInputBindingDescription(1, 16, VK_VERTEX_INPUT_RATE_VERTEX),
			vks::initializers::vertexInputBindingDescription(2, 8, VK_VERTEX_INPUT_RATE_VERTEX) };
		std::array<VkVertexInputAttributeDescription, 3> charAttributes{
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),
			vks::initializers::vertexInputAttributeDescription(1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0),
			vks::initializers::vertexInputAttributeDescription(2, 2, VK_FORMAT_R32G32_SFLOAT, 0) };
		vertexInput.vertexBindingDescriptionCount = 3; vertexInput.pVertexBindingDescriptions = charBindings.data();
		vertexInput.vertexAttributeDescriptionCount = 3; vertexInput.pVertexAttributeDescriptions = charAttributes.data();
		pipeline.pStages = shaders.data();
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipeline, nullptr, &hood.character));

		shaders = { loadShader(getShadersPath() + "gnncloth/hood_cloth.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "gnncloth/hood_cloth.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT) };
		const auto clothBinding = vks::initializers::vertexInputBindingDescription(0, 16, VK_VERTEX_INPUT_RATE_VERTEX);
		const auto clothAttribute = vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0);
		vertexInput.vertexBindingDescriptionCount = 1; vertexInput.pVertexBindingDescriptions = &clothBinding;
		vertexInput.vertexAttributeDescriptionCount = 1; vertexInput.pVertexAttributeDescriptions = &clothAttribute;
		pipeline.pStages = shaders.data();
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipeline, nullptr, &hood.cloth));

		shaders = { loadShader(getShadersPath() + "gnncloth/sky.vert.spv", VK_SHADER_STAGE_VERTEX_BIT), loadShader(getShadersPath() + "gnncloth/sky.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT) };
		vertexInput.vertexBindingDescriptionCount = 0; vertexInput.vertexAttributeDescriptionCount = 0;
		VkPipelineDepthStencilStateCreateInfo skyDepth = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS);
		pipeline.pDepthStencilState = &skyDepth; pipeline.pStages = shaders.data();
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipeline, nullptr, &hood.sky));
		for (uint32_t frame = 0; frame < maxConcurrentFrames; ++frame) {
			VkQueryPoolCreateInfo queryInfo{ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = hoodTimestampCount };
			VK_CHECK_RESULT(vkCreateQueryPool(device, &queryInfo, nullptr, &hood.queryPools[frame]));
		}
	}

	void hoodPrepare()
	{
		hoodLoadAssets();
		hoodPrepareDescriptors();
		hoodPreparePipelines();
		camera.type = Camera::CameraType::lookat;
		// Baked CH10032 assets are conventional Y-up. The projection is flipped
		// when copied to the real-scene graphics UBO; keeping Camera itself in the
		// upstream convention preserves its stable translation/root-follow logic.
		camera.setPerspective(55.0f, static_cast<float>(width) / static_cast<float>(height), 0.05f, 256.0f);
		camera.setRotation(glm::vec3(-5.0f, 0.0f, 0.0f));
		std::cout << (hoodGridScene ? "HOOD grid64 sphere scene: " : "CH10032 native scene: ") << hoodCharacterCount
			<< (hoodGridScene ? " obstacle" : " body") << " vertices, " << hoodClothCount
			<< " cloth vertices, " << hoodBoneCount << " core bones, " << hoodFrameCount << " frames @ " << hoodFps << " Hz, solver "
			<< (hoodSolver == HoodToy2L ? "Toy2L" : (hoodSolver == HoodTinyStudent ? "TinyHOOD" + hoodStudentLabel() : (hoodSolver == HoodPostCvpr ? "PostCVPR" : "Fine15"))) << "\n";
	}

	void hoodAdvance()
	{
		hoodSimulateFrame = false;
		if (hoodRequestReset) {
			hoodFrame = hoodComputeFrame = hoodNextFrame = hoodRenderFrame = 0;
			hoodCompletedSteps = 0;
			hoodAccumulator = 0.0f;
			hoodFirstStep = true;
			return;
		}
		if (hoodPaused) { hoodComputeFrame = hoodNextFrame = hoodRenderFrame = hoodFrame; return; }
		// Automated validation has no presentation pacing, so advance exactly one
		// 30 Hz simulation step per benchmark frame instead of depending on wall time.
		hoodAccumulator += benchmark.active ? 1.0f / static_cast<float>(hoodFps) : frameTimer;
		if (hoodAccumulator < 1.0f / static_cast<float>(hoodFps)) { hoodComputeFrame = hoodNextFrame = hoodRenderFrame = hoodFrame; return; }
		hoodAccumulator = std::fmod(hoodAccumulator, 1.0f / static_cast<float>(hoodFps));
		if (hoodFrameCount == 1) {
			hoodComputeFrame = hoodNextFrame = hoodRenderFrame = 0;
			hoodSimulateFrame = true;
			return;
		}
		hoodComputeFrame = hoodFrame;
		hoodNextFrame = hoodFrame + 1;
		if (hoodNextFrame >= hoodFrameCount) {
			hoodFrame = hoodComputeFrame = hoodNextFrame = hoodRenderFrame = 0;
			hoodRequestReset = true;
			hoodFirstStep = true;
			return;
		}
		hoodRenderFrame = hoodNextFrame;
		hoodSimulateFrame = true;
	}

	void hoodUpdateUniforms()
	{
		HoodSkinParams skin{ hoodComputeFrame, hoodNextFrame, hoodBoneCount, hoodCharacterCount, hoodProxyCount, hoodClothCount,
			hoodRequestReset ? 1u : 0u, hoodRenderFrame };
		std::memcpy(hood.skinUniforms[currentBuffer].mapped, &skin, sizeof(skin));
		const float material0 = static_cast<float>((std::log(3.9625778333333325e-5) - std::log(6.370782056371576e-8)) / (std::log(0.0013139737991266374) - std::log(6.370782056371576e-8)));
		const float material1 = static_cast<float>((std::log(23600.0) - std::log(15909.0)) / (std::log(63636.0) - std::log(15909.0)));
		const float material2 = static_cast<float>((44400.0 - 3535.414406069427) / (93333.73508005822 - 3535.414406069427));
		if (hoodSolver == HoodPostCvpr) {
			HoodPostFeatureParams features{ hoodClothCount, hoodProxyCount, hoodTriangleCount, hoodMeshEdgeCount,
				hoodCoarseEdgeCounts[0], hoodCoarseEdgeCounts[1], hoodEmbeddingOffset, hoodLevelEmbeddingOffset,
				hoodFirstStep ? 1.0f / 3.0f : 1.0f / 30.0f, hoodCollisionRadius, material0, material1, material2 };
			std::memcpy(hood.featureUniforms[currentBuffer].mapped, &features, sizeof(features));
		} else {
			HoodFeatureParams features{ hoodClothCount, hoodProxyCount, hoodTriangleCount, hoodMeshEdgeCount, hoodEmbeddingOffset,
				hoodFirstStep ? 1u : 0u, 0, 0, hoodFirstStep ? 1.0f / 3.0f : 1.0f / 30.0f, hoodCollisionRadius, material0, material1, material2 };
			std::memcpy(hood.featureUniforms[currentBuffer].mapped, &features, sizeof(features));
		}
		// When XPBD is on it owns contacts, and it resolves them inside every iteration rather than
		// once at the end. Leaving integrate's own projection on as well would apply the half-plane
		// twice per step and would not match the configuration gate G0 measured in Python, where
		// `integrate()` does no projection at all.
		HoodIntegrateParams integrate{ hoodClothCount, hoodFirstStep ? 1u : 0u,
			(hoodCollisionProjection && !hoodXpbdEnabled) ? 1u : 0u, hoodDecoderMlpId };
		std::memcpy(hood.integrateUniforms[currentBuffer].mapped, &integrate, sizeof(integrate));
		HoodToyParams toy{ hoodClothCount, 1.0f / static_cast<float>(hoodFps), 8.0f, 30.0f, glm::vec4(0.0f, -9.8f, 0.0f, 0.0f) };
		std::memcpy(hood.toyUniforms[currentBuffer].mapped, &toy, sizeof(toy));
		const glm::vec3 root = hoodRootPositions[hoodRenderFrame];
		camera.setTranslation(glm::vec3(-root.x, 0.35f - root.y, -3.2f - root.z));
		glm::mat4 projection = camera.matrices.perspective;
		projection[1][1] *= -1.0f;
		HoodGraphicsUniform graphics{ projection, camera.matrices.view, glm::vec4(-2.0f, 4.0f, -2.0f, 1.0f), glm::vec4(root, 1.0f) };
		std::memcpy(hood.graphicsUniforms[currentBuffer].mapped, &graphics, sizeof(graphics));
	}

	void hoodComputeBarrier(VkCommandBuffer command)
	{
		VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
		vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
	}

	// Jacobi XPBD after the network's integrate pass. See hood_xpbd.comp for why one dispatch per
	// iteration is the whole point, and plans/gnn/gnn-xpbd-v2.md section 3.3 for what it buys.
	void hoodRecordXpbd(VkCommandBuffer command)
	{
		if (!hoodXpbdEnabled || hoodXpbdIterations <= 0) return;
		const uint32_t iterations = static_cast<uint32_t>(hoodXpbdIterations);

		// lambda accumulates within a step and must start at zero in every step, exactly as
		// real_scene/xpbd.py::project builds a fresh multiplier vector per call. This barrier also
		// has to cover hood_integrate.comp's write to clothPosition, which the first iteration
		// reads -- the fill is a transfer but the position the sweep starts from is a shader write.
		vkCmdFillBuffer(command, hoodBuffers.xpbdLambda.buffer, 0, VK_WHOLE_SIZE, 0);
		VkMemoryBarrier clearBarrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
		vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &clearBarrier, 0, nullptr, 0, nullptr);

		HoodXpbdPush push{};
		push.clothCount = hoodClothCount;
		push.constraintCount = hoodXpbdConstraintCount;
		push.slotWidth = hoodXpbdSlotWidth;
		push.flags = (hoodXpbdOneSided ? hoodXpbdOneSidedFlag : 0u)
			| (hoodXpbdCollision ? hoodXpbdCollisionFlag : 0u);
		// The reference pipeline's first step is a 1/3 s settle rather than a physical substep,
		// and alpha = compliance / dt^2 has to use the same value make_graph did.
		push.timestep = hoodFirstStep ? 1.0f / 3.0f : 1.0f / 30.0f;
		push.relaxation = hoodXpbdRelaxation;
		push.contactOffset = hoodXpbdContactOffset;
		push.stretchCompliance = hoodXpbdStretchCompliance;
		push.bendCompliance = hoodXpbdBendCompliance;

		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.xpbd);
		vkCmdPushConstants(command, hood.xpbdPipeline, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
		for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
			vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.xpbdPipeline, 0, 1,
				&hood.xpbdSets[iteration & 1u], 0, nullptr);
			vkCmdDispatch(command, (hoodClothCount + 127) / 128, 1, 1);
			hoodComputeBarrier(command);
		}

		// An odd iteration count leaves the result in the scratch buffer. k is 128 by default so
		// this never fires, but silently rounding the count would be worse than one transfer.
		if (iterations & 1u) {
			VkBufferCopy copy{ .size = hoodClothCount * sizeof(glm::vec4) };
			vkCmdCopyBuffer(command, hoodBuffers.xpbdScratch.buffer, hoodBuffers.clothPosition.buffer, 1, &copy);
			hoodTransferThenComputeBarrier(command);
		}

		// hood_integrate.comp sets clothPrevious to the *uncorrected* prediction on the settle step
		// (every later step gets `effective`, which XPBD does not touch). tools/gate_g0.py does the
		// same through `previous = corrected if step == 0`, so the correction has to replace it here
		// or step 1 builds its velocity from a position the solver already rejected.
		if (hoodFirstStep) {
			VkBufferCopy copy{ .size = hoodClothCount * sizeof(glm::vec4) };
			vkCmdCopyBuffer(command, hoodBuffers.clothPosition.buffer, hoodBuffers.clothPrevious.buffer, 1, &copy);
			hoodTransferThenComputeBarrier(command);
		}
	}

	void hoodTransferThenComputeBarrier(VkCommandBuffer command)
	{
		VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT };
		vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
	}

	void hoodRecord(VkCommandBuffer command)
	{
		vkCmdResetQueryPool(command, hood.queryPools[currentBuffer], 0, hoodTimestampCount);
		// These state buffers are shared by the in-flight command buffers. Declare
		// the dependency from the preceding frame's fills, compute writes and
		// vertex fetches before this frame starts overwriting/reading them. The
		// same-queue submission order alone is not a Vulkan memory dependency.
		VkMemoryBarrier frameBarrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
		vkCmdPipelineBarrier(command,
			VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &frameBarrier, 0, nullptr, 0, nullptr);
		vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, hood.queryPools[currentBuffer], 0);
		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.skin);
		vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.skinPipeline, 0, 1, &hood.skinSets[currentBuffer], 0, nullptr);
		const uint32_t skinCount = std::max({ hoodCharacterCount, hoodProxyCount, hoodClothCount });
		vkCmdDispatch(command, (skinCount + 127) / 128, 1, 1);
		hoodComputeBarrier(command);
		vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 1);
		if (hoodSimulateFrame) {
			if (hoodSolver == HoodToy2L) {
				vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.toyPipeline, 0, 1, &hood.toySets[currentBuffer], 0, nullptr);
				vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.toyLayer0);
				vkCmdDispatch(command, hoodClothCount, 1, 1);
				hoodComputeBarrier(command);
				vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 2);
				vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.toyLayer1);
				vkCmdDispatch(command, hoodClothCount, 1, 1);
				vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 3);
				for (uint32_t timestamp = 4; timestamp < hoodTimestampCount; ++timestamp)
					vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], timestamp);
			} else {
			vkCmdFillBuffer(command, hoodBuffers.activeProxy.buffer, 0, VK_WHOLE_SIZE, 0);
			vkCmdFillBuffer(command, hoodBuffers.worldReverseCount.buffer, 0, VK_WHOLE_SIZE, 0);
			VkMemoryBarrier clearBarrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
			vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &clearBarrier, 0, nullptr, 0, nullptr);
			// Pick the nearest body proxy per cloth vertex before the feature pass, which now
			// reads worldObstacle instead of scanning every proxy itself. Both passes sit
			// inside the same timestamp pair, so `features_world` still covers all of the
			// feature work and stays comparable against earlier runs.
			vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.worldNearest);
			vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.worldNearestPipeline, 0, 1, &hood.worldNearestSet, 0, nullptr);
			const struct { uint32_t clothCount, proxyCount; float collisionRadius; } nearestPush{ hoodClothCount, hoodProxyCount, hoodCollisionRadius };
			vkCmdPushConstants(command, hood.worldNearestPipeline, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(nearestPush), &nearestPush);
			vkCmdDispatch(command, hoodClothCount, 1, 1);
			hoodComputeBarrier(command);
			// Invert the map once here so every processor block reads a short per-proxy list
			// instead of rescanning all cloth vertices.
			vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.worldReverse);
			vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.worldReversePipeline, 0, 1, &hood.worldReverseSet, 0, nullptr);
			vkCmdPushConstants(command, hood.worldReversePipeline, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hoodClothCount), &hoodClothCount);
			vkCmdDispatch(command, hoodClothCount, 1, 1);
			hoodComputeBarrier(command);
			vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.features);
			vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.featuresPipeline, 0, 1, &hood.featureSets[currentBuffer], 0, nullptr);
			const uint32_t featureCount = hoodSolver == HoodPostCvpr
				? std::max({ hoodClothCount, hoodProxyCount, hoodMeshEdgeCount, hoodCoarseEdgeCounts[0], hoodCoarseEdgeCounts[1] })
				: std::max({ hoodClothCount, hoodProxyCount, hoodMeshEdgeCount });
			vkCmdDispatch(command, (featureCount + 127) / 128, 1, 1);
			hoodComputeBarrier(command);
			vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 2);

			vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.encode);
			vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.encodePipeline, 0, 1, &hood.encodeSet, 0, nullptr);
			if (hoodSolver == HoodPostCvpr) {
				const uint32_t encodePushes[6][4] = {
					{0,0,hoodClothCount + hoodProxyCount,24}, {1,1,hoodMeshEdgeCount,12},
					{3,2,hoodCoarseEdgeCounts[0],12}, {4,3,hoodCoarseEdgeCounts[1],12},
					{2,4,hoodClothCount,9}, {2,5,hoodClothCount,9}
				};
				for (uint32_t encoder = 0; encoder < 6; ++encoder) {
					const auto& push = encodePushes[encoder];
					vkCmdPushConstants(command, hood.encodePipeline, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
					vkCmdDispatch(command, push[2], 1, 1);
					hoodComputeBarrier(command);
					if (encoder == 0) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 3);
					else if (encoder == 3) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 4);
					else if (encoder == 4) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 5);
					else if (encoder == 5) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 6);
				}
			} else {
				const uint32_t encodePushes[4][4] = { {0,0,hoodClothCount + hoodProxyCount,20}, {1,1,hoodMeshEdgeCount,12}, {2,2,hoodClothCount,9}, {2,3,hoodClothCount,9} };
				for (uint32_t encoder = 0; encoder < 4; ++encoder) {
					const auto& push = encodePushes[encoder];
					vkCmdPushConstants(command, hood.encodePipeline, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
					vkCmdDispatch(command, push[2], 1, 1);
					hoodComputeBarrier(command);
					vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 3 + encoder);
				}
			}

			for (uint32_t block = 0; block < hoodActiveProcessorBlocks; ++block) {
				const uint32_t ping = block & 1u;
				vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.edge);
				vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.edgePipeline, 0, 1, &hood.edgeSets[ping], 0, nullptr);
				if (hoodSolver == HoodPostCvpr) {
					const uint32_t counts[5]{ hoodMeshEdgeCount, hoodCoarseEdgeCounts[0], hoodCoarseEdgeCounts[1], hoodClothCount, hoodClothCount };
					for (uint32_t kind = 0; kind < 5; ++kind) {
						const uint32_t mlpId = hoodPostEdgeMlpIds[block][kind < 3 ? kind : 3];
						const uint32_t push[5]{ mlpId, kind, counts[kind], hoodClothCount, hoodPostActiveLevels[block] };
						vkCmdPushConstants(command, hood.edgePipeline, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, push);
						vkCmdDispatch(command, counts[kind], 1, 1);
					}
				} else {
					const uint32_t edgePushes[3][4] = { {block,0,hoodMeshEdgeCount,hoodClothCount}, {block,1,hoodClothCount,hoodClothCount}, {block,2,hoodClothCount,hoodClothCount} };
					for (const auto& push : edgePushes) { vkCmdPushConstants(command, hood.edgePipeline, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push); vkCmdDispatch(command, push[2], 1, 1); }
				}
				hoodComputeBarrier(command);
				vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 7 + block * 2);
				vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.node);
				vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.nodePipeline, 0, 1, &hood.nodeSets[ping], 0, nullptr);
				if (hoodSolver == HoodPostCvpr) {
					const uint32_t nodePush[5]{ hoodPostNodeMlpIds[block], hoodClothCount + hoodProxyCount, hoodClothCount,
						hoodPostEdgeMasks[block], hoodPostActiveLevels[block] };
					vkCmdPushConstants(command, hood.nodePipeline, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, nodePush);
				} else {
					const uint32_t nodePush[4]{ block, hoodClothCount + hoodProxyCount, hoodClothCount, hoodMeshEdgeCount };
					vkCmdPushConstants(command, hood.nodePipeline, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, nodePush);
				}
				vkCmdDispatch(command, hoodClothCount + hoodProxyCount, 1, 1);
				hoodComputeBarrier(command);
				vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 8 + block * 2);
			}
			for (uint32_t block = hoodActiveProcessorBlocks; block < hoodProcessorBlocks; ++block) {
				vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 7 + block * 2);
				vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], 8 + block * 2);
			}
			vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.integrate);
			vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, hood.integratePipeline, 0, 1, &hood.integrateSets[currentBuffer], 0, nullptr);
			vkCmdDispatch(command, hoodClothCount, 1, 1);
			vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], hoodIntegrateTimestamp);
			hoodRecordXpbd(command);
			vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], hoodXpbdTimestamp);
			}
		} else {
			for (uint32_t timestamp = 2; timestamp < hoodTimestampCount; ++timestamp)
				vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, hood.queryPools[currentBuffer], timestamp);
		}
		hood.queryWritten[currentBuffer] = true;
		hood.querySimulated[currentBuffer] = hoodSimulateFrame;
		VkMemoryBarrier vertexBarrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT };
		vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &vertexBarrier, 0, nullptr, 0, nullptr);
	}

	void hoodCollectTiming()
	{
		if (!hood.queryWritten[currentBuffer]) return;
		std::array<uint64_t, hoodTimestampCount> values{};
		if (vkGetQueryPoolResults(device, hood.queryPools[currentBuffer], 0, hoodTimestampCount, sizeof(values), values.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
		// Paused/reset-only frames write a complete zero-duration query sequence so
		// pools remain valid, but the UI should retain the last simulated sample.
		if (!hood.querySimulated[currentBuffer]) return;
		const double scale = deviceProperties.limits.timestampPeriod / 1.0e6;
		hoodTiming = {};
		hoodTiming.skin = (values[1] - values[0]) * scale;
		hoodTiming.features = (values[2] - values[1]) * scale;
		for (uint32_t encoder = 0; encoder < 4; ++encoder) hoodTiming.encoders[encoder] = (values[3 + encoder] - values[2 + encoder]) * scale;
		for (uint32_t block = 0; block < hoodProcessorBlocks; ++block) {
			hoodTiming.edgeBlocks[block] = (values[7 + block * 2] - values[6 + block * 2]) * scale;
			hoodTiming.nodeBlocks[block] = (values[8 + block * 2] - values[7 + block * 2]) * scale;
		}
		hoodTiming.integrate = (values[hoodIntegrateTimestamp] - values[hoodIntegrateTimestamp - 1]) * scale;
		hoodTiming.xpbd = (values[hoodXpbdTimestamp] - values[hoodIntegrateTimestamp]) * scale;
		hoodTiming.total = (values[hoodXpbdTimestamp] - values[0]) * scale;
		if (hoodStaticBenchmarkMode && hood.querySimulated[currentBuffer] && hoodStaticBenchmarkSamples.size() < hoodStaticBenchmarkTarget) {
			if (hoodStaticBenchmarkDiscarded < hoodStaticBenchmarkWarmup) ++hoodStaticBenchmarkDiscarded;
			else hoodStaticBenchmarkSamples.push_back(hoodTiming);
		}
	}

	void hoodWriteStaticBenchmarkCsv()
	{
		if (hoodStaticBenchmarkSamples.size() < hoodStaticBenchmarkTarget) {
			std::cerr << "Static HOOD benchmark collected " << hoodStaticBenchmarkSamples.size() << " of "
				<< hoodStaticBenchmarkTarget << " timestamp samples after " << hoodStaticBenchmarkDiscarded << " warmup samples\n";
			vks::tools::exitFatal("Static real-cloth benchmark did not collect the requested number of samples", -1);
			return;
		}
		if (hoodStaticBenchmarkOutput.has_parent_path()) std::filesystem::create_directories(hoodStaticBenchmarkOutput.parent_path());
		std::ofstream stream(hoodStaticBenchmarkOutput);
		if (!stream) {
			vks::tools::exitFatal("Could not create the static real-cloth benchmark CSV", -1);
			return;
		}
		std::ostringstream driver;
		if (deviceProperties.vendorID == 0x10de) {
			driver << ((deviceProperties.driverVersion >> 22) & 0x3ff) << '.' << ((deviceProperties.driverVersion >> 14) & 0xff)
				<< '.' << ((deviceProperties.driverVersion >> 6) & 0xff) << '.' << (deviceProperties.driverVersion & 0x3f);
		} else {
			driver << VK_VERSION_MAJOR(deviceProperties.driverVersion) << '.' << VK_VERSION_MINOR(deviceProperties.driverVersion)
				<< '.' << VK_VERSION_PATCH(deviceProperties.driverVersion);
		}
		const std::string solverName = hoodSolver == HoodToy2L ? "toy2l" : (hoodSolver == HoodTinyStudent ? "tinyhood" + hoodStudentLabel()
			: (hoodSolver == HoodPostCvpr ? "postcvpr" : "fine15"));
		stream << "device,driver_version,driver_raw,motion,solver,cloth_nodes,directed_mesh_edges,proxy_vertices,samples,stage,mean_ms,min_ms,p95_ms,max_ms\n";
		auto writeStage = [&](const std::string& stage, const auto& select) {
			std::vector<double> values;
			values.reserve(hoodStaticBenchmarkSamples.size());
			for (const auto& sample : hoodStaticBenchmarkSamples) values.push_back(select(sample));
			const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
			const auto bounds = std::minmax_element(values.begin(), values.end());
			stream << '"' << deviceProperties.deviceName << '"' << ',' << driver.str() << ',' << deviceProperties.driverVersion << ','
				<< hoodMotion << ',' << solverName << ',' << hoodClothCount << ',' << hoodMeshEdgeCount << ',' << hoodProxyCount << ',' << values.size() << ','
				<< stage << ',' << std::fixed << std::setprecision(6) << mean << ',' << *bounds.first << ','
				<< percentile(values, 0.95) << ',' << *bounds.second << '\n';
		};
		writeStage("skin", [](const HoodTiming& value) { return value.skin; });
		if (hoodSolver == HoodToy2L) {
			writeStage("toy_layer0", [](const HoodTiming& value) { return value.features; });
			writeStage("toy_layer1_integrate", [](const HoodTiming& value) { return value.encoders[0]; });
		} else {
			writeStage("features_world", [](const HoodTiming& value) { return value.features; });
			const std::array<const char*, 4> encoderNames{ "encoder_node", "encoder_mesh", "encoder_world_direct", "encoder_world_inverse" };
			for (uint32_t encoder = 0; encoder < encoderNames.size(); ++encoder)
				writeStage(encoderNames[encoder], [encoder](const HoodTiming& value) { return value.encoders[encoder]; });
			for (uint32_t block = 0; block < hoodActiveProcessorBlocks; ++block) {
				std::ostringstream edgeName, nodeName;
				edgeName << "block_" << std::setfill('0') << std::setw(2) << block << "_edge";
				nodeName << "block_" << std::setfill('0') << std::setw(2) << block << "_node";
				writeStage(edgeName.str(), [block](const HoodTiming& value) { return value.edgeBlocks[block]; });
				writeStage(nodeName.str(), [block](const HoodTiming& value) { return value.nodeBlocks[block]; });
			}
			writeStage("encoder_total", [](const HoodTiming& value) { return value.encodeTotal(); });
			writeStage(hoodSolver == HoodTinyStudent ? "processor_" + std::to_string(hoodActiveProcessorBlocks) + "_total"
				: (hoodSolver == HoodPostCvpr ? "hierarchical_processor_15_total" : "processor_15_total"),
				[](const HoodTiming& value) { return value.processorTotal(); });
			writeStage("decoder_integrate", [](const HoodTiming& value) { return value.integrate; });
			// The number plans/gnn/gnn-xpbd-v2.md section 2.3 could only estimate by multiplying a
			// 2.8 us dispatch price by the iteration count. Emitted whenever the stage exists so a
			// run with XPBD off records a zero column rather than changing the schema.
			if (hoodXpbdEnabled) writeStage("xpbd_" + std::to_string(hoodXpbdIterations) + "_jacobi",
				[](const HoodTiming& value) { return value.xpbd; });
		}
		writeStage("total", [](const HoodTiming& value) { return value.total; });
		std::cout << "Wrote " << hoodStaticBenchmarkOutput << " with " << hoodStaticBenchmarkSamples.size()
			<< " static T-pose timestamp samples\n";
	}

	template <typename T>
	std::vector<T> hoodReadback(VkBuffer source, uint32_t count)
	{
		const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(T);
		vks::Buffer readback;
		VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &readback, bytes));
		VkCommandBuffer command = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .buffer = source, .offset = 0, .size = bytes };
		vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
		VkBufferCopy copy{ .size = bytes };
		vkCmdCopyBuffer(command, source, readback.buffer, 1, &copy);
		vulkanDevice->flushCommandBuffer(command, queue, true);
		VK_CHECK_RESULT(readback.map());
		std::vector<T> result(count);
		std::memcpy(result.data(), readback.mapped, static_cast<size_t>(bytes));
		readback.destroy();
		return result;
	}

	void hoodWriteStabilityJson()
	{
		if (hoodStabilityOutput.empty()) return;
		const auto positions = hoodReadback<glm::vec4>(hoodBuffers.clothPosition.buffer, hoodClothCount);
		const auto pinTargets = hoodReadback<glm::vec4>(hoodBuffers.pinTarget.buffer, hoodClothCount);
		const auto world = hoodReadback<uint32_t>(hoodBuffers.worldObstacle.buffer, hoodClothCount);
		uint32_t invalidVertices = 0, activeWorldEdges = 0, collapsedEdges = 0, stretchedEdges = 0;
		uint32_t degenerateTriangles = 0, flippedTriangles = 0;
		double maximumDisplacement = 0.0, maximumPinnedError = 0.0;
		glm::dvec3 minimum(std::numeric_limits<double>::max()), maximum(-std::numeric_limits<double>::max());
		std::vector<double> edgeRatios, triangleAreaRatios;
		edgeRatios.reserve(hoodMeshSendersCpu.size());
		triangleAreaRatios.reserve(hoodTrianglesCpu.size());
		auto finite3 = [](const glm::vec4& value) {
			return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
		};
		for (uint32_t vertex = 0; vertex < hoodClothCount; ++vertex) {
			if (world[vertex] != 0xffffffffu) ++activeWorldEdges;
			if (!finite3(positions[vertex])) { ++invalidVertices; continue; }
			const glm::dvec3 position(positions[vertex]);
			const glm::dvec3 rest(hoodRestPositionsCpu[vertex]);
			minimum = glm::min(minimum, position); maximum = glm::max(maximum, position);
			const double displacement = glm::length(position - rest);
			maximumDisplacement = std::max(maximumDisplacement, displacement);
			if (hoodPinMaskCpu[vertex] && finite3(pinTargets[vertex])) {
				maximumPinnedError = std::max(maximumPinnedError,
					glm::length(position - glm::dvec3(pinTargets[vertex])));
			}
		}
		if (invalidVertices == hoodClothCount) {
			minimum = glm::dvec3(0.0);
			maximum = glm::dvec3(0.0);
		}
		for (size_t edge = 0; edge < hoodMeshSendersCpu.size(); ++edge) {
			const uint32_t sender = hoodMeshSendersCpu[edge], receiver = hoodMeshReceiversCpu[edge];
			if (!finite3(positions[sender]) || !finite3(positions[receiver])) continue;
			const double restLength = glm::length(glm::dvec3(hoodRestPositionsCpu[sender]) - glm::dvec3(hoodRestPositionsCpu[receiver]));
			if (restLength <= 1.0e-12) continue;
			const double ratio = glm::length(glm::dvec3(positions[sender]) - glm::dvec3(positions[receiver])) / restLength;
			edgeRatios.push_back(ratio);
			if (ratio < 0.5) ++collapsedEdges;
			if (ratio > 1.5) ++stretchedEdges;
		}
		for (const auto& triangle : hoodTrianglesCpu) {
			const uint32_t ids[3] = { triangle.x, triangle.y, triangle.z };
			if (!finite3(positions[ids[0]]) || !finite3(positions[ids[1]]) || !finite3(positions[ids[2]])) continue;
			const glm::dvec3 restA(hoodRestPositionsCpu[ids[0]]), restB(hoodRestPositionsCpu[ids[1]]), restC(hoodRestPositionsCpu[ids[2]]);
			const glm::dvec3 nowA(positions[ids[0]]), nowB(positions[ids[1]]), nowC(positions[ids[2]]);
			const glm::dvec3 restNormal = glm::cross(restB - restA, restC - restA);
			const glm::dvec3 nowNormal = glm::cross(nowB - nowA, nowC - nowA);
			const double restArea2 = glm::length(restNormal);
			if (restArea2 <= 1.0e-15) continue;
			const double areaRatio = glm::length(nowNormal) / restArea2;
			triangleAreaRatios.push_back(areaRatio);
			if (areaRatio < 0.1) ++degenerateTriangles;
			if (glm::dot(restNormal, nowNormal) < 0.0) ++flippedTriangles;
		}
		const double edgeMean = edgeRatios.empty() ? 0.0
			: std::accumulate(edgeRatios.begin(), edgeRatios.end(), 0.0) / static_cast<double>(edgeRatios.size());
		const double areaMean = triangleAreaRatios.empty() ? 0.0
			: std::accumulate(triangleAreaRatios.begin(), triangleAreaRatios.end(), 0.0) / static_cast<double>(triangleAreaRatios.size());
		const double edgeP95 = edgeRatios.empty() ? 0.0 : percentile(edgeRatios, 0.95);
		const double edgeMaximum = edgeRatios.empty() ? 0.0 : *std::max_element(edgeRatios.begin(), edgeRatios.end());
		const double areaMedian = triangleAreaRatios.empty() ? 0.0 : percentile(triangleAreaRatios, 0.5);
		const double collapsedFraction = hoodMeshSendersCpu.empty() ? 0.0 : collapsedEdges / static_cast<double>(hoodMeshSendersCpu.size());
		const double stretchedFraction = hoodMeshSendersCpu.empty() ? 0.0 : stretchedEdges / static_cast<double>(hoodMeshSendersCpu.size());
		const double degenerateFraction = hoodTrianglesCpu.empty() ? 0.0 : degenerateTriangles / static_cast<double>(hoodTrianglesCpu.size());
		const double flippedFraction = hoodTrianglesCpu.empty() ? 0.0 : flippedTriangles / static_cast<double>(hoodTrianglesCpu.size());
		const bool structurePreserved = invalidVertices == 0 && maximumPinnedError <= 1.0e-5 && edgeP95 <= 2.0
			&& collapsedFraction <= 0.05 && degenerateFraction <= 0.05;
		if (hoodStabilityOutput.has_parent_path()) std::filesystem::create_directories(hoodStabilityOutput.parent_path());
		std::ofstream output(hoodStabilityOutput);
		if (!output) throw std::runtime_error("Could not create HOOD stability JSON");
		output << std::setprecision(10)
			<< "{\n  \"scene\": \"" << (hoodGridScene ? "hood_grid64" : "ch10032") << "\","
			<< "\n  \"solver\": \"" << (hoodSolver == HoodToy2L ? "toy2l" : (hoodSolver == HoodTinyStudent ? "tinyhood" + hoodStudentLabel() : (hoodSolver == HoodPostCvpr ? "postcvpr" : "fine15"))) << "\","
			<< "\n  \"xpbd\": " << (hoodXpbdEnabled ? "true" : "false")
			<< ",\n  \"xpbd_iterations\": " << (hoodXpbdEnabled ? hoodXpbdIterations : 0)
			<< ",\n  \"collision_projection\": " << ((hoodCollisionProjection && !hoodXpbdEnabled) ? "true" : "false") << ','
			<< "\n  \"completed_steps\": " << hoodCompletedSteps << ",\n  \"structure_preserved\": " << (structurePreserved ? "true" : "false") << ','
			<< "\n  \"invalid_vertices\": " << invalidVertices << ",\n  \"active_world_edges\": " << activeWorldEdges << ','
			<< "\n  \"maximum_displacement_m\": " << maximumDisplacement << ",\n  \"maximum_pinned_error_m\": " << maximumPinnedError << ','
			<< "\n  \"edge_length_ratio\": {\"mean\": " << edgeMean << ", \"p95\": " << edgeP95 << ", \"max\": " << edgeMaximum
			<< ", \"collapsed_fraction_lt_0_5\": " << collapsedFraction << ", \"stretched_fraction_gt_1_5\": " << stretchedFraction << "},"
			<< "\n  \"triangle_area_ratio\": {\"mean\": " << areaMean << ", \"median\": " << areaMedian
			<< ", \"degenerate_fraction_lt_0_1\": " << degenerateFraction << ", \"flipped_fraction\": " << flippedFraction << "},"
			<< "\n  \"bounds_min_m\": [" << minimum.x << ", " << minimum.y << ", " << minimum.z << "],"
			<< "\n  \"bounds_max_m\": [" << maximum.x << ", " << maximum.y << ", " << maximum.z << "]\n}\n";
		std::cout << "Wrote " << hoodStabilityOutput << " after " << hoodCompletedSteps << " steps; structure "
			<< (structurePreserved ? "preserved" : "failed") << ", edge p95=" << edgeP95 << ", degenerate triangles=" << degenerateFraction << "\n";
	}

	void hoodVerifyCurrentStep()
	{
		if (!hoodVerifyMode || !hoodSimulateFrame || hoodVerifyStep >= hoodGoldenSteps) return;
		const auto positions = hoodReadback<glm::vec4>(hoodBuffers.clothPosition.buffer, hoodClothCount);
		const auto acceleration = hoodVerifyStep == 0 ? hoodReadback<glm::vec4>(hoodBuffers.acceleration.buffer, hoodClothCount) : std::vector<glm::vec4>{};
		double stepMaximum = 0.0, stepSum = 0.0;
		for (uint32_t vertex = 0; vertex < hoodClothCount; ++vertex) {
			const auto& expected = hoodGoldenPositions[static_cast<size_t>(hoodVerifyStep) * hoodClothCount + vertex];
			for (uint32_t component = 0; component < 3; ++component) {
				const double difference = std::abs(static_cast<double>(positions[vertex][component]) - (&expected.x)[component]);
				stepMaximum = std::max(stepMaximum, difference); stepSum += difference;
			}
		}
		hoodVerifyMaximum = std::max(hoodVerifyMaximum, stepMaximum);
		hoodVerifyMeanSum += stepSum;
		hoodVerifyValueCount += static_cast<uint64_t>(hoodClothCount) * 3;
		hoodVerifyStepMaximums.push_back(stepMaximum);
		hoodVerifyStepMeans.push_back(stepSum / (hoodClothCount * 3));
		if (hoodVerifyStep == 0) {
			double accelerationSum = 0.0;
			for (uint32_t vertex = 0; vertex < hoodClothCount; ++vertex) for (uint32_t component = 0; component < 3; ++component) {
				const double difference = std::abs(static_cast<double>(acceleration[vertex][component]) - (&hoodGoldenAcceleration[vertex].x)[component]);
				hoodVerifyAccelerationMaximum = std::max(hoodVerifyAccelerationMaximum, difference);
				accelerationSum += difference;
			}
			hoodVerifyAccelerationMean = accelerationSum / (hoodClothCount * 3);
			const auto world = hoodReadback<uint32_t>(hoodBuffers.worldObstacle.buffer, hoodClothCount);
			for (uint32_t vertex = 0; vertex < hoodClothCount; ++vertex) if (world[vertex] != hoodGoldenWorld[vertex]) ++hoodVerifyWorldMismatches;
			auto dumpFloats = [&](const char* label, VkBuffer buffer, uint32_t count) {
				const auto values = hoodReadback<float>(buffer, count);
				const auto path = hoodVerifyOutput.parent_path() / (std::string("hood_debug_") + label + ".bin");
				std::ofstream stream(path, std::ios::binary);
				stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
			};
			dumpFloats("node_features", hoodBuffers.nodeFeatures.buffer, (hoodClothCount + hoodProxyCount) * (hoodSolver == HoodPostCvpr ? 24u : 20u));
			dumpFloats("mesh_features", hoodBuffers.meshFeatures.buffer, hoodMeshEdgeCount * 12);
			dumpFloats("world_direct_features", hoodBuffers.worldDirectFeatures.buffer, hoodClothCount * 9);
			dumpFloats("world_inverse_features", hoodBuffers.worldInverseFeatures.buffer, hoodClothCount * 9);
			dumpFloats("node_latent", hoodBuffers.nodeLatent[hoodActiveProcessorBlocks & 1u].buffer, (hoodClothCount + hoodProxyCount) * hoodLatentSize);
			std::cout << (hoodSolver == HoodTinyStudent ? "TinyHOOD" : (hoodSolver == HoodPostCvpr ? "PostCVPR" : "Fine15")) << " Vulkan step 1: position max=" << stepMaximum << " mean=" << stepSum / (hoodClothCount * 3)
				<< " acceleration max=" << hoodVerifyAccelerationMaximum << " world mismatches=" << hoodVerifyWorldMismatches << "\n";
		}
		++hoodVerifyStep;
		if (hoodVerifyStep == hoodGoldenSteps && !hoodVerifyWritten) {
			const bool passed = hoodVerifyStepMaximums[0] <= 2.0e-4 && hoodVerifyStepMeans[0] <= 2.0e-5 && hoodVerifyMaximum <= 2.0e-3;
			if (hoodVerifyOutput.has_parent_path()) std::filesystem::create_directories(hoodVerifyOutput.parent_path());
			// The stream lives in its own scope so it is flushed and closed before the failure
			// throw below. Nothing catches that exception, and MSVC does not unwind the stack
			// for an unhandled exception, so a stream still in scope here would never flush --
			// leaving a zero-byte result file exactly when the numbers are needed to diagnose
			// the failure.
			{
				std::ofstream output(hoodVerifyOutput);
				output << "{\n  \"passed\": " << (passed ? "true" : "false") << ",\n  \"steps\": " << hoodVerifyStep
					<< ",\n  \"max_abs_error\": " << std::setprecision(10) << hoodVerifyMaximum
					<< ",\n  \"mean_abs_error\": " << hoodVerifyMeanSum / hoodVerifyValueCount
					<< ",\n  \"first_acceleration_max_abs_error\": " << hoodVerifyAccelerationMaximum
					<< ",\n  \"first_acceleration_mean_abs_error\": " << hoodVerifyAccelerationMean
					<< ",\n  \"first_world_edge_mismatches\": " << hoodVerifyWorldMismatches << ",\n  \"per_step\": [\n";
				for (uint32_t step = 0; step < hoodVerifyStep; ++step) {
					output << "    {\"step\": " << step + 1 << ", \"max_abs_error\": " << hoodVerifyStepMaximums[step]
						<< ", \"mean_abs_error\": " << hoodVerifyStepMeans[step] << "}" << (step + 1 == hoodVerifyStep ? "\n" : ",\n");
				}
				output << "  ]\n}\n";
			}
			std::cout << (hoodSolver == HoodTinyStudent ? "TinyHOOD" : (hoodSolver == HoodPostCvpr ? "PostCVPR" : "Fine15")) << " Vulkan " << hoodVerifyStep << " step max=" << hoodVerifyMaximum << " mean=" << hoodVerifyMeanSum / hoodVerifyValueCount << "\n";
			hoodVerifyWritten = true;
			if (!passed) throw std::runtime_error("HOOD Vulkan verification exceeded the one-step or ten-step error threshold");
		}
	}

	void hoodRender()
	{
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[currentBuffer], VK_TRUE, UINT64_MAX));
		hoodCollectTiming();
		VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentBuffer]));
		VulkanExampleBase::prepareFrame(false);
		hoodAdvance();
		hoodUpdateUniforms();
		VkCommandBuffer command = drawCmdBuffers[currentBuffer];
		auto begin = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(command, &begin));
		hoodRecord(command);
		VkClearValue clears[2]{}; clears[0].color = { {0.35f,0.60f,0.88f,1.0f} }; clears[1].depthStencil = {1.0f,0};
		auto renderBegin = vks::initializers::renderPassBeginInfo(); renderBegin.renderPass = renderPass; renderBegin.framebuffer = frameBuffers[currentImageIndex];
		renderBegin.renderArea.extent = { width,height }; renderBegin.clearValueCount = 2; renderBegin.pClearValues = clears;
		vkCmdBeginRenderPass(command, &renderBegin, VK_SUBPASS_CONTENTS_INLINE);
		const auto viewport = vks::initializers::viewport(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
		const auto scissor = vks::initializers::rect2D(width, height, 0, 0); vkCmdSetViewport(command, 0, 1, &viewport); vkCmdSetScissor(command, 0, 1, &scissor);
		vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, hood.graphicsPipeline, 0, 1, &hood.graphicsSets[currentBuffer], 0, nullptr);
		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, hood.sky); vkCmdDraw(command, 3, 1, 0, 0);
		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, hood.character);
		const VkBuffer characterBuffers[3]{ hoodBuffers.characterPosition.buffer, hoodBuffers.characterNormal.buffer, hoodBuffers.characterUv.buffer };
		const VkDeviceSize offsets[3]{}; vkCmdBindVertexBuffers(command, 0, 3, characterBuffers, offsets);
		vkCmdBindIndexBuffer(command, hoodBuffers.characterIndices.buffer, 0, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(command, hoodCharacterIndexCount, 1, 0, 0, 0);
		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, hood.cloth); const VkBuffer clothBuffer = hoodBuffers.clothPosition.buffer; const VkDeviceSize zero = 0;
		vkCmdBindVertexBuffers(command, 0, 1, &clothBuffer, &zero); vkCmdBindIndexBuffer(command, hoodBuffers.clothIndices.buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(command, hoodClothIndexCount, 1, 0, 0, 0);
		drawUI(command); vkCmdEndRenderPass(command); VK_CHECK_RESULT(vkEndCommandBuffer(command));
		VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		auto submit = vks::initializers::submitInfo(); submit.waitSemaphoreCount = 1; submit.pWaitSemaphores = &presentCompleteSemaphores[currentBuffer];
		submit.pWaitDstStageMask = &waitStage; submit.commandBufferCount = 1; submit.pCommandBuffers = &command; submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores = &renderCompleteSemaphores[currentImageIndex]; VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submit, waitFences[currentBuffer]));
		if (hoodVerifyMode) {
			VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[currentBuffer], VK_TRUE, UINT64_MAX));
			hoodVerifyCurrentStep();
		}
		VulkanExampleBase::submitFrame(true);
		if (hoodRequestReset) hoodRequestReset = false;
		if (hoodSimulateFrame) {
			hoodFrame = hoodRenderFrame;
			hoodFirstStep = false;
			++hoodCompletedSteps;
			if (hoodPauseAfterSteps != 0 && hoodCompletedSteps >= hoodPauseAfterSteps) hoodPaused = true;
		}
	}

	void hoodUI(vks::UIOverlay* overlay)
	{
		const std::string header = hoodGridScene
			? (hoodSolver == HoodTinyStudent ? "Grid64 + sphere + TinyHOOD " + hoodStudentLabel()
				: (hoodSolver == HoodPostCvpr ? "Grid64 + sphere + HOOD PostCVPR" : "Grid64 + sphere + HOOD Fine15"))
			: (hoodSolver == HoodToy2L ? "CH10032 + Toy GNN 10-16-3"
				: (hoodSolver == HoodTinyStudent ? "CH10032 + TinyHOOD " + hoodStudentLabel()
					: (hoodSolver == HoodPostCvpr ? "CH10032 + HOOD PostCVPR" : "CH10032 + HOOD Fine15")));
		if (!overlay->header(header.c_str())) return;
		overlay->checkBox("Paused", &hoodPaused);
		if (hoodSolver != HoodToy2L) overlay->checkBox("Body collision projection", &hoodCollisionProjection);
		if (hoodXpbdAvailable && hoodSolver != HoodToy2L) {
			overlay->checkBox("XPBD (Jacobi)", &hoodXpbdEnabled);
			if (hoodXpbdEnabled) {
				overlay->sliderInt("XPBD iterations", &hoodXpbdIterations, 0, 256);
				overlay->checkBox("XPBD one-sided", &hoodXpbdOneSided);
				overlay->checkBox("XPBD contacts", &hoodXpbdCollision);
				// Gate G0 measured 0..1e-6 as completely inert: alpha = compliance / dt^2 is then
				// seven orders below the inverse-mass sum. The usable range starts near 1e-2.
				overlay->sliderFloat("XPBD stretch compliance", &hoodXpbdStretchCompliance, 0.0f, 0.1f);
				overlay->sliderFloat("XPBD bend compliance", &hoodXpbdBendCompliance, 0.0f, 0.1f);
				overlay->text("%u constraints (%u stretch), %u slots/vertex",
					hoodXpbdConstraintCount, hoodXpbdStretchCount, hoodXpbdSlotWidth);
			}
		}
		if (overlay->button("Reset")) hoodRequestReset = true;
		overlay->text(hoodFrameCount == 1 ? "Pose: static %s" : "Native animation: %s", hoodMotion.c_str());
		if (hoodFrameCount > 1) overlay->text("Time: %.2f / %.2f s (%u/%u)", hoodFrame / float(hoodFps), (hoodFrameCount - 1) / float(hoodFps), hoodFrame, hoodFrameCount - 1);
		overlay->text("Core bones: %u", hoodBoneCount);
		overlay->text("Cloth: %u nodes, %u mesh edges", hoodClothCount, hoodMeshEdgeCount);
		if (hoodSolver == HoodPostCvpr) overlay->text("Hierarchy: c0 %u, c1 %u directed edges", hoodCoarseEdgeCounts[0], hoodCoarseEdgeCounts[1]);
		overlay->text("%s: %u vertices", hoodGridScene ? "Sphere proxy" : "Collision proxy", hoodProxyCount);
		overlay->text("Simulation steps: %u", hoodCompletedSteps);
		if (hoodGridScene) overlay->text("Constraints: top edge pins only");
		if (!hoodXpbdAvailable) overlay->text("XPBD: no .vxpbd asset baked");
		if (hoodSolver == HoodToy2L) {
			overlay->text("Toy model: 10 -> 16 -> 3, no XPBD/collision");
			overlay->text("GPU skin %.3f, layer 0 %.3f ms", hoodTiming.skin, hoodTiming.features);
			overlay->text("Layer 1 + integrate %.3f, total %.3f ms", hoodTiming.encoders[0], hoodTiming.total);
		} else {
			overlay->text("GPU skin %.3f, features/world %.3f ms", hoodTiming.skin, hoodTiming.features);
			overlay->text("GPU encoders %.3f, %u blocks %.3f ms", hoodTiming.encodeTotal(), hoodActiveProcessorBlocks, hoodTiming.processorTotal());
			overlay->text("Block 0 edge/node %.3f / %.3f ms", hoodTiming.edgeBlocks[0], hoodTiming.nodeBlocks[0]);
			overlay->text("GPU integrate %.3f, total %.3f ms", hoodTiming.integrate, hoodTiming.total);
			if (hoodXpbdEnabled && hoodXpbdIterations > 0)
				overlay->text("GPU XPBD %.3f ms (%d dispatches, %.4f ms each)", hoodTiming.xpbd,
					hoodXpbdIterations, hoodTiming.xpbd / static_cast<double>(hoodXpbdIterations));
		}
		overlay->text("R: reset, P: pause");
	}

	void hoodDestroy()
	{
		#define HOOD_DESTROY_BUFFER(name) hoodBuffers.name.destroy()
		HOOD_DESTROY_BUFFER(skinMatrices); HOOD_DESTROY_BUFFER(characterRestPosition); HOOD_DESTROY_BUFFER(characterRestNormal);
		HOOD_DESTROY_BUFFER(characterBoneIndices); HOOD_DESTROY_BUFFER(characterBoneWeights); HOOD_DESTROY_BUFFER(characterPosition);
		HOOD_DESTROY_BUFFER(characterNormal); HOOD_DESTROY_BUFFER(characterUv); HOOD_DESTROY_BUFFER(characterIndices);
		HOOD_DESTROY_BUFFER(proxyRestPosition); HOOD_DESTROY_BUFFER(proxyRestNormal); HOOD_DESTROY_BUFFER(proxyBoneIndices); HOOD_DESTROY_BUFFER(proxyBoneWeights);
		HOOD_DESTROY_BUFFER(proxyPosition); HOOD_DESTROY_BUFFER(proxyNormal); HOOD_DESTROY_BUFFER(proxyTarget);
		HOOD_DESTROY_BUFFER(clothRestPosition); HOOD_DESTROY_BUFFER(clothBoneIndices); HOOD_DESTROY_BUFFER(clothBoneWeights); HOOD_DESTROY_BUFFER(pinTarget);
		HOOD_DESTROY_BUFFER(pinMask); HOOD_DESTROY_BUFFER(mass); HOOD_DESTROY_BUFFER(clothPosition); HOOD_DESTROY_BUFFER(clothPrevious);
		HOOD_DESTROY_BUFFER(effectivePosition); HOOD_DESTROY_BUFFER(acceleration); HOOD_DESTROY_BUFFER(clothTriangles); HOOD_DESTROY_BUFFER(clothIndices);
		HOOD_DESTROY_BUFFER(clothTriangleOffsets); HOOD_DESTROY_BUFFER(clothTriangleIndices);
		HOOD_DESTROY_BUFFER(worldReverseCloth); HOOD_DESTROY_BUFFER(worldReverseBegin); HOOD_DESTROY_BUFFER(worldReverseCount);
		HOOD_DESTROY_BUFFER(meshSenders); HOOD_DESTROY_BUFFER(meshReceivers); HOOD_DESTROY_BUFFER(csrOffsets); HOOD_DESTROY_BUFFER(worldObstacle); HOOD_DESTROY_BUFFER(activeProxy); HOOD_DESTROY_BUFFER(nodeFeatures);
		HOOD_DESTROY_BUFFER(meshFeatures); HOOD_DESTROY_BUFFER(worldDirectFeatures); HOOD_DESTROY_BUFFER(worldInverseFeatures);
		if (hoodBuffers.vertexLevel.buffer != VK_NULL_HANDLE) hoodBuffers.vertexLevel.destroy();
		for (uint32_t level = 0; level < 2; ++level) {
			if (hoodBuffers.coarseSenders[level].buffer != VK_NULL_HANDLE) hoodBuffers.coarseSenders[level].destroy();
			if (hoodBuffers.coarseReceivers[level].buffer != VK_NULL_HANDLE) hoodBuffers.coarseReceivers[level].destroy();
			if (hoodBuffers.coarseOffsets[level].buffer != VK_NULL_HANDLE) hoodBuffers.coarseOffsets[level].destroy();
			if (hoodBuffers.coarseFeatures[level].buffer != VK_NULL_HANDLE) hoodBuffers.coarseFeatures[level].destroy();
		}
		HOOD_DESTROY_BUFFER(weights); HOOD_DESTROY_BUFFER(mlpTable); HOOD_DESTROY_BUFFER(normalizers); HOOD_DESTROY_BUFFER(toyWeights); HOOD_DESTROY_BUFFER(toyHidden);
		// Only created when a .vxpbd asset was present, so these have to be guarded.
		for (auto* buffer : { &hoodBuffers.xpbdPairs, &hoodBuffers.xpbdTargetLength, &hoodBuffers.xpbdWeightSum,
				&hoodBuffers.xpbdKind, &hoodBuffers.xpbdSlots, &hoodBuffers.xpbdSigns, &hoodBuffers.xpbdIncident,
				&hoodBuffers.xpbdInverseMass, &hoodBuffers.xpbdMinEdge, &hoodBuffers.xpbdLambda, &hoodBuffers.xpbdScratch })
			if (buffer->buffer != VK_NULL_HANDLE) buffer->destroy();
		for (uint32_t i=0;i<2;++i) {
			hoodBuffers.nodeLatent[i].destroy(); hoodBuffers.meshLatent[i].destroy(); hoodBuffers.worldDirectLatent[i].destroy(); hoodBuffers.worldInverseLatent[i].destroy();
			for (uint32_t level = 0; level < 2; ++level)
				if (hoodBuffers.coarseLatent[level][i].buffer != VK_NULL_HANDLE) hoodBuffers.coarseLatent[level][i].destroy();
		}
		#undef HOOD_DESTROY_BUFFER
		for (uint32_t i=0;i<maxConcurrentFrames;++i) { hood.skinUniforms[i].destroy(); hood.featureUniforms[i].destroy(); hood.integrateUniforms[i].destroy(); hood.toyUniforms[i].destroy(); hood.graphicsUniforms[i].destroy(); vkDestroyQueryPool(device,hood.queryPools[i],nullptr); }
		for (auto pipeline : {hood.skin,hood.features,hood.encode,hood.edge,hood.node,hood.integrate,hood.toyLayer0,hood.toyLayer1,hood.worldNearest,hood.worldReverse,hood.sky,hood.character,hood.cloth}) vkDestroyPipeline(device,pipeline,nullptr);
		for (auto layout : {hood.skinPipeline,hood.featuresPipeline,hood.encodePipeline,hood.edgePipeline,hood.nodePipeline,hood.integratePipeline,hood.toyPipeline,hood.worldNearestPipeline,hood.worldReversePipeline,hood.graphicsPipeline}) vkDestroyPipelineLayout(device,layout,nullptr);
		for (auto layout : {hood.skinLayout,hood.featuresLayout,hood.encodeLayout,hood.edgeLayout,hood.nodeLayout,hood.integrateLayout,hood.toyLayout,hood.worldNearestLayout,hood.worldReverseLayout,hood.graphicsLayout}) vkDestroyDescriptorSetLayout(device,layout,nullptr);
		if (hood.xpbd != VK_NULL_HANDLE) vkDestroyPipeline(device, hood.xpbd, nullptr);
		if (hood.xpbdPipeline != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, hood.xpbdPipeline, nullptr);
		if (hood.xpbdLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, hood.xpbdLayout, nullptr);
		vkDestroyDescriptorPool(device,hood.pool,nullptr);
	}
