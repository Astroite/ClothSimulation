#pragma once

#include "real_scene_format.h"

#include <bit>
#include <cmath>

namespace vhood
{
constexpr uint32_t latentSize = 128;
constexpr uint32_t processorBlocks = 15;
constexpr uint32_t mlpCount = 49;
constexpr uint32_t postCvprMlpCount = 64;
constexpr uint32_t noTensor = 0xffffffffu;

struct MlpGpu {
	uint32_t w0{}, b0{}, w1{}, b1{}, w2{}, b2{}, layerNormWeight{ noTensor }, layerNormBias{ noTensor };
	uint32_t inputDimension{}, outputDimension{}, hasLayerNorm{}, reserved{};
};
static_assert(sizeof(MlpGpu) == 48);

struct Fine15GpuModel {
	std::vector<float> weights;
	std::vector<MlpGpu> mlps;
	std::vector<float> normalizers; // mean/std pairs: node17, mesh9, world9, output3
	uint32_t embeddingOffset{};
	uint32_t vertexLevelEmbeddingOffset{ noTensor };
	uint32_t nodeFeatureDimension{ 20 };
	uint32_t decoderMlpId{ 48 };
	uint32_t outputNormalizerMeanOffset{ 70 };
	bool hierarchical{};
	std::array<std::array<uint32_t, 4>, processorBlocks> postEdgeMlpIds{}; // mesh, coarse0, coarse1, world
	std::array<uint32_t, processorBlocks> postNodeMlpIds{};
	std::array<uint32_t, processorBlocks> postEdgeMasks{};
	std::array<uint32_t, processorBlocks> postActiveLevels{};
};

inline std::span<const float> tensorFloats(const TensorView& view)
{
	return { reinterpret_cast<const float*>(view.bytes.data()), view.count };
}

struct TinyArchitecture {
	uint32_t latent{};
	uint32_t blocks{};
};

// Read the student's shape out of the checkpoint instead of hard-coding it, so retraining at a
// different width or depth needs a new .vhood and nothing else. The node encoder's first
// weight is [latent, 20], and the processor blocks are numbered contiguously from zero.
inline TinyArchitecture inferTinyArchitecture(const TensorAsset& asset)
{
	const auto encoder = asset.tensors.find("model._learned_model.node_encoder.0.layers.0.weight");
	if (encoder == asset.tensors.end()) throw std::runtime_error("Student checkpoint has no node encoder weight to infer the latent width from");
	const auto& shape = encoder->second.shape;
	if (shape.size() != 2 || shape[1] != 20) throw std::runtime_error("Student node encoder does not take the expected 20 node features");
	TinyArchitecture architecture{ shape[0], 0 };
	while (asset.tensors.contains("model._learned_model.processor_steps." + std::to_string(architecture.blocks) + ".mesh_edge_processor.0.layers.0.weight"))
		++architecture.blocks;
	// One lane owns one latent channel, so the width is also the workgroup size and has to
	// match a compiled SPIR-V variant; see TINY_LATENT_VARIANTS in tools/compile_shaders.py.
	if (architecture.latent != 32 && architecture.latent != 64)
		throw std::runtime_error("Student latent width has no compiled shader variant (expected 32 or 64)");
	if (architecture.blocks == 0 || architecture.blocks > processorBlocks)
		throw std::runtime_error("Student processor block count is outside the range the Vulkan schedule supports");
	return architecture;
}

inline Fine15GpuModel buildGpuModelFor(const TensorAsset& asset, uint32_t latent, uint32_t blocks)
{
	// The teacher is fixed at 128x15; students are 32 or 64 wide with any depth the Vulkan
	// timestamp schedule can hold.
	const bool teacher = latent == 128 && blocks == processorBlocks;
	const bool student = (latent == 64 || latent == 32) && blocks >= 1 && blocks <= processorBlocks;
	if (!teacher && !student) throw std::runtime_error("Unsupported HOOD GPU architecture");
	Fine15GpuModel result;
	result.mlps.resize(3 + blocks * 3 + 1);
	auto append = [&](const TensorView& tensor) {
		const uint32_t offset = static_cast<uint32_t>(result.weights.size());
		const auto values = tensorFloats(tensor);
		result.weights.insert(result.weights.end(), values.begin(), values.end());
		return offset;
	};
	// PyTorch stores nn.Linear.weight as [out][in]. The shaders give each lane one output
	// channel and walk the inputs, so a row-major matrix makes adjacent lanes read addresses
	// `in` floats apart: a 32-lane warp then touches 32 distinct cache lines to consume 128
	// useful bytes, and a 128-lane workgroup's layer-0 footprint (128 x 384 x 4 B = 192 KB)
	// exceeds the 128 KB L1, so lines are evicted before they can be reused. Storing the
	// transpose turns the same access into `weights[w + input * outputs + lane]`, which is
	// fully coalesced. This is a pure relayout; the arithmetic and its results are unchanged.
	auto appendTransposed = [&](const TensorView& tensor, uint32_t outputs, uint32_t inputs) {
		const uint32_t offset = static_cast<uint32_t>(result.weights.size());
		const auto values = tensorFloats(tensor);
		if (values.size() != static_cast<size_t>(outputs) * inputs) throw std::runtime_error("HOOD weight matrix has an unexpected element count");
		result.weights.resize(result.weights.size() + values.size());
		float* const destination = result.weights.data() + offset;
		for (uint32_t output = 0; output < outputs; ++output)
			for (uint32_t input = 0; input < inputs; ++input)
				destination[static_cast<size_t>(input) * outputs + output] = values[static_cast<size_t>(output) * inputs + input];
		return offset;
	};
	auto addMlp = [&](uint32_t id, const std::string& prefix, uint32_t input, uint32_t output, bool layerNorm) {
		if (id >= result.mlps.size()) throw std::runtime_error("HOOD MLP id is out of range");
		const std::string network = asset.tensors.contains(prefix + ".0.layers.0.weight") ? prefix + ".0" : prefix;
		MlpGpu descriptor{};
		descriptor.inputDimension = input;
		descriptor.outputDimension = output;
		descriptor.w0 = appendTransposed(asset.require(network + ".layers.0.weight", { latent, input }), latent, input);
		descriptor.b0 = append(asset.require(network + ".layers.0.bias", { latent }));
		descriptor.w1 = appendTransposed(asset.require(network + ".layers.2.weight", { latent, latent }), latent, latent);
		descriptor.b1 = append(asset.require(network + ".layers.2.bias", { latent }));
		descriptor.w2 = appendTransposed(asset.require(network + ".layers.4.weight", { output, latent }), output, latent);
		descriptor.b2 = append(asset.require(network + ".layers.4.bias", { output }));
		if (layerNorm) {
			descriptor.layerNormWeight = append(asset.require(prefix + ".1.weight", { output }));
			descriptor.layerNormBias = append(asset.require(prefix + ".1.bias", { output }));
			descriptor.hasLayerNorm = 1;
		}
		result.mlps[id] = descriptor;
	};
	addMlp(0, "model._learned_model.node_encoder", 20, latent, true);
	addMlp(1, "model._learned_model.edgeset_encoders.mesh", 12, latent, true);
	addMlp(2, "model._learned_model.edgeset_encoders.world", 9, latent, true);
	for (uint32_t block = 0; block < blocks; ++block) {
		const std::string base = "model._learned_model.processor_steps." + std::to_string(block);
		addMlp(3 + block * 3, base + ".mesh_edge_processor", latent * 3, latent, true);
		addMlp(4 + block * 3, base + ".world_edge_processor", latent * 3, latent, true);
		addMlp(5 + block * 3, base + ".node_processor", latent * 3, latent, true);
	}
	addMlp(3 + blocks * 3, "model._learned_model.decoder", latent, 3, false);
	result.embeddingOffset = append(asset.require("model.nodetype_embedding.weight", { 9, 9 }));

	for (const auto& [label, count] : std::array<std::pair<const char*, uint32_t>, 4>{ {
		{ "node", 17 }, { "mesh_edge", 9 }, { "world_edge", 9 }, { "output", 3 }
	} }) {
		const std::string prefix = std::string("model._") + label + "_normalizer";
		const float accumulationCount = tensorFloats(asset.require(prefix + "._acc_count", { 1 }))[0];
		if (!(accumulationCount >= 1.0f) || !std::isfinite(accumulationCount)) throw std::runtime_error("Fine15 normalizer has invalid count");
		const auto sum = tensorFloats(asset.require(prefix + "._acc_sum", { 1, count }));
		const auto squared = tensorFloats(asset.require(prefix + "._acc_sum_squared", { 1, count }));
		const size_t meanStart = result.normalizers.size();
		for (uint32_t i = 0; i < count; ++i) result.normalizers.push_back(sum[i] / accumulationCount);
		for (uint32_t i = 0; i < count; ++i) {
			const float mean = result.normalizers[meanStart + i];
			const float variance = std::max(0.0f, squared[i] / accumulationCount - mean * mean);
			result.normalizers.push_back(std::max(1.0e-8f, std::sqrt(variance)));
		}
	}
	result.decoderMlpId = 3 + blocks * 3;
	return result;
}

inline Fine15GpuModel buildGpuModel(const TensorAsset& asset) { return buildGpuModelFor(asset, 128, 15); }
inline Fine15GpuModel buildTinyGpuModel(const TensorAsset& asset)
{
	const auto architecture = inferTinyArchitecture(asset);
	return buildGpuModelFor(asset, architecture.latent, architecture.blocks);
}

inline Fine15GpuModel buildPostCvprGpuModel(const TensorAsset& asset)
{
	Fine15GpuModel result;
	result.mlps.resize(postCvprMlpCount);
	result.hierarchical = true;
	result.nodeFeatureDimension = 24;
	result.outputNormalizerMeanOffset = 78;
	for (auto& ids : result.postEdgeMlpIds) ids.fill(noTensor);
	auto append = [&](const TensorView& tensor) {
		const uint32_t offset = static_cast<uint32_t>(result.weights.size());
		const auto values = tensorFloats(tensor);
		result.weights.insert(result.weights.end(), values.begin(), values.end());
		return offset;
	};
	auto appendTransposed = [&](const TensorView& tensor, uint32_t outputs, uint32_t inputs) {
		const uint32_t offset = static_cast<uint32_t>(result.weights.size());
		const auto values = tensorFloats(tensor);
		if (values.size() != static_cast<size_t>(outputs) * inputs) throw std::runtime_error("PostCVPR weight matrix has an unexpected element count");
		result.weights.resize(result.weights.size() + values.size());
		float* destination = result.weights.data() + offset;
		for (uint32_t output = 0; output < outputs; ++output)
			for (uint32_t input = 0; input < inputs; ++input)
				destination[static_cast<size_t>(input) * outputs + output] = values[static_cast<size_t>(output) * inputs + input];
		return offset;
	};
	auto addMlp = [&](uint32_t id, const std::string& prefix, uint32_t input, uint32_t output, bool layerNorm) {
		if (id >= result.mlps.size()) throw std::runtime_error("PostCVPR MLP id is out of range");
		const std::string network = asset.tensors.contains(prefix + ".0.layers.0.weight") ? prefix + ".0" : prefix;
		MlpGpu descriptor{};
		descriptor.inputDimension = input;
		descriptor.outputDimension = output;
		descriptor.w0 = appendTransposed(asset.require(network + ".layers.0.weight", { latentSize, input }), latentSize, input);
		descriptor.b0 = append(asset.require(network + ".layers.0.bias", { latentSize }));
		descriptor.w1 = appendTransposed(asset.require(network + ".layers.2.weight", { latentSize, latentSize }), latentSize, latentSize);
		descriptor.b1 = append(asset.require(network + ".layers.2.bias", { latentSize }));
		descriptor.w2 = appendTransposed(asset.require(network + ".layers.4.weight", { output, latentSize }), output, latentSize);
		descriptor.b2 = append(asset.require(network + ".layers.4.bias", { output }));
		if (layerNorm) {
			descriptor.layerNormWeight = append(asset.require(prefix + ".1.weight", { output }));
			descriptor.layerNormBias = append(asset.require(prefix + ".1.bias", { output }));
			descriptor.hasLayerNorm = 1;
		}
		result.mlps[id] = descriptor;
	};

	uint32_t next = 0;
	addMlp(next++, "model._learned_model.node_encoder", 24, latentSize, true);
	addMlp(next++, "model._learned_model.edgeset_encoders.mesh", 12, latentSize, true);
	addMlp(next++, "model._learned_model.edgeset_encoders.world", 9, latentSize, true);
	addMlp(next++, "model._learned_model.edgeset_encoders.coarse0", 12, latentSize, true);
	addMlp(next++, "model._learned_model.edgeset_encoders.coarse1", 12, latentSize, true);
	addMlp(next++, "model._learned_model.edgeset_encoders.coarse2", 12, latentSize, true);

	const std::array<uint32_t, processorBlocks> levels{ 0,0,0,1,1,1,2,2,2,3,3,3,4,4,4 };
	const std::array<uint32_t, processorBlocks> activeLevels{ 0,0,0,1,1,1,2,2,2,1,1,1,0,0,0 };
	const std::array<uint32_t, processorBlocks> masks{
		0x3,0x3,0x3, 0x6,0x6,0x6, 0x4,0x4,0x4, 0x6,0x6,0x6, 0x3,0x3,0x3
	}; // bit0 mesh, bit1 coarse0, bit2 coarse1; world is always present
	const std::array<const char*, 3> edgeNames{ "mesh_edge", "coarse_edge0", "coarse_edge1" };
	for (uint32_t block = 0; block < processorBlocks; ++block) {
		const uint32_t step = block % 3;
		const std::string base = "model._learned_model.levels." + std::to_string(levels[block]) + "." + std::to_string(step);
		result.postEdgeMasks[block] = masks[block];
		result.postActiveLevels[block] = activeLevels[block];
		for (uint32_t kind = 0; kind < edgeNames.size(); ++kind) {
			if ((masks[block] & (1u << kind)) == 0) continue;
			result.postEdgeMlpIds[block][kind] = next;
			addMlp(next++, base + ".edge_processor_dict." + edgeNames[kind], latentSize * 3, latentSize, true);
		}
		result.postEdgeMlpIds[block][3] = next;
		addMlp(next++, base + ".edge_processor_dict.world_edge", latentSize * 3, latentSize, true);
		const uint32_t edgeKeyCount = 1 + std::popcount(masks[block]);
		result.postNodeMlpIds[block] = next;
		addMlp(next++, base + ".node_processor_dict.node", latentSize * (1 + edgeKeyCount), latentSize, true);
	}
	result.decoderMlpId = next;
	addMlp(next++, "model._learned_model.decoder", latentSize, 3, false);
	if (next != postCvprMlpCount) throw std::runtime_error("PostCVPR packed MLP count differs from the fixed Vulkan schedule");
	result.embeddingOffset = append(asset.require("model.nodetype_embedding.weight", { 9, 9 }));
	result.vertexLevelEmbeddingOffset = append(asset.require("model.vertexlevel_embedding.weight", { 4, 4 }));

	for (const auto& [label, count] : std::array<std::pair<const char*, uint32_t>, 4>{ {
		{ "node", 21 }, { "mesh_edge", 9 }, { "world_edge", 9 }, { "output", 3 }
	} }) {
		const std::string prefix = std::string("model._") + label + "_normalizer";
		const float accumulationCount = tensorFloats(asset.require(prefix + "._acc_count", { 1 }))[0];
		if (!(accumulationCount >= 1.0f) || !std::isfinite(accumulationCount)) throw std::runtime_error("PostCVPR normalizer has invalid count");
		const auto sum = tensorFloats(asset.require(prefix + "._acc_sum", { 1, count }));
		const auto squared = tensorFloats(asset.require(prefix + "._acc_sum_squared", { 1, count }));
		const size_t meanStart = result.normalizers.size();
		for (uint32_t i = 0; i < count; ++i) result.normalizers.push_back(sum[i] / accumulationCount);
		for (uint32_t i = 0; i < count; ++i) {
			const float mean = result.normalizers[meanStart + i];
			const float variance = std::max(0.0f, squared[i] / accumulationCount - mean * mean);
			result.normalizers.push_back(std::max(1.0e-8f, std::sqrt(variance)));
		}
	}
	return result;
}
}
