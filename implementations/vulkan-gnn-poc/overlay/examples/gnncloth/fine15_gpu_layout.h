#pragma once

#include "real_scene_format.h"

#include <cmath>

namespace vhood
{
constexpr uint32_t latentSize = 128;
constexpr uint32_t processorBlocks = 15;
constexpr uint32_t mlpCount = 49;
constexpr uint32_t noTensor = 0xffffffffu;

struct MlpGpu {
	uint32_t w0{}, b0{}, w1{}, b1{}, w2{}, b2{}, layerNormWeight{ noTensor }, layerNormBias{ noTensor };
	uint32_t inputDimension{}, outputDimension{}, hasLayerNorm{}, reserved{};
};
static_assert(sizeof(MlpGpu) == 48);

struct Fine15GpuModel {
	std::vector<float> weights;
	std::array<MlpGpu, mlpCount> mlps{};
	std::vector<float> normalizers; // mean/std pairs: node17, mesh9, world9, output3
	uint32_t embeddingOffset{};
};

inline std::span<const float> tensorFloats(const TensorView& view)
{
	return { reinterpret_cast<const float*>(view.bytes.data()), view.count };
}

inline Fine15GpuModel buildGpuModel(const TensorAsset& asset)
{
	Fine15GpuModel result;
	auto append = [&](const TensorView& tensor) {
		const uint32_t offset = static_cast<uint32_t>(result.weights.size());
		const auto values = tensorFloats(tensor);
		result.weights.insert(result.weights.end(), values.begin(), values.end());
		return offset;
	};
	auto addMlp = [&](uint32_t id, const std::string& prefix, uint32_t input, uint32_t output, bool layerNorm) {
		if (id >= result.mlps.size()) throw std::runtime_error("Fine15 MLP id is out of range");
		const std::string network = asset.tensors.contains(prefix + ".0.layers.0.weight") ? prefix + ".0" : prefix;
		MlpGpu descriptor{};
		descriptor.inputDimension = input;
		descriptor.outputDimension = output;
		descriptor.w0 = append(asset.require(network + ".layers.0.weight", { 128, input }));
		descriptor.b0 = append(asset.require(network + ".layers.0.bias", { 128 }));
		descriptor.w1 = append(asset.require(network + ".layers.2.weight", { 128, 128 }));
		descriptor.b1 = append(asset.require(network + ".layers.2.bias", { 128 }));
		descriptor.w2 = append(asset.require(network + ".layers.4.weight", { output, 128 }));
		descriptor.b2 = append(asset.require(network + ".layers.4.bias", { output }));
		if (layerNorm) {
			descriptor.layerNormWeight = append(asset.require(prefix + ".1.weight", { output }));
			descriptor.layerNormBias = append(asset.require(prefix + ".1.bias", { output }));
			descriptor.hasLayerNorm = 1;
		}
		result.mlps[id] = descriptor;
	};
	addMlp(0, "model._learned_model.node_encoder", 20, 128, true);
	addMlp(1, "model._learned_model.edgeset_encoders.mesh", 12, 128, true);
	addMlp(2, "model._learned_model.edgeset_encoders.world", 9, 128, true);
	for (uint32_t block = 0; block < processorBlocks; ++block) {
		const std::string base = "model._learned_model.processor_steps." + std::to_string(block);
		addMlp(3 + block * 3, base + ".mesh_edge_processor", 384, 128, true);
		addMlp(4 + block * 3, base + ".world_edge_processor", 384, 128, true);
		addMlp(5 + block * 3, base + ".node_processor", 384, 128, true);
	}
	addMlp(48, "model._learned_model.decoder", 128, 3, false);
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
	return result;
}
}
