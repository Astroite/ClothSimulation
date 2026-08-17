#include "fine15_gpu_layout.h"

#include <filesystem>
#include <fstream>
#include <iostream>

static void expectFailure(const std::filesystem::path& path, auto loader)
{
	try { loader(path); }
	catch (const std::exception&) { return; }
	throw std::runtime_error("Corrupt runtime asset was accepted: " + path.string());
}

int main(int argc, char** argv)
{
	if (argc != 3) {
		std::cerr << "usage: real_scene_format_test <asset-root> <fine15.vhood>\n";
		return 2;
	}
	const std::filesystem::path root = argv[1];
	const auto character = vhood::loadSectioned(root / "ch10032.vchar", "VCHAR001", 1);
	const auto animation = vhood::loadSectioned(root / "ch10032_sprint.vanim", "VANIM001", 1);
	const auto cloth = vhood::loadSectioned(root / "ch10032_lower.vcloth2", "VCLTH002", 2);
	const auto model = vhood::loadTensorAsset(argv[2]);
	const auto gpuModel = vhood::buildGpuModel(model);
	character.require("render_pos", 12);
	animation.require("skin_matrices", 48);
	cloth.require("positions", 12, 1377);
	model.require("model.nodetype_embedding.weight", { 9, 9 });
	if (gpuModel.weights.empty() || gpuModel.normalizers.size() != 76 || gpuModel.mlps[48].outputDimension != 3)
		throw std::runtime_error("Fine15 GPU packing failed");

	auto corrupt = vgnn::readFile(root / "ch10032.vchar");
	corrupt.back() ^= 1;
	const auto corruptPath = std::filesystem::temp_directory_path() / "vchar_bad_checksum.bin";
	{
		std::ofstream stream(corruptPath, std::ios::binary);
		stream.write(reinterpret_cast<const char*>(corrupt.data()), static_cast<std::streamsize>(corrupt.size()));
	}
	expectFailure(corruptPath, [](const auto& path) { vhood::loadSectioned(path, "VCHAR001", 1); });
	std::filesystem::remove(corruptPath);
	std::cout << "real scene formats ok: " << character.require("render_pos").count << " character vertices, "
		<< cloth.require("positions").count << " cloth vertices, " << model.tensors.size() << " tensors, "
		<< gpuModel.weights.size() << " packed floats\n";
	return 0;
}
