#include "vgnn_format.h"

#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
	if (argc != 3) {
		std::cerr << "usage: vgnn_format_test <model.bin> <golden.bin>\n";
		return 2;
	}
	try {
		const auto model = vgnn::loadModel(argv[1]);
		const auto golden = vgnn::loadGolden(argv[2]);
		if (model.payload.size() != vgnn::payloadFloatCount || golden.gridSize != 32 || golden.vertexCount != 1024) {
			throw std::runtime_error("valid artifact decoded to unexpected dimensions");
		}

		const auto source = vgnn::readFile(argv[1]);
		auto assertRejected = [&](std::vector<uint8_t> bytes, const char* name) {
			const std::filesystem::path path = std::filesystem::temp_directory_path() / (std::string("vgnn_") + name + ".bin");
			std::ofstream stream(path, std::ios::binary);
			stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
			stream.close();
			try {
				(void)vgnn::loadModel(path);
			} catch (const std::exception&) {
				std::filesystem::remove(path);
				return;
			}
			std::filesystem::remove(path);
			throw std::runtime_error(std::string("loader accepted invalid ") + name);
		};

		auto badMagic = source;
		badMagic[0] = 'X';
		assertRejected(std::move(badMagic), "magic");
		auto badVersion = source;
		badVersion[4] = 99;
		assertRejected(std::move(badVersion), "version");
		auto badDimension = source;
		badDimension[16] = 11;
		assertRejected(std::move(badDimension), "dimension");
		auto badChecksum = source;
		badChecksum.back() ^= 1;
		assertRejected(std::move(badChecksum), "checksum");
		auto truncated = source;
		truncated.resize(truncated.size() - sizeof(float));
		assertRejected(std::move(truncated), "truncated");

		std::cout << "model_floats=" << model.payload.size() << "\n";
		std::cout << "golden_vertices=" << golden.vertexCount << " golden_edges=" << golden.edgeCount << "\n";
		std::cout << "negative_loader_tests=5/5\n";
		return 0;
	} catch (const std::exception& exception) {
		std::cerr << exception.what() << "\n";
		return 1;
	}
}
