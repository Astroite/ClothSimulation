#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vgnn
{
constexpr uint32_t version = 1;
constexpr uint32_t scalarFp32 = 1;
constexpr uint32_t inputDimension = 10;
constexpr uint32_t hiddenDimension = 16;
constexpr uint32_t outputDimension = 3;
constexpr uint32_t payloadFloatCount = 448;

#pragma pack(push, 1)
struct ModelHeader {
	char magic[4];
	uint32_t version;
	uint32_t headerSize;
	uint32_t scalarType;
	uint32_t inputDimension;
	uint32_t hiddenDimension;
	uint32_t outputDimension;
	uint32_t payloadFloatCount;
	uint32_t payloadCrc32;
	uint32_t reserved[3];
};

struct GoldenHeader {
	char magic[4];
	uint32_t version;
	uint32_t headerSize;
	uint32_t gridSize;
	uint32_t vertexCount;
	uint32_t edgeCount;
	uint32_t payloadBytes;
	uint32_t payloadCrc32;
	uint32_t reserved[2];
	float externalAcceleration[3];
};
#pragma pack(pop)

static_assert(sizeof(ModelHeader) == 48);
static_assert(sizeof(GoldenHeader) == 52);

struct Model {
	std::vector<float> payload;
};

struct GoldenCase {
	uint32_t gridSize{};
	uint32_t vertexCount{};
	uint32_t edgeCount{};
	std::array<float, 3> externalAcceleration{};
	std::vector<uint32_t> offsets;
	std::vector<uint32_t> neighbors;
	std::vector<float> restPositions;
	std::vector<float> positions;
	std::vector<float> velocities;
	std::vector<float> pinned;
	std::vector<float> expectedAccelerations;
};

inline std::vector<uint8_t> readFile(const std::filesystem::path& path)
{
	std::ifstream stream(path, std::ios::binary | std::ios::ate);
	if (!stream) {
		throw std::runtime_error("Could not open " + path.string());
	}
	const auto length = stream.tellg();
	if (length < 0) {
		throw std::runtime_error("Could not determine length of " + path.string());
	}
	std::vector<uint8_t> bytes(static_cast<size_t>(length));
	stream.seekg(0);
	if (!bytes.empty() && !stream.read(reinterpret_cast<char*>(bytes.data()), length)) {
		throw std::runtime_error("Could not read " + path.string());
	}
	return bytes;
}

inline uint32_t crc32(const uint8_t* data, size_t size)
{
	uint32_t crc = 0xFFFFFFFFu;
	for (size_t i = 0; i < size; ++i) {
		crc ^= data[i];
		for (uint32_t bit = 0; bit < 8; ++bit) {
			const uint32_t mask = 0u - (crc & 1u);
			crc = (crc >> 1u) ^ (0xEDB88320u & mask);
		}
	}
	return ~crc;
}

template <typename T>
inline std::vector<T> consume(const uint8_t*& cursor, const uint8_t* end, size_t count)
{
	const size_t byteCount = sizeof(T) * count;
	if (static_cast<size_t>(end - cursor) < byteCount) {
		throw std::runtime_error("Binary payload is truncated");
	}
	std::vector<T> result(count);
	if (byteCount > 0) {
		std::memcpy(result.data(), cursor, byteCount);
	}
	cursor += byteCount;
	return result;
}

inline Model loadModel(const std::filesystem::path& path)
{
	const std::vector<uint8_t> bytes = readFile(path);
	if (bytes.size() < sizeof(ModelHeader)) {
		throw std::runtime_error("VGNN model is shorter than its header");
	}
	ModelHeader header{};
	std::memcpy(&header, bytes.data(), sizeof(header));
	if (std::memcmp(header.magic, "VGNN", 4) != 0) {
		throw std::runtime_error("Invalid VGNN magic");
	}
	if (header.version != version || header.headerSize != sizeof(ModelHeader) || header.scalarType != scalarFp32) {
		throw std::runtime_error("Unsupported VGNN version, header, or scalar type");
	}
	if (header.inputDimension != inputDimension || header.hiddenDimension != hiddenDimension || header.outputDimension != outputDimension) {
		throw std::runtime_error("VGNN dimensions do not match the fixed shader runtime");
	}
	if (header.payloadFloatCount != payloadFloatCount || header.reserved[0] || header.reserved[1] || header.reserved[2]) {
		throw std::runtime_error("Invalid VGNN payload declaration");
	}
	const size_t expectedBytes = sizeof(ModelHeader) + static_cast<size_t>(header.payloadFloatCount) * sizeof(float);
	if (bytes.size() != expectedBytes) {
		throw std::runtime_error("VGNN payload length mismatch");
	}
	const uint8_t* payload = bytes.data() + sizeof(ModelHeader);
	if (crc32(payload, bytes.size() - sizeof(ModelHeader)) != header.payloadCrc32) {
		throw std::runtime_error("VGNN payload checksum mismatch");
	}
	Model result;
	result.payload.resize(header.payloadFloatCount);
	std::memcpy(result.payload.data(), payload, header.payloadFloatCount * sizeof(float));
	return result;
}

inline GoldenCase loadGolden(const std::filesystem::path& path)
{
	const std::vector<uint8_t> bytes = readFile(path);
	if (bytes.size() < sizeof(GoldenHeader)) {
		throw std::runtime_error("VGLD case is shorter than its header");
	}
	GoldenHeader header{};
	std::memcpy(&header, bytes.data(), sizeof(header));
	if (std::memcmp(header.magic, "VGLD", 4) != 0 || header.version != version || header.headerSize != sizeof(GoldenHeader)) {
		throw std::runtime_error("Unsupported VGLD header");
	}
	if (header.vertexCount != header.gridSize * header.gridSize || header.reserved[0] || header.reserved[1]) {
		throw std::runtime_error("Invalid VGLD dimensions");
	}
	if (bytes.size() != sizeof(GoldenHeader) + header.payloadBytes) {
		throw std::runtime_error("VGLD payload length mismatch");
	}
	const uint8_t* cursor = bytes.data() + sizeof(GoldenHeader);
	const uint8_t* end = bytes.data() + bytes.size();
	if (crc32(cursor, header.payloadBytes) != header.payloadCrc32) {
		throw std::runtime_error("VGLD payload checksum mismatch");
	}

	GoldenCase result;
	result.gridSize = header.gridSize;
	result.vertexCount = header.vertexCount;
	result.edgeCount = header.edgeCount;
	std::memcpy(result.externalAcceleration.data(), header.externalAcceleration, sizeof(header.externalAcceleration));
	result.offsets = consume<uint32_t>(cursor, end, static_cast<size_t>(header.vertexCount) + 1);
	result.neighbors = consume<uint32_t>(cursor, end, header.edgeCount);
	result.restPositions = consume<float>(cursor, end, static_cast<size_t>(header.vertexCount) * 3);
	result.positions = consume<float>(cursor, end, static_cast<size_t>(header.vertexCount) * 3);
	result.velocities = consume<float>(cursor, end, static_cast<size_t>(header.vertexCount) * 3);
	result.pinned = consume<float>(cursor, end, header.vertexCount);
	result.expectedAccelerations = consume<float>(cursor, end, static_cast<size_t>(header.vertexCount) * 3);
	if (cursor != end) {
		throw std::runtime_error("VGLD payload contains trailing bytes");
	}
	return result;
}
}
