#pragma once

#include "vgnn_format.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace vhood
{
constexpr uint64_t alignment = 16;

inline uint32_t rotateRight(uint32_t value, uint32_t bits) { return (value >> bits) | (value << (32u - bits)); }

inline std::array<uint8_t, 32> sha256(std::span<const uint8_t> input)
{
	static constexpr uint32_t constants[64] = {
		0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
		0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
		0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
		0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
		0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
		0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
		0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
		0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
	};
	std::array<uint32_t, 8> state{ 0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u };
	const uint64_t bitLength = static_cast<uint64_t>(input.size()) * 8u;
	const size_t padded = ((input.size() + 9u + 63u) / 64u) * 64u;
	std::vector<uint8_t> message(padded, 0);
	std::memcpy(message.data(), input.data(), input.size());
	message[input.size()] = 0x80u;
	for (uint32_t i = 0; i < 8; ++i) message[padded - 1u - i] = static_cast<uint8_t>(bitLength >> (i * 8u));
	for (size_t chunk = 0; chunk < padded; chunk += 64) {
		uint32_t words[64]{};
		for (uint32_t i = 0; i < 16; ++i) {
			const size_t p = chunk + i * 4;
			words[i] = (static_cast<uint32_t>(message[p]) << 24u) | (static_cast<uint32_t>(message[p + 1]) << 16u)
				| (static_cast<uint32_t>(message[p + 2]) << 8u) | message[p + 3];
		}
		for (uint32_t i = 16; i < 64; ++i) {
			const uint32_t s0 = rotateRight(words[i - 15], 7) ^ rotateRight(words[i - 15], 18) ^ (words[i - 15] >> 3u);
			const uint32_t s1 = rotateRight(words[i - 2], 17) ^ rotateRight(words[i - 2], 19) ^ (words[i - 2] >> 10u);
			words[i] = words[i - 16] + s0 + words[i - 7] + s1;
		}
		auto [a,b,c,d,e,f,g,h] = state;
		for (uint32_t i = 0; i < 64; ++i) {
			const uint32_t s1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
			const uint32_t choose = (e & f) ^ (~e & g);
			const uint32_t temp1 = h + s1 + choose + constants[i] + words[i];
			const uint32_t s0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
			const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
			const uint32_t temp2 = s0 + majority;
			h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
		}
		state[0] += a; state[1] += b; state[2] += c; state[3] += d;
		state[4] += e; state[5] += f; state[6] += g; state[7] += h;
	}
	std::array<uint8_t, 32> result{};
	for (uint32_t i = 0; i < 8; ++i) for (uint32_t b = 0; b < 4; ++b) result[i * 4 + b] = static_cast<uint8_t>(state[i] >> (24u - b * 8u));
	return result;
}

#pragma pack(push, 1)
struct AssetHeader {
	char magic[8];
	uint32_t version;
	uint32_t count;
	uint64_t fileSize;
	uint64_t payloadOffset;
	uint8_t payloadSha256[32];
	uint8_t sourceSha256[32];
};
struct SectionEntry {
	char name[16];
	uint64_t offset;
	uint32_t count;
	uint32_t stride;
};
struct TensorEntry {
	char name[160];
	uint64_t offset;
	uint32_t count;
	uint32_t rank;
	uint32_t dimensions[8];
};
#pragma pack(pop)
static_assert(sizeof(AssetHeader) == 96);
static_assert(sizeof(SectionEntry) == 32);
static_assert(sizeof(TensorEntry) == 208);

struct ByteView {
	uint64_t offset{};
	uint32_t count{};
	uint32_t stride{};
	std::span<const uint8_t> bytes;
	template <typename T> std::span<const T> as(uint32_t expectedStride = sizeof(T)) const
	{
		if (stride != expectedStride || bytes.size() != static_cast<size_t>(count) * stride || offset % alignof(T)) {
			throw std::runtime_error("Runtime asset section has an incompatible element type");
		}
		return { reinterpret_cast<const T*>(bytes.data()), count };
	}
};

struct SectionedAsset {
	std::vector<uint8_t> storage;
	std::map<std::string, ByteView> sections;
	const ByteView& require(const std::string& name, uint32_t stride = 0, uint32_t count = UINT32_MAX) const
	{
		const auto found = sections.find(name);
		if (found == sections.end()) throw std::runtime_error("Missing runtime asset section: " + name);
		if ((stride && found->second.stride != stride) || (count != UINT32_MAX && found->second.count != count))
			throw std::runtime_error("Runtime asset section shape mismatch: " + name);
		return found->second;
	}
};

inline std::string fixedString(const char* value, size_t capacity)
{
	const void* zero = std::memchr(value, 0, capacity);
	return std::string(value, zero ? static_cast<const char*>(zero) : value + capacity);
}

inline void validatePayload(const std::vector<uint8_t>& bytes, const AssetHeader& header, uint64_t directoryEnd)
{
	const uint64_t expectedOffset = (directoryEnd + alignment - 1u) / alignment * alignment;
	if (!header.count || header.fileSize != bytes.size() || header.payloadOffset != expectedOffset || header.payloadOffset > bytes.size())
		throw std::runtime_error("Invalid runtime asset directory declaration");
	const auto actual = sha256(std::span<const uint8_t>(bytes).subspan(static_cast<size_t>(header.payloadOffset)));
	if (std::memcmp(actual.data(), header.payloadSha256, actual.size()) != 0) throw std::runtime_error("Runtime asset payload SHA-256 mismatch");
}

inline SectionedAsset loadSectioned(const std::filesystem::path& path, const char expectedMagic[8], uint32_t expectedVersion)
{
	SectionedAsset result;
	result.storage = vgnn::readFile(path);
	if (result.storage.size() < sizeof(AssetHeader)) throw std::runtime_error("Runtime asset is shorter than its header");
	AssetHeader header{};
	std::memcpy(&header, result.storage.data(), sizeof(header));
	if (std::memcmp(header.magic, expectedMagic, 8) || header.version != expectedVersion) throw std::runtime_error("Runtime asset magic/version mismatch");
	const uint64_t directoryEnd = sizeof(AssetHeader) + static_cast<uint64_t>(header.count) * sizeof(SectionEntry);
	if (directoryEnd > result.storage.size()) throw std::runtime_error("Runtime asset section directory is truncated");
	validatePayload(result.storage, header, directoryEnd);
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	for (uint32_t i = 0; i < header.count; ++i) {
		SectionEntry entry{};
		std::memcpy(&entry, result.storage.data() + sizeof(AssetHeader) + i * sizeof(entry), sizeof(entry));
		const std::string name = fixedString(entry.name, sizeof(entry.name));
		const uint64_t size = static_cast<uint64_t>(entry.count) * entry.stride;
		if (name.empty() || result.sections.contains(name) || !entry.stride || entry.offset < header.payloadOffset || entry.offset % alignment
			|| entry.offset > result.storage.size() || size > result.storage.size() - entry.offset)
			throw std::runtime_error("Invalid runtime asset section entry");
		result.sections.emplace(name, ByteView{ entry.offset, entry.count, entry.stride,
			std::span<const uint8_t>(result.storage).subspan(static_cast<size_t>(entry.offset), static_cast<size_t>(size)) });
		ranges.emplace_back(entry.offset, entry.offset + size);
	}
	std::sort(ranges.begin(), ranges.end());
	for (size_t i = 1; i < ranges.size(); ++i) if (ranges[i - 1].second > ranges[i].first) throw std::runtime_error("Overlapping runtime asset sections");
	return result;
}

struct TensorView : ByteView { std::vector<uint32_t> shape; };
struct TensorAsset {
	std::vector<uint8_t> storage;
	std::map<std::string, TensorView> tensors;
	const TensorView& require(const std::string& name, std::initializer_list<uint32_t> shape = {}) const
	{
		const auto found = tensors.find(name);
		if (found == tensors.end()) throw std::runtime_error("Missing Fine15 tensor: " + name);
		if (shape.size() && (shape.size() != found->second.shape.size()
			|| !std::equal(shape.begin(), shape.end(), found->second.shape.begin())))
			throw std::runtime_error("Fine15 tensor shape mismatch: " + name);
		return found->second;
	}
};

inline TensorAsset loadTensorAsset(const std::filesystem::path& path)
{
	TensorAsset result;
	result.storage = vgnn::readFile(path);
	if (result.storage.size() < sizeof(AssetHeader)) throw std::runtime_error("VHOOD is shorter than its header");
	AssetHeader header{};
	std::memcpy(&header, result.storage.data(), sizeof(header));
	if (std::memcmp(header.magic, "VHOOD001", 8) || header.version != 1) throw std::runtime_error("VHOOD magic/version mismatch");
	const uint64_t directoryEnd = sizeof(AssetHeader) + static_cast<uint64_t>(header.count) * sizeof(TensorEntry);
	if (directoryEnd > result.storage.size()) throw std::runtime_error("VHOOD tensor directory is truncated");
	validatePayload(result.storage, header, directoryEnd);
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	for (uint32_t i = 0; i < header.count; ++i) {
		TensorEntry entry{};
		std::memcpy(&entry, result.storage.data() + sizeof(AssetHeader) + i * sizeof(entry), sizeof(entry));
		const std::string name = fixedString(entry.name, sizeof(entry.name));
		if (name.empty() || result.tensors.contains(name) || entry.rank < 1 || entry.rank > 8 || entry.offset % alignment || entry.offset < header.payloadOffset)
			throw std::runtime_error("Invalid VHOOD tensor entry");
		uint64_t count = 1;
		std::vector<uint32_t> shape(entry.dimensions, entry.dimensions + entry.rank);
		for (uint32_t dimension : shape) {
			if (!dimension || count > UINT32_MAX / dimension) throw std::runtime_error("Invalid VHOOD tensor dimensions");
			count *= dimension;
		}
		const uint64_t size = count * sizeof(float);
		if (count != entry.count || entry.offset > result.storage.size() || size > result.storage.size() - entry.offset)
			throw std::runtime_error("Invalid VHOOD tensor range");
		result.tensors.emplace(name, TensorView{ { entry.offset, entry.count, sizeof(float),
			std::span<const uint8_t>(result.storage).subspan(static_cast<size_t>(entry.offset), static_cast<size_t>(size)) }, std::move(shape) });
		ranges.emplace_back(entry.offset, entry.offset + size);
	}
	std::sort(ranges.begin(), ranges.end());
	for (size_t i = 1; i < ranges.size(); ++i) if (ranges[i - 1].second > ranges[i].first) throw std::runtime_error("Overlapping VHOOD tensors");
	return result;
}
}
