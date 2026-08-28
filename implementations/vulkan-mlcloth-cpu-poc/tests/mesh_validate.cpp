// Validate a baked `.mlmesh` against the model it claims to belong to.
//
// Two things this is for. As a CLI step it is the gate between baking a mesh and
// running the sample: `parse_mesh` is strict, so a mesh that passes here will load,
// and one that fails says exactly which invariant it broke. As a test it is the
// cross-check between the two independent implementations of the same contract --
// tools/bake_cloth_topology.py derives the edge set, both CSRs and the boundary
// structure in Python, and this re-derives them in C++ and refuses the file if they
// disagree. Neither side is trusted to be self-consistent; they are pinned to each
// other.

#include "mlcloth_formats.h"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> readFile(const std::string& path, std::string& error) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) { error = "cannot open " + path; return {}; }
    const std::streamoff length = stream.tellg();
    if (length <= 0) { error = "file is empty: " + path; return {}; }
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(bytes.data()), length);
    if (!stream) { error = "cannot read " + path; return {}; }
    return bytes;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: mesh_validate <model.enc> <mesh.mlmesh> [--json <report.json>]\n";
        return 2;
    }
    std::string jsonPath;
    if (argc == 5) {
        if (std::strcmp(argv[3], "--json") != 0) { std::cerr << "unknown argument: " << argv[3] << "\n"; return 2; }
        jsonPath = argv[4];
    }

    std::string error;
    const std::vector<uint8_t> modelBytes = readFile(argv[1], error);
    if (modelBytes.empty()) { std::cerr << error << "\n"; return 2; }
    mlcloth::ModelInfo modelInfo{};
    if (!mlcloth::parse_model(modelBytes.data(), modelBytes.size(), modelInfo, error)) {
        std::cerr << "model rejected: " << error << "\n";
        return 1;
    }

    const std::vector<uint8_t> meshBytes = readFile(argv[2], error);
    if (meshBytes.empty()) { std::cerr << error << "\n"; return 2; }
    mlcloth::MeshInfo meshInfo{};
    if (!mlcloth::parse_mesh(meshBytes.data(), meshBytes.size(), mlcloth::sha256(modelBytes), meshInfo, error)) {
        std::cerr << "mesh rejected: " << error << "\n";
        return 1;
    }
    if (meshInfo.vertices != static_cast<uint32_t>(modelInfo.vertices)) {
        std::cerr << "mesh rejected: vertex count does not match the model's derived count\n";
        return 1;
    }

    std::cout << "mesh accepted: " << meshInfo.vertices << " vertices, " << meshInfo.triangles
              << " triangles, " << meshInfo.edges << " edges, " << meshInfo.boundaryEdges
              << " boundary edges in " << meshInfo.boundaryLoops << " loops, max triangle valence "
              << meshInfo.maxTriangleValence << ", " << meshInfo.pinnedVertices << " pinned\n";
    // The pin bind, reported rather than merely validated: a mesh silently missing it still
    // loads for every mode except the comparison, so its absence has to be visible at bake time
    // and not first noticed when `--compare` refuses to start.
    if (meshInfo.pinDriverIndices == nullptr) {
        std::cout << "pin bind: absent (no reference clip was measured; --compare needs it)\n";
    } else {
        std::map<uint32_t, uint32_t> perDriver;
        for (uint32_t v = 0; v < meshInfo.vertices; ++v) {
            if (meshInfo.pinDriverIndices[v] != mlcloth::kNoDriver) ++perDriver[meshInfo.pinDriverIndices[v]];
        }
        std::cout << "pin bind: " << perDriver.size() << " driver bone(s):";
        for (const auto& entry : perDriver) {
            std::cout << ' ' << modelInfo.driverNames[entry.first] << '(' << entry.second << ')';
        }
        std::cout << '\n';
    }

    if (!jsonPath.empty()) {
        std::ofstream report(jsonPath);
        if (!report) { std::cerr << "cannot write " << jsonPath << "\n"; return 2; }
        report << "{\n  \"model_sha256\": \"" << mlcloth::sha256_hex(mlcloth::sha256(modelBytes)) << "\","
               << "\n  \"payload_sha256\": \"" << mlcloth::sha256_hex(meshInfo.payloadSha256) << "\","
               << "\n  \"vertices\": " << meshInfo.vertices
               << ",\n  \"triangles\": " << meshInfo.triangles
               << ",\n  \"edges\": " << meshInfo.edges
               << ",\n  \"boundary_edges\": " << meshInfo.boundaryEdges
               << ",\n  \"boundary_loops\": " << meshInfo.boundaryLoops
               << ",\n  \"max_triangle_valence\": " << meshInfo.maxTriangleValence
               << ",\n  \"pinned_vertices\": " << meshInfo.pinnedVertices
               // Zero for a disc or an annulus; a non-zero value means the edge
               // derivation and the triangle list disagree about the surface.
               << ",\n  \"euler_characteristic\": "
               << (static_cast<int64_t>(meshInfo.vertices) - meshInfo.edges + meshInfo.triangles)
               << ",\n  \"accepted\": true\n}\n";
    }
    return 0;
}
