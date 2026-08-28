// Validate a baked `.mlbody` against the model it claims to belong to.
//
// Same role as `mesh_validate`: the gate between baking a body and running the sample, and
// the cross-check between the two independent implementations of the same contract --
// `tools/bake_mlcloth_body.py` composes the bind pose, inverts the coordinate mapping and
// merges the folded influences in Python, and `parse_body` re-checks the invariants that
// survive into the file in C++. Neither side is trusted to be self-consistent.
//
// What this cannot check is registration: whether the body is *this* character in *this*
// pose needs per-frame bone transforms, so that gate lives in the runtime's `--verify` pass,
// where the garment's distance to the skin is measured.

#include "mlcloth_formats.h"

#include <cmath>
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
    if (argc != 3) {
        std::cerr << "usage: body_validate <model.enc> <body.mlbody>\n";
        return 2;
    }
    std::string error;
    const std::vector<uint8_t> modelBytes = readFile(argv[1], error);
    if (modelBytes.empty()) { std::cerr << error << "\n"; return 2; }
    mlcloth::ModelInfo modelInfo{};
    if (!mlcloth::parse_model(modelBytes.data(), modelBytes.size(), modelInfo, error)) {
        std::cerr << "model rejected: " << error << "\n";
        return 1;
    }

    const std::vector<uint8_t> bodyBytes = readFile(argv[2], error);
    if (bodyBytes.empty()) { std::cerr << error << "\n"; return 2; }
    mlcloth::BodyInfo body{};
    if (!mlcloth::parse_body(bodyBytes.data(), bodyBytes.size(), mlcloth::sha256(modelBytes), body, error)) {
        std::cerr << "body rejected: " << error << "\n";
        return 1;
    }

    // Anatomy, re-derived here rather than trusted from the bake's report: the rest pose is in
    // component centimetres, so the lowest vertex is the floor and the mesh straddles x = 0. A
    // coordinate mapping inverted wrongly lands metres away from both.
    float lowest = body.restPositionsCm[2], highest = body.restPositionsCm[2];
    float minimumX = body.restPositionsCm[0], maximumX = body.restPositionsCm[0];
    for (uint32_t v = 1; v < body.vertices; ++v) {
        const float* p = body.restPositionsCm + size_t(v) * 3;
        lowest = std::min(lowest, p[2]);
        highest = std::max(highest, p[2]);
        minimumX = std::min(minimumX, p[0]);
        maximumX = std::max(maximumX, p[0]);
    }

    std::map<uint32_t, uint32_t> perDriver;
    for (uint32_t v = 0; v < body.vertices; ++v) {
        for (uint32_t k = 0; k < body.influences; ++k) {
            const size_t entry = size_t(v) * body.influences + k;
            if (body.boneWeights[entry] > 0.0f) ++perDriver[body.boneIndices[entry]];
        }
    }

    std::cout << std::fixed << std::setprecision(3)
              << "body accepted: " << body.vertices << " vertices, " << body.triangles
              << " triangles, " << body.influences << " influences, " << perDriver.size()
              << " of " << body.drivers << " driver bones used, " << body.foldedBones << " folded\n"
              << "rest pose: floor z = " << lowest << " cm, height " << (highest - lowest)
              << " cm, x span " << minimumX << ".." << maximumX << " cm\n";
    return 0;
}
