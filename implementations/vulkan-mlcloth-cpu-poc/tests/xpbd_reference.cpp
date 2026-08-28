// Run one XPBD scenario from a binary description and write the result back.
//
// This exists so `tests/test_mlcloth_xpbd.py` can hand the same scenario to
// `vulkan-gnn-poc/real_scene/xpbd.py::project` and to this port and compare, which is the
// only assertion here that is not checking the port against its own assumptions. Writing
// it paid for itself before it ran: reading the reference closely enough to build it
// surfaced two errors in the guide pass -- confidence multiplied onto the correction
// instead of dividing into alpha-tilde, and measured against the evolving position
// instead of the inertial extrapolation.
//
// The format is deliberately dumb and self-describing rather than JSON: this is a test
// fixture channel, and a float parser between two float implementations would be one more
// thing that could be the reason the numbers differ.
//
//   magic          char[8]  "MLXPBSC1"
//   vertexCount    u32
//   edgeCount      u32
//   triangleCount  u32
//   capsuleCount   u32
//   bendStart      u32      index of the first bend pair; == edgeCount for stretch only
//   iterations     i32
//   flags          u32      bit 0 one-sided, bit 1 collision, bit 2 area, bit 3 guide,
//                           bit 4 an inertial block is present
//   timestep       f32
//   relaxation     f32
//   stretchCompliance f32
//   bendCompliance f32
//   areaFloor      f32
//   areaCompliance f32
//   guideCompliance f32
//   guideTrustRatio f32
//   contactOffset  f32
//   guide          f32[vertexCount * 3]
//   inertial       f32[vertexCount * 3]   (only when bit 4 is set)
//   edges          u32[edgeCount * 2]
//   triangles      u32[triangleCount * 3]
//   targetLength   f32[edgeCount]
//   targetArea     f32[triangleCount]
//   mass           f32[vertexCount]
//   pinMask        u32[vertexCount]
//   capsules       f32[capsuleCount * 8]  centre xyz, axis xyz, radius, halfLength
//
// Output is `f32[vertexCount * 3]` followed by the four StepStats fields as
// u32, u32, u32, f32.

#include "mlcloth_xpbd.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

struct Cursor {
    const std::vector<char>& data;
    size_t offset{ 0 };
    bool ok{ true };

    const char* take(size_t bytes) {
        if (!ok || offset + bytes > data.size()) {
            ok = false;
            return nullptr;
        }
        const char* pointer = data.data() + offset;
        offset += bytes;
        return pointer;
    }

    template <typename T>
    T scalar() {
        T value{};
        if (const char* pointer = take(sizeof(T))) std::memcpy(&value, pointer, sizeof(T));
        return value;
    }

    template <typename T>
    std::vector<T> array(size_t count) {
        std::vector<T> values(count);
        if (count == 0) return values;
        if (const char* pointer = take(sizeof(T) * count)) std::memcpy(values.data(), pointer, sizeof(T) * count);
        return values;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: xpbd_reference <scenario.bin> <result.bin>\n";
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        std::cerr << "cannot open scenario: " << argv[1] << "\n";
        return 2;
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    Cursor cursor{ blob };

    const char* magic = cursor.take(8);
    if (!magic || std::memcmp(magic, "MLXPBSC1", 8) != 0) {
        std::cerr << "scenario magic is wrong\n";
        return 2;
    }
    const uint32_t vertexCount = cursor.scalar<uint32_t>();
    const uint32_t edgeCount = cursor.scalar<uint32_t>();
    const uint32_t triangleCount = cursor.scalar<uint32_t>();
    const uint32_t capsuleCount = cursor.scalar<uint32_t>();
    const uint32_t bendStart = cursor.scalar<uint32_t>();

    mlcloth::XpbdConfig config{};
    config.iterations = cursor.scalar<int32_t>();
    const uint32_t flags = cursor.scalar<uint32_t>();
    config.oneSided = (flags & 1u) != 0;
    config.collision = (flags & 2u) != 0;
    config.timestep = cursor.scalar<float>();
    config.relaxation = cursor.scalar<float>();
    config.stretchCompliance = cursor.scalar<float>();
    config.bendCompliance = cursor.scalar<float>();
    config.areaFloor = cursor.scalar<float>();
    config.areaCompliance = cursor.scalar<float>();
    config.guideCompliance = cursor.scalar<float>();
    config.guideTrustRatio = cursor.scalar<float>();
    config.contactOffset = cursor.scalar<float>();
    if ((flags & 4u) == 0) config.areaFloor = 0.0f;
    if ((flags & 8u) == 0) config.guideCompliance = -1.0f;

    const std::vector<float> guide = cursor.array<float>(size_t(vertexCount) * 3);
    std::vector<float> inertial;
    if (flags & 16u) inertial = cursor.array<float>(size_t(vertexCount) * 3);
    const std::vector<uint32_t> edges = cursor.array<uint32_t>(size_t(edgeCount) * 2);
    const std::vector<uint32_t> triangles = cursor.array<uint32_t>(size_t(triangleCount) * 3);
    const std::vector<float> targetLength = cursor.array<float>(edgeCount);
    const std::vector<float> targetArea = cursor.array<float>(triangleCount);
    const std::vector<float> mass = cursor.array<float>(vertexCount);
    const std::vector<uint32_t> pinMask = cursor.array<uint32_t>(vertexCount);
    const std::vector<float> capsuleData = cursor.array<float>(size_t(capsuleCount) * 8);
    if (!cursor.ok) {
        std::cerr << "scenario is truncated\n";
        return 2;
    }
    if (cursor.offset != blob.size()) {
        std::cerr << "scenario has " << (blob.size() - cursor.offset) << " trailing bytes\n";
        return 2;
    }

    std::vector<mlcloth::Capsule> capsules(capsuleCount);
    for (uint32_t index = 0; index < capsuleCount; ++index) {
        const float* source = &capsuleData[size_t(index) * 8];
        for (int axis = 0; axis < 3; ++axis) {
            capsules[index].centre[axis] = source[axis];
            capsules[index].axis[axis] = source[3 + axis];
        }
        capsules[index].radius = source[6];
        capsules[index].halfLength = source[7];
    }

    mlcloth::XpbdSolver solver;
    std::string err;
    // The scenario supplies the whole constraint list, stretch pairs first, so the harness
    // can drive a stretch-only configuration (which is what a comparison against
    // `build_constraints(include_bend=False)` needs) as well as a bend one.
    if (!solver.build(vertexCount, edges.data(), edgeCount, bendStart,
                      triangleCount ? triangles.data() : nullptr, triangleCount,
                      mass.data(), pinMask.data(), targetLength.data(),
                      triangleCount ? targetArea.data() : nullptr, err)) {
        std::cerr << "solver build failed: " << err << "\n";
        return 2;
    }
    solver.configure(config);

    std::vector<float> out(size_t(vertexCount) * 3, 0.0f);
    solver.step(guide.data(), inertial.empty() ? nullptr : inertial.data(), capsules, out.data());

    std::ofstream output(argv[2], std::ios::binary);
    if (!output) {
        std::cerr << "cannot write result: " << argv[2] << "\n";
        return 2;
    }
    output.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * sizeof(float)));
    const mlcloth::XpbdSolver::StepStats& stats = solver.lastStep();
    const uint32_t counters[3] = { stats.contactsResolved, stats.areaFloorsActive, stats.stretchActive };
    output.write(reinterpret_cast<const char*>(counters), sizeof(counters));
    output.write(reinterpret_cast<const char*>(&stats.maxCorrectionM), sizeof(float));
    if (!output) {
        std::cerr << "result write failed\n";
        return 2;
    }
    return 0;
}
