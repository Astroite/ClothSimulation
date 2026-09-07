#include "demo_physics.h"
#include <json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace mlcloth::demo;

static nlohmann::json packVecs(const std::vector<Vec3>& v) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& p : v) arr.push_back({p.x, p.y, p.z});
    return arr;
}

static nlohmann::json runFixture(
    const char* name,
    const Mesh& mesh,
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>& velocities,
    const std::vector<Vec3>& pinTargets,
    const PhysicsConfig& config,
    float dt,
    const TriangleCollider* collider) {

    PhysicsSolver solver;
    solver.build(mesh);
    solver.reset(positions, velocities);
    solver.step(dt, pinTargets, nullptr, collider, config);

    nlohmann::json out;
    out["name"] = name;
    out["rest"] = packVecs(mesh.rest);
    out["triangles"] = mesh.triangles;
    out["mass"] = mesh.mass;
    out["pinned"] = mesh.pinned;
    out["positions"] = packVecs(positions);
    out["velocities"] = packVecs(velocities);
    out["pin_targets"] = packVecs(pinTargets);
    out["dt"] = dt;
    out["config"] = {
        {"gravity", config.gravity},
        {"iterations", config.iterations},
        {"dampingPerSecond", config.dampingPerSecond},
        {"stretchCompliance", config.stretchCompliance},
        {"shearCompliance", config.shearCompliance},
        {"bendCompliance", config.bendCompliance},
        {"enableCollision", config.enableCollision},
        {"enableSelfCollision", config.enableSelfCollision},
        {"thickness", config.thickness},
        {"friction", config.friction},
    };

    if (collider) {
        nlohmann::json col;
        col["previous"] = packVecs(collider->previous);
        col["current"] = packVecs(collider->current);
        col["triangles"] = collider->triangles;
        col["capsules"] = nlohmann::json::array();
        for (const auto& cap : collider->capsules) {
            col["capsules"].push_back({
                {"previousA", {cap.previousA.x, cap.previousA.y, cap.previousA.z}},
                {"previousB", {cap.previousB.x, cap.previousB.y, cap.previousB.z}},
                {"currentA", {cap.currentA.x, cap.currentA.y, cap.currentA.z}},
                {"currentB", {cap.currentB.x, cap.currentB.y, cap.currentB.z}},
                {"radius", cap.radius},
            });
        }
        out["collider"] = col;
    } else {
        out["collider"] = nullptr;
    }

    out["expected_positions"] = packVecs(solver.positions());
    out["expected_velocities"] = packVecs(solver.velocities());
    out["tolerance"] = 1e-4;
    return out;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Usage: temporal_reference_probe <output_dir>");

        std::string dir = argv[1];

        // Fixture 1: Basic stretch/bend (no contacts)
        {
            Mesh mesh;
            mesh.rest = { {0, 0, 0}, {1, 0, 0}, {0.5f, 1, 0} };
            mesh.triangles = { 0, 1, 2 };
            mesh.mass = {1, 1, 1};
            mesh.pinned = {0, 0, 0};

            std::vector<Vec3> positions = { {0, 0, 0}, {1.1f, 0, 0}, {0.5f, 0.9f, 0.1f} };
            std::vector<Vec3> velocities = { {0, 0, 0}, {0, 0, 0}, {0, -1, 0} };
            std::vector<Vec3> pinTargets = mesh.rest;

            PhysicsConfig config;
            config.gravity = -9.81f;
            config.iterations = 2;
            config.dampingPerSecond = 0.5f;

            auto result = runFixture("basic_stretch_bend", mesh, positions, velocities, pinTargets, config, 1.0f/60.0f, nullptr);
            std::ofstream f(dir + "/fixture_basic.json");
            f << result.dump(2) << std::endl;
            std::cout << "Written fixture_basic.json" << std::endl;
        }

        // Fixture 2: Active body (STM) contact
        {
            Mesh mesh;
            mesh.rest = { {0, 1, 0.2f}, {1, 1, 0}, {1, 1, 0.1f} };
            mesh.triangles = { 0, 1, 2 };
            mesh.mass = {1, 1, 1};
            mesh.pinned = {0, 1, 1};

            std::vector<Vec3> positions = mesh.rest;
            std::vector<Vec3> velocities = { {0, 0, -96}, {0, 0, 0}, {0, 0, 0} };
            std::vector<Vec3> pinTargets = mesh.rest;

            TriangleCollider body;
            body.previous = mesh.rest;
            body.current = mesh.rest;
            body.triangles = {0, 1, 2};

            PhysicsConfig config;
            config.gravity = 0;
            config.iterations = 0;
            config.dampingPerSecond = 0;
            config.enableCollision = true;
            config.thickness = 0.05f;

            auto result = runFixture("active_body_contact", mesh, positions, velocities, pinTargets, config, 1.0f/240.0f, &body);
            std::ofstream f(dir + "/fixture_body_contact.json");
            f << result.dump(2) << std::endl;
            std::cout << "Written fixture_body_contact.json" << std::endl;
        }

        // Fixture 3: Ground contact with friction
        {
            Mesh mesh;
            mesh.rest = { {0, 0, 0}, {1, 0, 0}, {0.5f, 1, 0} };
            mesh.triangles = { 0, 1, 2 };
            mesh.mass = {1, 1, 1};
            mesh.pinned = {0, 0, 0};

            std::vector<Vec3> positions = { {0, 0.005f, 0}, {1, 0.005f, 0}, {0.5f, 0.005f, 0} };
            std::vector<Vec3> velocities = { {1, -1, 0}, {1, -1, 0}, {1, -1, 0} };
            std::vector<Vec3> pinTargets = mesh.rest;

            PhysicsConfig config;
            config.gravity = 0;
            config.iterations = 0;
            config.dampingPerSecond = 0;
            config.enableCollision = true;
            config.thickness = 0.01f;
            config.friction = 0.5f;

            auto result = runFixture("ground_friction", mesh, positions, velocities, pinTargets, config, 1.0f/60.0f, nullptr);
            std::ofstream f(dir + "/fixture_ground_friction.json");
            f << result.dump(2) << std::endl;
            std::cout << "Written fixture_ground_friction.json" << std::endl;
        }

        // Fixture 4: Moving pin targets
        {
            Mesh mesh;
            mesh.rest = { {0, 0, 0}, {1, 0, 0}, {0.5f, 1, 0} };
            mesh.triangles = { 0, 1, 2 };
            mesh.mass = {1, 1, 1};
            mesh.pinned = {1, 0, 0};

            std::vector<Vec3> positions = mesh.rest;
            std::vector<Vec3> velocities = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
            std::vector<Vec3> pinTargets = { {0.1f, 0.1f, 0}, {1, 0, 0}, {0.5f, 1, 0} };

            PhysicsConfig config;
            config.gravity = -9.81f;
            config.iterations = 2;
            config.dampingPerSecond = 0.1f;

            auto result = runFixture("moving_pins", mesh, positions, velocities, pinTargets, config, 1.0f/60.0f, nullptr);
            std::ofstream f(dir + "/fixture_moving_pins.json");
            f << result.dump(2) << std::endl;
            std::cout << "Written fixture_moving_pins.json" << std::endl;
        }

        // Fixture 5: Unequal masses
        {
            Mesh mesh;
            mesh.rest = { {0, 0, 0}, {1, 0, 0}, {0.5f, 1, 0}, {0.5f, 0.5f, 0.5f} };
            mesh.triangles = { 0, 1, 2 };
            mesh.mass = {0.01f, 10.0f, 1.0f, 0.1f};
            mesh.pinned = {0, 0, 0, 0};

            std::vector<Vec3> positions = mesh.rest;
            std::vector<Vec3> velocities = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
            std::vector<Vec3> pinTargets = mesh.rest;

            PhysicsConfig config;
            config.gravity = -9.81f;
            config.iterations = 4;
            config.dampingPerSecond = 0.2f;
            config.stretchCompliance = 0.0f;
            config.shearCompliance = 0.0f;
            config.bendCompliance = 0.0f;

            auto result = runFixture("unequal_masses", mesh, positions, velocities, pinTargets, config, 1.0f/60.0f, nullptr);
            std::ofstream f(dir + "/fixture_unequal_masses.json");
            f << result.dump(2) << std::endl;
            std::cout << "Written fixture_unequal_masses.json" << std::endl;
        }

        // Fixture 6: Bend with dihedral angle
        {
            Mesh mesh;
            mesh.rest = { {0, 0, 0}, {1, 0, 0}, {0.5f, 1, 0}, {0.5f, 0, 1} };
            mesh.triangles = { 0, 1, 2, 0, 1, 3 };
            mesh.mass = {1, 1, 1, 1};
            mesh.pinned = {1, 0, 0, 1};

            std::vector<Vec3> positions = mesh.rest;
            positions[2].y = 0.5f;
            std::vector<Vec3> velocities = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
            std::vector<Vec3> pinTargets = mesh.rest;

            PhysicsConfig config;
            config.gravity = 0;
            config.iterations = 10;
            config.dampingPerSecond = 0;
            config.stretchCompliance = 0;
            config.shearCompliance = 0;
            config.bendCompliance = 0;

            auto result = runFixture("bend_dihedral", mesh, positions, velocities, pinTargets, config, 1.0f/60.0f, nullptr);
            std::ofstream f(dir + "/fixture_bend.json");
            f << result.dump(2) << std::endl;
            std::cout << "Written fixture_bend.json" << std::endl;
        }

        // Fixture 7: Capsule-only contact (empty triangles)
        {
            Mesh mesh;
            mesh.rest = { {0, 1, 0.2f}, {1, 1, 0}, {1, 1, 0.1f} };
            mesh.triangles = { 0, 1, 2 };
            mesh.mass = {1, 1, 1};
            mesh.pinned = {0, 1, 1};

            std::vector<Vec3> positions = mesh.rest;
            std::vector<Vec3> velocities = { {0, 0, -96}, {0, 0, 0}, {0, 0, 0} };
            std::vector<Vec3> pinTargets = mesh.rest;

            TriangleCollider body;
            body.previous = mesh.rest;
            body.current = mesh.rest;
            body.triangles = {}; // empty triangles
            body.capsules.push_back({{0,0.8f,0},{0,1.2f,0},{0,0.8f,0},{0,1.2f,0},0.05f});

            PhysicsConfig config;
            config.gravity = 0;
            config.iterations = 0;
            config.dampingPerSecond = 0;
            config.enableCollision = true;
            config.thickness = 0.003f;

            auto result = runFixture("capsule_only", mesh, positions, velocities, pinTargets, config, 1.0f/240.0f, &body);
            std::ofstream f(dir + "/fixture_capsule_only.json");
            f << result.dump(2) << std::endl;
            std::cout << "Written fixture_capsule_only.json" << std::endl;
        }

        std::cout << "\nAll fixtures written to " << dir << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
