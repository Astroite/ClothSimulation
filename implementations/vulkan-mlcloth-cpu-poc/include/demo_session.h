#pragma once
#include "demo_assets.h"
#include "mlcloth_runtime.h"
#include "demo_physics.h"
#include <array>
#include <deque>
#include <future>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <memory>
#include <optional>
#include <functional>

namespace mlcloth::demo {
struct Prediction {
    uint64_t tick{},generation{};
    std::vector<glm::vec3> localCm;
    MLClothSequenceState state;
    double milliseconds{};
};

// A serial, bounded stream owns its AILab handle. Synchronised A/C share this stream.
class InferenceStream {
public:
    InferenceStream(const DemoAssetManifest& assets,const Animation& animation,int threads,bool loop,
                    std::shared_ptr<const Prediction> seed={});
    ~InferenceStream();
    std::shared_ptr<const Prediction> at(uint64_t tick);
    void prefetch(uint64_t tick);
private:
    void work();
    const DemoAssetManifest& assets_;
    const Animation& animation_;
    int threads_{};bool loop_{},stopping_{};
    struct Job { uint64_t tick; std::promise<std::shared_ptr<const Prediction>> promise; };
    std::deque<Job> jobs_;
    std::map<uint64_t,std::shared_future<std::shared_ptr<const Prediction>>> results_;
    uint64_t scheduled_{};
    std::shared_ptr<const Prediction> seed_;
    std::mutex mutex_;std::condition_variable wake_,space_;
    std::thread worker_;
};

struct DemoSettings {
    bool synchronized{true},paused{false},loop{true},inPlace{false};
    bool autoSpacing{true};
    float speed{1},spacing{2.5f};
    int physicsHz{240},threads{1},focus{-1},view{0},material{0};
    int collisionMode{}; // 0 STM + fitted leg capsules; 1 STM; 2 leg capsules (diagnostic)
    int hybridAlgorithm{}; // 0 MLCloth shape guidance; 1 TinyHOOD, 2 temporal TinyHOOD learned acceleration
    float gnnStrength{1},gnnTrust{2};
    bool autoCamera{true},showStats{false},showSettings{false},showCollision{false};
    PhysicsConfig physics{2,1e-7f,1e-6f,1e4f,1e-3f,.7f,.1f,.003f,-9.81f,true,true};
};
struct PhysicsStep {
    int actor{};float dt{};uint64_t generation{},checkpoint{};
    std::vector<Vec3> pins,guide;
    TriangleCollider collider;
    PhysicsConfig config;
    bool inferGnn{};double gnnTime{};
    std::vector<Vec3> gnnPins;
    TriangleCollider gnnCurrent,gnnFuture;
};
struct StateCheckpoint {
    uint64_t id{},steps{};
    std::shared_ptr<const Prediction> prediction;
    std::vector<Vec3> positions,velocities; // GPU backend stores these in device buffers.
};
struct Actor {
    int animation{};bool paused{},interpolationReady{};double time{};
    Pose pose;
    PhysicsSolver solver;
    std::shared_ptr<InferenceStream> inference;
    std::vector<glm::vec3> cloth,previousCloth,pinTargets,guide,colliderPoints;
    TriangleCollider collider;
    glm::vec3 displayOffset{};
    double inferenceMs{},solveMs{};
    uint64_t generation{1},steps{};
    uint64_t restoreCheckpoint{};
    std::map<uint64_t,StateCheckpoint> checkpoints;
    std::optional<uint64_t> seekTarget;
};
struct Presentation {
    Pose pose;
    glm::vec3 displayOffset{};
    double time{};
    float blend{1}; // previous/current physics endpoints, same time for body and cloth
};
class DemoSession {
public:
    DemoAssetManifest assets;
    DemoSettings settings;
    std::array<Actor,3> actors;
    uint64_t generation{1};
    double accumulator{};
    bool renderOnly{}; // explicit renderer validation; never a quality fallback
    bool gpuMode{};
    std::deque<PhysicsStep> physicsSteps;
    std::string status;
    void load(const std::filesystem::path& manifest);
    void reset(int actor=-1);
    void select(int actor,int animation);
    void synchronize(bool enabled);
    void seek(int actor,double time);
    void update(double seconds);
    void step(int actor,float dt);
    void saveSettings() const;
    void loadSettings();
    bool seeking() const;
    TriangleCollider diagnosticCollider(int actor,bool presentation=false) const;
    Presentation presentation(int actor) const;
    glm::vec3 instanceOffset(int actor) const;
    void warmup(int actor,const std::function<void(const PhysicsStep&)>& consume) const;
    const Mesh& physicsMesh()const{return mesh_;}
private:
    Mesh mesh_;
    uint64_t checkpointSerial_{};
    void initialize(int index);
    void updateDisplay(int index);
    void updateCollider(const Pose&,TriangleCollider&,std::vector<glm::vec3>&,bool resetHistory=false) const;
    uint64_t checkpoint(int index);
    void restore(int index,const StateCheckpoint& state);
};
}
