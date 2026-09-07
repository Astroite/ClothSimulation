#pragma once
#include "demo_gpu.h"
#include "demo_surface.h"

namespace mlcloth::demo {
// TinyHOOD is a stateless acceleration evaluator of the *corrected* physical
// state. Its held acceleration belongs to GpuPhysics and its checkpoints.
class GpuGnn {
public:
    GpuGnn();
    ~GpuGnn();
    void build(vks::VulkanDevice*,VkQueue,VkPipelineCache,const std::filesystem::path& shaders,
               const std::filesystem::path& weights,const Mesh&,const GpuPhysics&,bool temporal=false,uint32_t bodyCapacity=8192);
    bool ready()const;
    bool captureValidation{};
    void record(VkCommandBuffer,const std::vector<Vec3>& futurePins,
                const TriangleCollider& current,const TriangleCollider& future,
                float strength,float trustRatio,float gravity);
    void record(VkCommandBuffer,const std::vector<Vec3>& futurePins,const BodySurfaceFrame&,
                float strength,float trustRatio,float gravity);
    bool temporal()const;
    uint64_t allocatedBytes()const; // device buffer allocations, includes checkpoints
    void resetHistory(); // caller waits for in-flight work
    void recordCheckpoint(VkCommandBuffer,uint64_t id);
    void restoreCheckpoint(uint64_t id);
    void retainCheckpoints(const std::vector<uint64_t>& ids);
    // Explicit validation only: normalized features and raw decoder displacement.
    void dumpValidation(const std::filesystem::path&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
