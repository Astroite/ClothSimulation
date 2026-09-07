#pragma once
#include "demo_physics.h"
#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include <array>
#include <filesystem>
#include <memory>

namespace mlcloth::demo {
// Persistent compute state. A caller must wait for GPU work before reset/destruction.
class GpuPhysics {
public:
    GpuPhysics();
    ~GpuPhysics();
    GpuPhysics(const GpuPhysics&)=delete;
    GpuPhysics& operator=(const GpuPhysics&)=delete;
    void build(vks::VulkanDevice*,VkQueue,VkPipelineCache,const std::filesystem::path&,const Mesh&);
    void reset(const std::vector<Vec3>&,const std::vector<Vec3>& velocities={});
    void recordCheckpoint(VkCommandBuffer,uint64_t id);
    void restoreCheckpoint(uint64_t id);
    void retainCheckpoints(const std::vector<uint64_t>& ids); // caller waits for in-flight work
    void recordStep(VkCommandBuffer,float,const std::vector<Vec3>& pins,const std::vector<Vec3>* guide,
                    const TriangleCollider*,const PhysicsConfig&,VkQueryPool profile=VK_NULL_HANDLE,uint32_t firstQuery=0,
                    const GpuPhysics* sharedBody=nullptr);
    const vks::Buffer& positionBuffer() const;
    const vks::Buffer& previousPositionBuffer() const;
    const vks::Buffer& velocityBuffer() const;
    const vks::Buffer& accelerationBuffer() const;
    std::vector<Vec3> readback();
    std::array<uint32_t,7> contactDiagnostics(); // explicit validation readback
    bool captureContacts{};
    bool cacheSelfContacts{false};
    std::vector<SelfContactRecord> readContactRecords();
    std::array<uint32_t,3> candidateCacheDiagnostics(); // rebuilds, reuses, overflow queries
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
