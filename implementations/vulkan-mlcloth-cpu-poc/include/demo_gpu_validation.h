#pragma once
#include "demo_gpu.h"
#include <json.hpp>
namespace mlcloth::demo {
nlohmann::json validateGpuPhysics(vks::VulkanDevice*,VkQueue,VkPipelineCache,const std::filesystem::path&);
}
