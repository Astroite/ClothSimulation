/*
* Vulkan GNN cloth proof of concept
*
* The rendering, queue ownership and ping-pong structure is derived from
* SaschaWillems/Vulkan examples/computecloth (MIT).
*/

#include "vulkanexamplebase.h"
#include "vgnn_format.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

class VulkanExample : public VulkanExampleBase
{
public:
	enum Solver : int32_t { MassSpring = 0, GNN = 1 };
	static constexpr uint32_t xpbdColorCount = 16;

	struct Particle {
		glm::vec4 pos{};
		glm::vec4 vel{};
		glm::vec4 uv{};
		glm::vec4 normal{};
		glm::vec4 rest{};
	};
	static_assert(sizeof(Particle) == 80);

	struct TimingSample {
		double layer0Ms{};
		double layer1Ms{};
		double totalMs{};
	};

	uint32_t readSet{ 0 };
	uint32_t indexCount{ 0 };
	bool simulateWind{ false };
	bool animateSphere{ true };
	float sphereMotionTime{ 0.0f };
	bool dedicatedComputeQueue{ false };
	bool resetRequested{ false };
	bool verifyMode{ false };
	bool verificationDone{ false };
	bool gnnBenchmarkMode{ false };
	uint32_t verificationFrame{ 0 };
	double goldenMaxAbsolute{ 0.0 };
	double goldenMeanAbsolute{ 0.0 };
	std::vector<glm::vec4> repeatabilityBaseline;
	int32_t solver{ GNN };
	int32_t xpbdIterations{ 8 };
	uint32_t timestampWarmup{ 200 };
	std::filesystem::path benchmarkOutput{ "gnn_benchmark.csv" };
	std::vector<TimingSample> timingSamples;

	struct Cloth {
		glm::uvec2 gridSize{ 32, 32 };
		glm::vec2 size{ 5.0f, 5.0f };
	} cloth;

	struct StorageBuffers {
		vks::Buffer input;
		vks::Buffer output;
		vks::Buffer reset;
	} storageBuffers;

	struct GraphBuffers {
		vks::Buffer offsets;
		vks::Buffer neighbors;
		uint32_t edgeCount{};
	} graphBuffers;

	struct XpbdBuffers {
		vks::Buffer edges;
		vks::Buffer lambdas;
		uint32_t edgeCount{};
		std::array<uint32_t, xpbdColorCount + 1> colorOffsets{};
	} xpbdBuffers;

	struct GnnBuffers {
		vks::Buffer weights;
		vks::Buffer hidden;
		vks::Buffer acceleration;
	} gnnBuffers;

	struct Graphics {
		VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
		std::array<VkDescriptorSet, maxConcurrentFrames> descriptorSets{};
		VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
		VkPipeline skyPipeline{ VK_NULL_HANDLE };
		VkPipeline clothPipeline{ VK_NULL_HANDLE };
		VkPipeline spherePipeline{ VK_NULL_HANDLE };
		vks::Buffer indices;
		struct UniformData {
			glm::mat4 projection{};
			glm::mat4 modelview{};
			glm::vec4 lightPos{ -2.0f, 4.0f, -2.0f, 1.0f };
			glm::vec4 spherePosRadius{ 0.0f, 0.0f, 0.0f, 1.0f };
		} uniformData;
		std::array<vks::Buffer, maxConcurrentFrames> uniformBuffers;
	} graphics;

	struct Compute {
		struct Semaphores {
			VkSemaphore ready{ VK_NULL_HANDLE };
			VkSemaphore complete{ VK_NULL_HANDLE };
		};
		std::array<Semaphores, maxConcurrentFrames> semaphores{};
		std::array<VkFence, maxConcurrentFrames> fences{};
		std::array<VkQueryPool, maxConcurrentFrames> queryPools{};
		std::array<bool, maxConcurrentFrames> queryWritten{};
		VkQueue queue{ VK_NULL_HANDLE };
		VkCommandPool commandPool{ VK_NULL_HANDLE };
		std::array<VkCommandBuffer, maxConcurrentFrames> commandBuffers{};
		VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
		std::array<VkDescriptorSet, 2> descriptorSets{};
		VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
		VkPipeline massSpringPipeline{ VK_NULL_HANDLE };
		VkPipeline layer0Pipeline{ VK_NULL_HANDLE };
		VkPipeline layer1Pipeline{ VK_NULL_HANDLE };
		VkPipeline xpbdPipeline{ VK_NULL_HANDLE };
		VkPipeline finalizePipeline{ VK_NULL_HANDLE };
		struct alignas(16) UniformData {
			float deltaT{ 0.0f };
			float particleMass{ 0.1f };
			float springStiffness{ 2000.0f };
			float damping{ 0.25f };
			float restDistH{ 0.0f };
			float restDistV{ 0.0f };
			float restDistD{ 0.0f };
			float sphereRadius{ 1.0f };
			glm::vec4 spherePos{ 0.0f };
			glm::vec4 gravity{ 0.0f, 9.8f, 0.0f, 0.0f };
			glm::ivec2 particleCount{ 0 };
			uint32_t vertexCount{ 0 };
			uint32_t edgeCount{ 0 };
			float maxSpeed{ 8.0f };
			float maxAcceleration{ 30.0f };
			float stretchComplianceMicro{ 1.0f };
			float bendComplianceMicro{ 10000.0f };
			float xpbdVelocityDamping{ 1.5f };
			glm::vec3 sphereVelocity{ 0.0f };
		} uniformData;
		vks::Buffer uniformBuffer;
	} compute;
	static_assert(sizeof(Compute::UniformData) == 112);

	std::vector<Particle> initialParticles;
	std::vector<uint32_t> vertexOffsets;
	std::vector<uint32_t> neighborIndices;
	std::vector<glm::uvec4> xpbdEdges;
	vgnn::Model model;
	vgnn::GoldenCase golden;

	static bool hasArgument(const char* name)
	{
		return std::find_if(args.begin(), args.end(), [=](const char* value) { return std::strcmp(value, name) == 0; }) != args.end();
	}

	static std::string argumentValue(const char* name, const std::string& fallback)
	{
		for (size_t i = 0; i + 1 < args.size(); ++i) {
			if (std::strcmp(args[i], name) == 0) return args[i + 1];
		}
		return fallback;
	}

	VulkanExample() : VulkanExampleBase()
	{
		verifyMode = hasArgument("--gnn-verify");
		gnnBenchmarkMode = hasArgument("--gnn-benchmark");
		const uint32_t requestedGrid = static_cast<uint32_t>(std::strtoul(argumentValue("--gnn-grid", "32").c_str(), nullptr, 10));
		const uint32_t grid = (requestedGrid == 16 || requestedGrid == 32 || requestedGrid == 64) ? requestedGrid : 32;
		cloth.gridSize = verifyMode ? glm::uvec2(32) : glm::uvec2(grid);
		benchmarkOutput = argumentValue("--gnn-benchmark-output", "gnn_benchmark.csv");
		if (gnnBenchmarkMode || verifyMode) {
			benchmark.active = true;
			benchmark.warmup = 0;
			benchmark.duration = 600;
			benchmark.outputFrames = 1204;
			settings.overlay = false;
#if defined(_WIN32)
			setupConsole("Vulkan GNN cloth");
#endif
		}
		title = "Vulkan GNN cloth PoC";
		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, static_cast<float>(width) / static_cast<float>(height), 0.1f, 512.0f);
		camera.setRotation(glm::vec3(-30.0f, -45.0f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -7.5f));
	}

	~VulkanExample()
	{
		if (!device) return;
		vkDeviceWaitIdle(device);
		if (gnnBenchmarkMode) writeBenchmarkCsv();

		graphics.indices.destroy();
		for (auto& buffer : graphics.uniformBuffers) buffer.destroy();
		vkDestroyPipeline(device, graphics.skyPipeline, nullptr);
		vkDestroyPipeline(device, graphics.clothPipeline, nullptr);
		vkDestroyPipeline(device, graphics.spherePipeline, nullptr);
		vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout, nullptr);

		compute.uniformBuffer.destroy();
		vkDestroyPipeline(device, compute.massSpringPipeline, nullptr);
		vkDestroyPipeline(device, compute.layer0Pipeline, nullptr);
		vkDestroyPipeline(device, compute.layer1Pipeline, nullptr);
		vkDestroyPipeline(device, compute.xpbdPipeline, nullptr);
		vkDestroyPipeline(device, compute.finalizePipeline, nullptr);
		vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
		for (auto fence : compute.fences) vkDestroyFence(device, fence, nullptr);
		for (auto semaphore : compute.semaphores) {
			vkDestroySemaphore(device, semaphore.ready, nullptr);
			vkDestroySemaphore(device, semaphore.complete, nullptr);
		}
		for (auto pool : compute.queryPools) vkDestroyQueryPool(device, pool, nullptr);
		vkDestroyCommandPool(device, compute.commandPool, nullptr);

		storageBuffers.input.destroy();
		storageBuffers.output.destroy();
		storageBuffers.reset.destroy();
		graphBuffers.offsets.destroy();
		graphBuffers.neighbors.destroy();
		xpbdBuffers.edges.destroy();
		xpbdBuffers.lambdas.destroy();
		gnnBuffers.weights.destroy();
		gnnBuffers.hidden.destroy();
		gnnBuffers.acceleration.destroy();
	}

	void generateGraph(uint32_t gridSize)
	{
		vertexOffsets.clear();
		neighborIndices.clear();
		vertexOffsets.push_back(0);
		for (int32_t y = 0; y < static_cast<int32_t>(gridSize); ++y) {
			for (int32_t x = 0; x < static_cast<int32_t>(gridSize); ++x) {
				for (int32_t dy = -1; dy <= 1; ++dy) {
					for (int32_t dx = -1; dx <= 1; ++dx) {
						if (dx == 0 && dy == 0) continue;
						const int32_t nx = x + dx;
						const int32_t ny = y + dy;
						if (nx >= 0 && ny >= 0 && nx < static_cast<int32_t>(gridSize) && ny < static_cast<int32_t>(gridSize)) {
							neighborIndices.push_back(static_cast<uint32_t>(ny) * gridSize + static_cast<uint32_t>(nx));
						}
					}
				}
				vertexOffsets.push_back(static_cast<uint32_t>(neighborIndices.size()));
			}
		}
	}

	void generateXpbdEdges(uint32_t width, uint32_t height)
	{
		// Each color contains vertex-disjoint edges, so a dispatch can update both
		// endpoints in place without float atomics. Colors 0..7 are stretch/shear;
		// colors 8..15 are two-hop horizontal/vertical bend-distance constraints.
		std::array<std::vector<glm::uvec4>, xpbdColorCount> colors;
		auto vertex = [=](uint32_t x, uint32_t y) { return y * width + x; };
		for (uint32_t y = 0; y < height; ++y) {
			for (uint32_t x = 0; x + 1 < width; ++x) {
				colors[x & 1u].emplace_back(vertex(x, y), vertex(x + 1, y), 0u, 0u);
			}
		}
		for (uint32_t y = 0; y + 1 < height; ++y) {
			for (uint32_t x = 0; x < width; ++x) {
				colors[2u + (y & 1u)].emplace_back(vertex(x, y), vertex(x, y + 1), 0u, 0u);
			}
		}
		for (uint32_t y = 0; y + 1 < height; ++y) {
			for (uint32_t x = 0; x + 1 < width; ++x) {
				colors[4u + (x & 1u)].emplace_back(vertex(x, y), vertex(x + 1, y + 1), 0u, 0u);
			}
			for (uint32_t x = 1; x < width; ++x) {
				colors[6u + (x & 1u)].emplace_back(vertex(x, y), vertex(x - 1, y + 1), 0u, 0u);
			}
		}
		for (uint32_t y = 0; y < height; ++y) {
			for (uint32_t x = 0; x + 2 < width; ++x) {
				colors[8u + (x & 3u)].emplace_back(vertex(x, y), vertex(x + 2, y), 1u, 0u);
			}
		}
		for (uint32_t y = 0; y + 2 < height; ++y) {
			for (uint32_t x = 0; x < width; ++x) {
				colors[12u + (y & 3u)].emplace_back(vertex(x, y), vertex(x, y + 2), 1u, 0u);
			}
		}

		xpbdEdges.clear();
		xpbdBuffers.colorOffsets[0] = 0;
		for (uint32_t color = 0; color < xpbdColorCount; ++color) {
			xpbdEdges.insert(xpbdEdges.end(), colors[color].begin(), colors[color].end());
			xpbdBuffers.colorOffsets[color + 1] = static_cast<uint32_t>(xpbdEdges.size());
		}
		xpbdBuffers.edgeCount = static_cast<uint32_t>(xpbdEdges.size());
	}

	void loadModelAndState()
	{
		const std::filesystem::path resourceDirectory = std::filesystem::path(getShadersPath()) / "gnncloth";
		model = vgnn::loadModel(resourceDirectory / "model.bin");
		if (verifyMode) {
			golden = vgnn::loadGolden(resourceDirectory / "golden.bin");
			cloth.gridSize = glm::uvec2(golden.gridSize);
			vertexOffsets = golden.offsets;
			neighborIndices = golden.neighbors;
			compute.uniformData.gravity = glm::vec4(golden.externalAcceleration[0], golden.externalAcceleration[1], golden.externalAcceleration[2], 0.0f);
		} else {
			generateGraph(cloth.gridSize.x);
		}
		generateXpbdEdges(cloth.gridSize.x, cloth.gridSize.y);

		const uint32_t vertexCount = cloth.gridSize.x * cloth.gridSize.y;
		initialParticles.resize(vertexCount);
		const float dx = cloth.size.x / static_cast<float>(cloth.gridSize.x - 1);
		const float dy = cloth.size.y / static_cast<float>(cloth.gridSize.y - 1);
		const float du = 1.0f / static_cast<float>(cloth.gridSize.x - 1);
		const float dv = 1.0f / static_cast<float>(cloth.gridSize.y - 1);
		for (uint32_t y = 0; y < cloth.gridSize.y; ++y) {
			for (uint32_t x = 0; x < cloth.gridSize.x; ++x) {
				const uint32_t index = y * cloth.gridSize.x + x;
				Particle& particle = initialParticles[index];
				// Interactive cloth starts as a vertical sheet. Pinning the complete
				// top row produces a clothes-rack constraint instead of two corner spikes.
				const bool pinned = (y == 0);
				glm::vec3 rest(-cloth.size.x * 0.5f + dx * x, -cloth.size.y * 0.5f + dy * y, 0.0f);
				glm::vec3 position = rest;
				glm::vec3 velocity(0.0f);
				if (verifyMode) {
					rest = glm::make_vec3(&golden.restPositions[index * 3]);
					position = glm::make_vec3(&golden.positions[index * 3]);
					velocity = glm::make_vec3(&golden.velocities[index * 3]);
				}
				particle.pos = glm::vec4(position, 1.0f);
				particle.vel = glm::vec4(velocity, 0.0f);
				particle.uv = glm::vec4(du * x, dv * y, 0.0f, 0.0f);
				particle.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
				particle.rest = glm::vec4(rest, verifyMode ? golden.pinned[index] : (pinned ? 1.0f : 0.0f));
			}
		}
		graphBuffers.edgeCount = static_cast<uint32_t>(neighborIndices.size());
	}

	void addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer, VkAccessFlags srcAccess, VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
	{
		if (!dedicatedComputeQueue) return;
		std::array<VkBufferMemoryBarrier, 2> barriers{};
		for (auto& barrier : barriers) {
			barrier = vks::initializers::bufferMemoryBarrier();
			barrier.srcAccessMask = srcAccess;
			barrier.dstAccessMask = dstAccess;
			barrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
			barrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
			barrier.size = VK_WHOLE_SIZE;
		}
		barriers[0].buffer = storageBuffers.input.buffer;
		barriers[1].buffer = storageBuffers.output.buffer;
		vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
	}

	void addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer, VkAccessFlags srcAccess, VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
	{
		if (!dedicatedComputeQueue) return;
		std::array<VkBufferMemoryBarrier, 2> barriers{};
		for (auto& barrier : barriers) {
			barrier = vks::initializers::bufferMemoryBarrier();
			barrier.srcAccessMask = srcAccess;
			barrier.dstAccessMask = dstAccess;
			barrier.srcQueueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
			barrier.dstQueueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
			barrier.size = VK_WHOLE_SIZE;
		}
		barriers[0].buffer = storageBuffers.input.buffer;
		barriers[1].buffer = storageBuffers.output.buffer;
		vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
	}

	void particleComputeBarrier(VkCommandBuffer commandBuffer, uint32_t descriptorSetIndex)
	{
		VkBufferMemoryBarrier barrier = vks::initializers::bufferMemoryBarrier();
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer = descriptorSetIndex == 0 ? storageBuffers.output.buffer : storageBuffers.input.buffer;
		barrier.size = VK_WHOLE_SIZE;
		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
	}

	void beginXpbdBarriers(VkCommandBuffer commandBuffer, uint32_t descriptorSetIndex)
	{
		std::array<VkBufferMemoryBarrier, 2> barriers{};
		barriers[0] = vks::initializers::bufferMemoryBarrier();
		barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[0].buffer = descriptorSetIndex == 0 ? storageBuffers.output.buffer : storageBuffers.input.buffer;
		barriers[0].size = VK_WHOLE_SIZE;
		barriers[1] = vks::initializers::bufferMemoryBarrier();
		barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[1].buffer = xpbdBuffers.lambdas.buffer;
		barriers[1].size = VK_WHOLE_SIZE;
		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
	}

	void xpbdIterationBarrier(VkCommandBuffer commandBuffer, uint32_t descriptorSetIndex)
	{
		std::array<VkBufferMemoryBarrier, 2> barriers{};
		for (auto& barrier : barriers) {
			barrier = vks::initializers::bufferMemoryBarrier();
			barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.size = VK_WHOLE_SIZE;
		}
		barriers[0].buffer = descriptorSetIndex == 0 ? storageBuffers.output.buffer : storageBuffers.input.buffer;
		barriers[1].buffer = xpbdBuffers.lambdas.buffer;
		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
	}

	void prepareBuffers()
	{
		const VkDeviceSize particleBytes = initialParticles.size() * sizeof(Particle);
		vks::Buffer staging;
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging, particleBytes, initialParticles.data());
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &storageBuffers.input, particleBytes);
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &storageBuffers.output, particleBytes);
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &storageBuffers.reset, particleBytes, initialParticles.data());

		VkCommandBuffer copyCommand = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copy{ .size = particleBytes };
		vkCmdCopyBuffer(copyCommand, staging.buffer, storageBuffers.input.buffer, 1, &copy);
		vkCmdCopyBuffer(copyCommand, staging.buffer, storageBuffers.output.buffer, 1, &copy);
		addGraphicsToComputeBarriers(copyCommand, VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
		vulkanDevice->flushCommandBuffer(copyCommand, queue, true);
		staging.destroy();

		vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &graphBuffers.offsets, vertexOffsets.size() * sizeof(uint32_t), vertexOffsets.data());
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &graphBuffers.neighbors, neighborIndices.size() * sizeof(uint32_t), neighborIndices.data());
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &xpbdBuffers.edges, xpbdEdges.size() * sizeof(glm::uvec4), xpbdEdges.data());
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &xpbdBuffers.lambdas, xpbdEdges.size() * sizeof(float));
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &gnnBuffers.weights, model.payload.size() * sizeof(float), model.payload.data());
		const uint32_t vertexCount = static_cast<uint32_t>(initialParticles.size());
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &gnnBuffers.hidden, static_cast<VkDeviceSize>(vertexCount) * 4 * sizeof(glm::vec4));
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &gnnBuffers.acceleration, static_cast<VkDeviceSize>(vertexCount) * sizeof(glm::vec4));

		std::vector<uint32_t> indices;
		for (uint32_t y = 0; y < cloth.gridSize.y - 1; ++y) {
			for (uint32_t x = 0; x < cloth.gridSize.x; ++x) {
				indices.push_back((y + 1) * cloth.gridSize.x + x);
				indices.push_back(y * cloth.gridSize.x + x);
			}
			indices.push_back(0xFFFFFFFFu);
		}
		indexCount = static_cast<uint32_t>(indices.size());
		const VkDeviceSize indexBytes = indices.size() * sizeof(uint32_t);
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging, indexBytes, indices.data());
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &graphics.indices, indexBytes);
		copyCommand = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		copy = { .size = indexBytes };
		vkCmdCopyBuffer(copyCommand, staging.buffer, graphics.indices.buffer, 1, &copy);
		vulkanDevice->flushCommandBuffer(copyCommand, queue, true);
		staging.destroy();
	}

	void prepareDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> sizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32),
		};
		VkDescriptorPoolCreateInfo info = vks::initializers::descriptorPoolCreateInfo(sizes, 8);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &info, nullptr, &descriptorPool));
	}

	void prepareGraphics()
	{
		for (auto& buffer : graphics.uniformBuffers) {
			vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, sizeof(Graphics::UniformData));
			VK_CHECK_RESULT(buffer.map());
		}
		const auto layoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
		VkDescriptorSetLayoutCreateInfo layoutInfo = vks::initializers::descriptorSetLayoutCreateInfo(&layoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &graphics.descriptorSetLayout));
		for (uint32_t i = 0; i < maxConcurrentFrames; ++i) {
			VkDescriptorSetAllocateInfo allocate = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &graphics.descriptorSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocate, &graphics.descriptorSets[i]));
			const auto write = vks::initializers::writeDescriptorSet(graphics.descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &graphics.uniformBuffers[i].descriptor);
			vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
		}
		VkPipelineLayoutCreateInfo pipelineLayout = vks::initializers::pipelineLayoutCreateInfo(&graphics.descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayout, nullptr, &graphics.pipelineLayout));

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, 0, VK_TRUE);
		VkPipelineRasterizationStateCreateInfo rasterization = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachment = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo blend = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachment);
		VkPipelineDepthStencilStateCreateInfo depth = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewport = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisample = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamic = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStates);
		std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
			loadShader(getShadersPath() + "gnncloth/cloth.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "gnncloth/cloth.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
		};
		std::vector<VkVertexInputBindingDescription> bindings = { vks::initializers::vertexInputBindingDescription(0, sizeof(Particle), VK_VERTEX_INPUT_RATE_VERTEX) };
		std::vector<VkVertexInputAttributeDescription> attributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Particle, pos)),
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Particle, uv)),
		};
		VkPipelineVertexInputStateCreateInfo vertexInput = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
		vertexInput.pVertexBindingDescriptions = bindings.data();
		vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
		vertexInput.pVertexAttributeDescriptions = attributes.data();
		VkGraphicsPipelineCreateInfo pipeline = vks::initializers::pipelineCreateInfo(graphics.pipelineLayout, renderPass);
		pipeline.pVertexInputState = &vertexInput;
		pipeline.pInputAssemblyState = &inputAssembly;
		pipeline.pRasterizationState = &rasterization;
		pipeline.pColorBlendState = &blend;
		pipeline.pMultisampleState = &multisample;
		pipeline.pViewportState = &viewport;
		pipeline.pDepthStencilState = &depth;
		pipeline.pDynamicState = &dynamic;
		pipeline.stageCount = static_cast<uint32_t>(stages.size());
		pipeline.pStages = stages.data();
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipeline, nullptr, &graphics.clothPipeline));

		vertexInput.vertexBindingDescriptionCount = 0;
		vertexInput.vertexAttributeDescriptionCount = 0;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;
		VkPipelineDepthStencilStateCreateInfo skyDepth = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS);
		pipeline.pDepthStencilState = &skyDepth;
		stages = {
			loadShader(getShadersPath() + "gnncloth/sky.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "gnncloth/sky.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
		};
		pipeline.pStages = stages.data();
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipeline, nullptr, &graphics.skyPipeline));

		pipeline.pDepthStencilState = &depth;
		stages = {
			loadShader(getShadersPath() + "gnncloth/sphere.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "gnncloth/sphere.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
		};
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipeline, nullptr, &graphics.spherePipeline));
	}

	void prepareCompute()
	{
		vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &compute.uniformBuffer, sizeof(Compute::UniformData));
		VK_CHECK_RESULT(compute.uniformBuffer.map());
		const float dx = cloth.size.x / static_cast<float>(cloth.gridSize.x - 1);
		const float dz = cloth.size.y / static_cast<float>(cloth.gridSize.y - 1);
		compute.uniformData.restDistH = dx;
		compute.uniformData.restDistV = dz;
		compute.uniformData.restDistD = std::sqrt(dx * dx + dz * dz);
		compute.uniformData.particleCount = cloth.gridSize;
		compute.uniformData.vertexCount = static_cast<uint32_t>(initialParticles.size());
		compute.uniformData.edgeCount = graphBuffers.edgeCount;
		if (!verifyMode) {
			// Offset the sphere slightly behind the hanging sheet so collision
			// projection produces a visible out-of-plane drape.
			compute.uniformData.spherePos = glm::vec4(0.0f, 0.0f, 0.65f, 0.0f);
		}

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 6),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 7),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 8),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 9),
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(layoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));
		VkPipelineLayoutCreateInfo pipelineLayout = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);
		VkPushConstantRange pushRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(uint32_t) * 2, 0);
		pipelineLayout.pushConstantRangeCount = 1;
		pipelineLayout.pPushConstantRanges = &pushRange;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayout, nullptr, &compute.pipelineLayout));

		for (uint32_t setIndex = 0; setIndex < 2; ++setIndex) {
			VkDescriptorSetAllocateInfo allocate = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocate, &compute.descriptorSets[setIndex]));
			vks::Buffer* input = setIndex == 0 ? &storageBuffers.input : &storageBuffers.output;
			vks::Buffer* output = setIndex == 0 ? &storageBuffers.output : &storageBuffers.input;
			std::vector<VkWriteDescriptorSet> writes = {
				vks::initializers::writeDescriptorSet(compute.descriptorSets[setIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &input->descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[setIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &output->descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[setIndex], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2, &compute.uniformBuffer.descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[setIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &graphBuffers.offsets.descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[setIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &graphBuffers.neighbors.descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[setIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, &gnnBuffers.weights.descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[setIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6, &gnnBuffers.hidden.descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[setIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7, &gnnBuffers.acceleration.descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[setIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8, &xpbdBuffers.edges.descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[setIndex], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9, &xpbdBuffers.lambdas.descriptor),
			};
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		}

		auto makePipeline = [&](const std::string& name, VkPipeline& destination) {
			VkComputePipelineCreateInfo info = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
			info.stage = loadShader(getShadersPath() + "gnncloth/" + name, VK_SHADER_STAGE_COMPUTE_BIT);
			VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &info, nullptr, &destination));
		};
		makePipeline("mass_spring.comp.spv", compute.massSpringPipeline);
		makePipeline("gnn_layer0.comp.spv", compute.layer0Pipeline);
		makePipeline("gnn_layer1_integrate.comp.spv", compute.layer1Pipeline);
		makePipeline("gnn_constraints.comp.spv", compute.xpbdPipeline);
		makePipeline("gnn_finalize.comp.spv", compute.finalizePipeline);

		VkCommandPoolCreateInfo poolInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute };
		VK_CHECK_RESULT(vkCreateCommandPool(device, &poolInfo, nullptr, &compute.commandPool));
		VkCommandBufferAllocateInfo commandAllocate = vks::initializers::commandBufferAllocateInfo(compute.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, maxConcurrentFrames);
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandAllocate, compute.commandBuffers.data()));
		for (uint32_t i = 0; i < maxConcurrentFrames; ++i) {
			VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
			VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &compute.fences[i]));
			VkSemaphoreCreateInfo semaphoreInfo = vks::initializers::semaphoreCreateInfo();
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &compute.semaphores[i].ready));
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &compute.semaphores[i].complete));
			VkQueryPoolCreateInfo queryInfo{ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 3 };
			VK_CHECK_RESULT(vkCreateQueryPool(device, &queryInfo, nullptr, &compute.queryPools[i]));
		}
		VkSubmitInfo initialSignal = vks::initializers::submitInfo();
		initialSignal.signalSemaphoreCount = 1;
		initialSignal.pSignalSemaphores = &compute.semaphores[maxConcurrentFrames - 1].ready;
		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &initialSignal, VK_NULL_HANDLE));
	}

	void updateComputeUniform()
	{
		if (verifyMode) {
			// A fixed step makes the 600-frame reset/replay check deterministic.
			compute.uniformData.deltaT = 1.0f / 60.0f;
		} else if (paused) {
			compute.uniformData.deltaT = 0.0f;
		} else if (solver == MassSpring) {
			compute.uniformData.deltaT = std::min(frameTimer, 0.02f) * 0.0025f;
		} else {
			compute.uniformData.deltaT = std::min(frameTimer, 0.02f);
		}
		if (!verifyMode && !gnnBenchmarkMode) {
			if (animateSphere && !paused) sphereMotionTime += std::min(frameTimer, 0.05f);
			if (animateSphere) {
				const float xPhase = 0.7f * sphereMotionTime;
				const float zPhase = 1.1f * sphereMotionTime;
				compute.uniformData.spherePos = glm::vec4(1.2f * std::sin(xPhase), 0.0f, 0.65f + 0.55f * std::sin(zPhase), 0.0f);
				compute.uniformData.sphereVelocity = paused
					? glm::vec3(0.0f)
					: glm::vec3(0.84f * std::cos(xPhase), 0.0f, 0.605f * std::cos(zPhase));
			} else {
				compute.uniformData.spherePos = glm::vec4(0.0f, 0.0f, 0.65f, 0.0f);
				compute.uniformData.sphereVelocity = glm::vec3(0.0f);
			}
		} else {
			compute.uniformData.sphereVelocity = glm::vec3(0.0f);
		}
		if (simulateWind && !verifyMode) {
			compute.uniformData.gravity.x = std::cos(glm::radians(-timer * 360.0f)) * 2.0f;
			compute.uniformData.gravity.z = std::sin(glm::radians(timer * 360.0f)) * 2.0f;
		} else if (!verifyMode) {
			compute.uniformData.gravity.x = 0.0f;
			compute.uniformData.gravity.z = 0.0f;
		}
		std::memcpy(compute.uniformBuffer.mapped, &compute.uniformData, sizeof(compute.uniformData));
	}

	void updateGraphicsUniform()
	{
		graphics.uniformData.projection = camera.matrices.perspective;
		graphics.uniformData.modelview = camera.matrices.view;
		graphics.uniformData.spherePosRadius = glm::vec4(glm::vec3(compute.uniformData.spherePos), compute.uniformData.sphereRadius);
		std::memcpy(graphics.uniformBuffers[currentBuffer].mapped, &graphics.uniformData, sizeof(graphics.uniformData));
	}

	void prepare() override
	{
		VulkanExampleBase::prepare();
		dedicatedComputeQueue = vulkanDevice->queueFamilyIndices.graphics != vulkanDevice->queueFamilyIndices.compute;
		loadModelAndState();
		prepareBuffers();
		prepareDescriptorPool();
		prepareGraphics();
		prepareCompute();
		prepared = true;
	}

	void resetParticlesInCommandBuffer(VkCommandBuffer commandBuffer)
	{
		if (!resetRequested) return;
		const VkDeviceSize bytes = initialParticles.size() * sizeof(Particle);
		VkBufferCopy copy{ .size = bytes };
		vkCmdCopyBuffer(commandBuffer, storageBuffers.reset.buffer, storageBuffers.input.buffer, 1, &copy);
		vkCmdCopyBuffer(commandBuffer, storageBuffers.reset.buffer, storageBuffers.output.buffer, 1, &copy);
		std::array<VkBufferMemoryBarrier, 2> barriers{};
		for (auto& barrier : barriers) {
			barrier = vks::initializers::bufferMemoryBarrier();
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.size = VK_WHOLE_SIZE;
		}
		barriers[0].buffer = storageBuffers.input.buffer;
		barriers[1].buffer = storageBuffers.output.buffer;
		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
		readSet = 0;
		resetRequested = false;
	}

	void buildComputeCommandBuffer()
	{
		VkCommandBuffer commandBuffer = compute.commandBuffers[currentBuffer];
		VkCommandBufferBeginInfo begin = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &begin));
		addGraphicsToComputeBarriers(commandBuffer, 0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
		resetParticlesInCommandBuffer(commandBuffer);

		vkCmdResetQueryPool(commandBuffer, compute.queryPools[currentBuffer], 0, 3);
		if (solver == GNN) {
			readSet = 1 - readSet;
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSets[readSet], 0, nullptr);
			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, compute.queryPools[currentBuffer], 0);
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.layer0Pipeline);
			vkCmdDispatch(commandBuffer, compute.uniformData.vertexCount, 1, 1);

			VkBufferMemoryBarrier hiddenBarrier = vks::initializers::bufferMemoryBarrier();
			hiddenBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			hiddenBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			hiddenBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			hiddenBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			hiddenBarrier.buffer = gnnBuffers.hidden.buffer;
			hiddenBarrier.size = VK_WHOLE_SIZE;
			vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &hiddenBarrier, 0, nullptr);
			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, compute.queryPools[currentBuffer], 1);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.layer1Pipeline);
			vkCmdDispatch(commandBuffer, compute.uniformData.vertexCount, 1, 1);


			// XPBD owns one lambda per undirected constraint. Lambdas start at zero
			// for every time step and accumulate across iterations. The 16 edge-color
			// batches are vertex-disjoint within a dispatch, which permits in-place
			// two-endpoint updates without atomic floating-point operations.
			if (xpbdIterations > 0) {
				vkCmdFillBuffer(commandBuffer, xpbdBuffers.lambdas.buffer, 0, VK_WHOLE_SIZE, 0);
				beginXpbdBarriers(commandBuffer, readSet);
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.xpbdPipeline);
				for (int32_t iteration = 0; iteration < xpbdIterations; ++iteration) {
					for (uint32_t color = 0; color < xpbdColorCount; ++color) {
						const uint32_t edgeOffset = xpbdBuffers.colorOffsets[color];
						const uint32_t edgeCount = xpbdBuffers.colorOffsets[color + 1] - edgeOffset;
						const uint32_t pushConstants[2] = { edgeOffset, edgeCount };
						vkCmdPushConstants(commandBuffer, compute.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
						vkCmdDispatch(commandBuffer, (edgeCount + 127) / 128, 1, 1);
						xpbdIterationBarrier(commandBuffer, readSet);
					}
				}
			} else {
				particleComputeBarrier(commandBuffer, readSet);
			}
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.finalizePipeline);
			vkCmdDispatch(commandBuffer, (compute.uniformData.vertexCount + 127) / 128, 1, 1);
			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, compute.queryPools[currentBuffer], 2);
			compute.queryWritten[currentBuffer] = true;
		} else {
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.massSpringPipeline);
			uint32_t calculateNormals = 0;
			vkCmdPushConstants(commandBuffer, compute.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &calculateNormals);
			for (uint32_t iteration = 0; iteration < 64; ++iteration) {
				readSet = 1 - readSet;
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSets[readSet], 0, nullptr);
				vkCmdDispatch(commandBuffer, (cloth.gridSize.x + 9) / 10, (cloth.gridSize.y + 9) / 10, 1);
				if (iteration + 1 < 64) particleComputeBarrier(commandBuffer, readSet);
			}
			compute.queryWritten[currentBuffer] = false;
		}
		addComputeToGraphicsBarriers(commandBuffer, VK_ACCESS_SHADER_WRITE_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
		VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
	}

	void buildGraphicsCommandBuffer()
	{
		VkCommandBuffer commandBuffer = drawCmdBuffers[currentBuffer];
		VkCommandBufferBeginInfo begin = vks::initializers::commandBufferBeginInfo();
		VkClearValue clears[2]{};
		clears[0].color = { { 0.35f, 0.60f, 0.88f, 1.0f } };
		clears[1].depthStencil = { 1.0f, 0 };
		VkRenderPassBeginInfo renderBegin = vks::initializers::renderPassBeginInfo();
		renderBegin.renderPass = renderPass;
		renderBegin.renderArea.extent = { width, height };
		renderBegin.clearValueCount = 2;
		renderBegin.pClearValues = clears;
		renderBegin.framebuffer = frameBuffers[currentImageIndex];
		VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &begin));
		addComputeToGraphicsBarriers(commandBuffer, 0, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
		vkCmdBeginRenderPass(commandBuffer, &renderBegin, VK_SUBPASS_CONTENTS_INLINE);
		const VkViewport viewport = vks::initializers::viewport(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
		const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSets[currentBuffer], 0, nullptr);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.skyPipeline);
		vkCmdDraw(commandBuffer, 3, 1, 0, 0);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.spherePipeline);
		vkCmdDraw(commandBuffer, 32 * 16 * 6, 1, 0, 0);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.clothPipeline);
		vkCmdBindIndexBuffer(commandBuffer, graphics.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
		const VkBuffer latestBuffer = readSet == 0 ? storageBuffers.output.buffer : storageBuffers.input.buffer;
		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &latestBuffer, &offset);
		vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
		drawUI(commandBuffer);
		vkCmdEndRenderPass(commandBuffer);
		addGraphicsToComputeBarriers(commandBuffer, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, 0, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
		VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
	}

	void collectTimestamps(uint32_t frameIndex)
	{
		if (!compute.queryWritten[frameIndex]) return;
		std::array<uint64_t, 3> values{};
		const VkResult result = vkGetQueryPoolResults(device, compute.queryPools[frameIndex], 0, 3, sizeof(values), values.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		if (result != VK_SUCCESS) return;
		const double scale = static_cast<double>(deviceProperties.limits.timestampPeriod) / 1.0e6;
		const TimingSample sample{ (values[1] - values[0]) * scale, (values[2] - values[1]) * scale, (values[2] - values[0]) * scale };
		lastTiming = sample;
		if (gnnBenchmarkMode) {
			if (timestampWarmup > 0) --timestampWarmup;
			else if (timingSamples.size() < 1000) timingSamples.push_back(sample);
		}
	}

	TimingSample lastTiming{};

	std::vector<glm::vec4> readAccelerationForVerification()
	{
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &compute.fences[currentBuffer], VK_TRUE, UINT64_MAX));
		const VkDeviceSize bytes = static_cast<VkDeviceSize>(golden.vertexCount) * sizeof(glm::vec4);
		vks::Buffer readback;
		vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &readback, bytes);
		VkCommandBufferAllocateInfo allocate = vks::initializers::commandBufferAllocateInfo(compute.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VkCommandBuffer commandBuffer{};
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &allocate, &commandBuffer));
		VkCommandBufferBeginInfo begin = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &begin));
		VkBufferMemoryBarrier barrier = vks::initializers::bufferMemoryBarrier();
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer = gnnBuffers.acceleration.buffer;
		barrier.size = VK_WHOLE_SIZE;
		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
		VkBufferCopy copy{ .size = bytes };
		vkCmdCopyBuffer(commandBuffer, gnnBuffers.acceleration.buffer, readback.buffer, 1, &copy);
		VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
		VkSubmitInfo submit = vks::initializers::submitInfo();
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &commandBuffer;
		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &submit, VK_NULL_HANDLE));
		VK_CHECK_RESULT(vkQueueWaitIdle(compute.queue));
		VK_CHECK_RESULT(readback.map());
		const glm::vec4* actual = reinterpret_cast<const glm::vec4*>(readback.mapped);
		std::vector<glm::vec4> values(actual, actual + golden.vertexCount);
		readback.unmap();
		readback.destroy();
		vkFreeCommandBuffers(device, compute.commandPool, 1, &commandBuffer);
		return values;
	}

	void verifyAcceleration()
	{
		if (!verifyMode || verificationDone) return;
		++verificationFrame;
		if (verificationFrame != 1 && verificationFrame != 600 && verificationFrame != 1200) return;
		const std::vector<glm::vec4> actual = readAccelerationForVerification();

		if (verificationFrame == 1) {
			double sumAbsolute = 0.0;
			for (uint32_t vertex = 0; vertex < golden.vertexCount; ++vertex) {
				for (uint32_t channel = 0; channel < 3; ++channel) {
					const double difference = std::abs(static_cast<double>(actual[vertex][channel]) - golden.expectedAccelerations[vertex * 3 + channel]);
					goldenMaxAbsolute = std::max(goldenMaxAbsolute, difference);
					sumAbsolute += difference;
				}
			}
			goldenMeanAbsolute = sumAbsolute / static_cast<double>(golden.vertexCount * 3);
			if (goldenMaxAbsolute > 1.0e-4 || goldenMeanAbsolute > 1.0e-5) {
				vks::tools::exitFatal("Vulkan GNN golden verification failed", -1);
			}
			return;
		}

		if (verificationFrame == 600) {
			repeatabilityBaseline = actual;
			resetRequested = true;
			return;
		}

		double repeatabilityMaxAbsolute = 0.0;
		uint32_t healthFailures = 0;
		for (uint32_t vertex = 0; vertex < golden.vertexCount; ++vertex) {
			if (!std::isfinite(actual[vertex].x) || !std::isfinite(actual[vertex].y) || !std::isfinite(actual[vertex].z) || actual[vertex].w != 0.0f) {
				++healthFailures;
			}
			for (uint32_t channel = 0; channel < 4; ++channel) {
				repeatabilityMaxAbsolute = std::max(repeatabilityMaxAbsolute,
					std::abs(static_cast<double>(actual[vertex][channel]) - repeatabilityBaseline[vertex][channel]));
			}
		}
		const bool repeatable = repeatabilityMaxAbsolute == 0.0;
		const bool passed = goldenMaxAbsolute <= 1.0e-4 && goldenMeanAbsolute <= 1.0e-5 && healthFailures == 0 && repeatable;

		std::ofstream report("gnn_verify.json");
		report << std::setprecision(10) << "{\n  \"golden_max_abs\": " << goldenMaxAbsolute << ",\n  \"golden_mean_abs\": " << goldenMeanAbsolute
			<< ",\n  \"max_limit\": 0.0001,\n  \"mean_limit\": 0.00001,\n  \"stability_frames\": 1200,"
			<< "\n  \"health_failures\": " << healthFailures << ",\n  \"reset_replay_max_abs\": " << repeatabilityMaxAbsolute
			<< ",\n  \"passed\": " << (passed ? "true" : "false") << "\n}\n";
		report.close();
		std::cout << std::setprecision(10) << "Vulkan verification max_abs=" << goldenMaxAbsolute << " mean_abs=" << goldenMeanAbsolute
			<< " health_failures=" << healthFailures << " reset_replay_max_abs=" << repeatabilityMaxAbsolute << "\n";
		verificationDone = true;
		if (!passed) {
			vks::tools::exitFatal("Vulkan GNN verification failed; see gnn_verify.json", -1);
		}
	}

	void render() override
	{
		if (!prepared) return;
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &compute.fences[currentBuffer], VK_TRUE, UINT64_MAX));
		collectTimestamps(currentBuffer);
		VK_CHECK_RESULT(vkResetFences(device, 1, &compute.fences[currentBuffer]));
		updateComputeUniform();
		buildComputeCommandBuffer();
		// Reset is a transfer operation, so the semaphore wait must cover both the
		// optional reset copy and the following shader reads/writes.
		VkPipelineStageFlags computeWaitStage = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		VkSubmitInfo computeSubmit = vks::initializers::submitInfo();
		computeSubmit.waitSemaphoreCount = 1;
		computeSubmit.pWaitSemaphores = &compute.semaphores[(currentBuffer + maxConcurrentFrames - 1) % maxConcurrentFrames].ready;
		computeSubmit.pWaitDstStageMask = &computeWaitStage;
		computeSubmit.signalSemaphoreCount = 1;
		computeSubmit.pSignalSemaphores = &compute.semaphores[currentBuffer].complete;
		computeSubmit.commandBufferCount = 1;
		computeSubmit.pCommandBuffers = &compute.commandBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmit, compute.fences[currentBuffer]));
		verifyAcceleration();

		VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[currentBuffer], VK_TRUE, UINT64_MAX));
		VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentBuffer]));
		VulkanExampleBase::prepareFrame(false);
		updateGraphicsUniform();
		buildGraphicsCommandBuffer();
		VkPipelineStageFlags graphicsWaitStages[2] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT };
		VkSemaphore waitSemaphores[2] = { presentCompleteSemaphores[currentBuffer], compute.semaphores[currentBuffer].complete };
		VkSemaphore signalSemaphores[2] = { renderCompleteSemaphores[currentImageIndex], compute.semaphores[currentBuffer].ready };
		VkSubmitInfo graphicsSubmit = vks::initializers::submitInfo();
		graphicsSubmit.waitSemaphoreCount = 2;
		graphicsSubmit.pWaitSemaphores = waitSemaphores;
		graphicsSubmit.pWaitDstStageMask = graphicsWaitStages;
		graphicsSubmit.commandBufferCount = 1;
		graphicsSubmit.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		graphicsSubmit.signalSemaphoreCount = 2;
		graphicsSubmit.pSignalSemaphores = signalSemaphores;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &graphicsSubmit, waitFences[currentBuffer]));
		VulkanExampleBase::submitFrame(true);
	}

	static double percentile(std::vector<double> values, double fraction)
	{
		if (values.empty()) return 0.0;
		std::sort(values.begin(), values.end());
		const double lastIndex = static_cast<double>(values.size() - 1);
		const size_t index = static_cast<size_t>(std::min(lastIndex, std::ceil(fraction * static_cast<double>(values.size())) - 1.0));
		return values[index];
	}

	void writeBenchmarkCsv()
	{
		if (timingSamples.empty()) return;
		if (benchmarkOutput.has_parent_path()) std::filesystem::create_directories(benchmarkOutput.parent_path());
		std::vector<double> layer0, layer1, total;
		for (const auto& sample : timingSamples) {
			layer0.push_back(sample.layer0Ms);
			layer1.push_back(sample.layer1Ms);
			total.push_back(sample.totalMs);
		}
		std::ofstream stream(benchmarkOutput);
		std::ostringstream driver;
		if (deviceProperties.vendorID == 0x10de) {
			driver << ((deviceProperties.driverVersion >> 22) & 0x3ff) << '.' << ((deviceProperties.driverVersion >> 14) & 0xff)
				<< '.' << ((deviceProperties.driverVersion >> 6) & 0xff) << '.' << (deviceProperties.driverVersion & 0x3f);
		} else {
			driver << VK_VERSION_MAJOR(deviceProperties.driverVersion) << '.' << VK_VERSION_MINOR(deviceProperties.driverVersion)
				<< '.' << VK_VERSION_PATCH(deviceProperties.driverVersion);
		}
		stream << "device,driver_version,driver_raw,grid,nodes,directed_edges,xpbd_constraints,samples,layer0_median_ms,layer1_integrate_xpbd_median_ms,total_median_ms,total_p95_ms\n";
		stream << '"' << deviceProperties.deviceName << '"' << ',' << driver.str() << ',' << deviceProperties.driverVersion << ',' << cloth.gridSize.x << ','
			<< initialParticles.size() << ',' << graphBuffers.edgeCount << ',' << xpbdBuffers.edgeCount << ',' << timingSamples.size() << ','
			<< std::fixed << std::setprecision(6) << percentile(layer0, 0.5) << ',' << percentile(layer1, 0.5) << ','
			<< percentile(total, 0.5) << ',' << percentile(total, 0.95) << '\n';
		std::cout << "Wrote " << benchmarkOutput << " with " << timingSamples.size() << " timestamp samples\n";
	}

	void keyPressed(uint32_t keyCode) override
	{
		// The upstream base does not define symbolic G/R key constants on Windows.
		if (keyCode == 0x47) solver = solver == GNN ? MassSpring : GNN;
		if (keyCode == 0x52) {
			resetRequested = true;
			sphereMotionTime = 0.0f;
		}
	}

	void OnUpdateUIOverlay(vks::UIOverlay* overlay) override
	{
		if (!overlay->header("GNN cloth")) return;
		if (overlay->comboBox("Solver", &solver, { "Mass spring", "GNN 10-16-3" })) resetRequested = true;
		overlay->checkBox("Paused", &paused);
		overlay->checkBox("Wind", &simulateWind);
		if (overlay->checkBox("Moving sphere", &animateSphere)) sphereMotionTime = 0.0f;
		if (overlay->button("Reset")) {
			resetRequested = true;
			sphereMotionTime = 0.0f;
		}
		if (solver == GNN) {
			overlay->sliderInt("XPBD iterations", &xpbdIterations, 0, 16);
			overlay->sliderFloat("Stretch compliance (x1e-6)", &compute.uniformData.stretchComplianceMicro, 0.0f, 100.0f);
			overlay->sliderFloat("Bend compliance (x1e-6)", &compute.uniformData.bendComplianceMicro, 0.0f, 50000.0f);
			overlay->sliderFloat("XPBD velocity damping", &compute.uniformData.xpbdVelocityDamping, 0.0f, 5.0f);
		}
		overlay->text("Grid: %u x %u", cloth.gridSize.x, cloth.gridSize.y);
		overlay->text("Pinned: complete top edge (%u)", cloth.gridSize.x);
		overlay->text("Nodes: %u", compute.uniformData.vertexCount);
		overlay->text("Directed edges: %u", graphBuffers.edgeCount);
		overlay->text("XPBD constraints: %u (%u colors)", xpbdBuffers.edgeCount, xpbdColorCount);
		overlay->text("Layer 0: %.4f ms", lastTiming.layer0Ms);
		overlay->text("Layer 1 + integrate + XPBD: %.4f ms", lastTiming.layer1Ms);
		overlay->text("GNN total: %.4f ms", lastTiming.totalMs);
		overlay->text("G: toggle solver, R: reset, P: pause");
	}
};

VULKAN_EXAMPLE_MAIN()
