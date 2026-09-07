// Cloth Studio: independent animation, inference and dynamic cloth instances.
#include "vulkanexamplebase.h"
#include "VulkanTexture.h"
#include "demo_session.h"
#include "demo_gpu_validation.h"
#include "demo_gnn.h"
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>
#include <fstream>
#include <numeric>
#include <limits>
#include <set>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace d=mlcloth::demo;
class VulkanExample final:public VulkanExampleBase {
public:
    d::DemoSession session;
    std::array<d::Presentation,3> presentation;
    struct Vertex {glm::vec4 position,normal,uv;};
    struct Scene {glm::mat4 viewProjection,lightProjection;glm::vec4 eye,lightDirection;};
    struct Draw {glm::vec4 offset,color;uint32_t kind,actor,influences,count,shadow,material,boneCount,gray;};
    static_assert(sizeof(Draw)==64);
    struct Frame {vks::Buffer scene,palette,cloth,normals;VkDescriptorSet draw{},normal{};VkQueryPool queries{};bool submitted{},hasGnn{};std::array<bool,2> hasStages{},hasSelfStages{};};
    std::array<Frame,maxConcurrentFrames> frame;
    vks::Buffer bodyVertices,bodyIds,bodyWeights,bodyIndices,clothIndices,groundIndices,csrOffsets,csrTriangles;
    uint32_t bodyFirst[2]{},bodyCount[2]{};
    std::array<vks::Texture2D,2> textures{};
    VkDescriptorSetLayout drawSetLayout{},normalSetLayout{};
    VkPipelineLayout drawLayout{},normalLayout{};
    VkPipeline surfacePipeline{},shadowPipeline{},normalsPipeline{};
    VkDescriptorPool pool{};
    VkImage shadowImage{};VkDeviceMemory shadowMemory{};VkImageView shadowView{};VkSampler shadowSampler{};
    VkRenderPass shadowPass{};VkFramebuffer shadowFrame{};
    Scene scene{};glm::vec3 target{0,1,0};float orbitYaw{180},orbitPitch{8},distance{5.8f},autoRadius{5.8f};
    struct BoneBounds {uint32_t bone;glm::vec3 lo,hi;};std::vector<BoneBounds> bodyBounds;
    bool uiHidden{},capture{},pendingReset{},pendingSeek{},pendingSync{},syncValue{};
    int selectedActor{},pendingClip{-1},pendingActor{-1},pendingThreads{};double seekTime{};
    uint32_t maxFrames{},rendered{};double gpuMs{},cpuMs{},graphicsMs{};std::vector<double> cpuTimes,gpuTimes,frameTimes;
    std::array<std::array<double,7>,2> stepStages{};
    d::Json contactDiagnostics;
    std::vector<std::pair<uint32_t,uint32_t>> collisionEdges;
    std::filesystem::path manifest;std::string operation;
    std::filesystem::path screenshotPath;vks::Buffer screenshotBuffer;
    std::array<d::GpuPhysics,2> gpuPhysics;
    std::unique_ptr<d::GpuGnn> gpuGnn;
    int pendingHybrid{-1};double gnnMs{};std::vector<double> gnnTimes;
    std::filesystem::path gnnModelPath(int requested=-1)const{
        const int mode=requested<0?session.settings.hybridAlgorithm:requested;
        if(mode==2){const auto override=argument("--temporal-model");if(!override.empty())return override;
            return session.assets.directory/session.assets.provenance.value("temporal_gnn_model",std::string("../temporal_v4/temporal-init.vthood"));}
        return session.assets.directory/session.assets.provenance.value("gnn_model",std::string("../../../vulkan-gnn-poc/.work/hood_data/student32x12_r1.vhood"));}
    void ensureGnn(int requested=-1){const int mode=requested<0?session.settings.hybridAlgorithm:requested;if(gpuGnn&&gpuGnn->temporal()==(mode==2))return;
        auto candidate=std::make_unique<d::GpuGnn>();candidate->build(vulkanDevice,queue,pipelineCache,getShadersPath()+"mlclothcpu",gnnModelPath(mode),session.physicsMesh(),gpuPhysics[1],mode==2,
            uint32_t(session.assets.collision.positions.size()+session.assets.capsules.size()*40));candidate->captureValidation=argument("--dump-gnn","0")=="1";gpuGnn=std::move(candidate);}
    void recordGnn(VkCommandBuffer cmd,const d::PhysicsStep& step,float strength){
        if(gpuGnn->temporal())gpuGnn->record(cmd,step.gnnPins,d::sampleBodySurface(step.gnnCurrent,step.gnnFuture,step.gnnTime),strength,session.settings.gnnTrust,step.config.gravity);
        else gpuGnn->record(cmd,step.gnnPins,step.gnnCurrent,step.gnnFuture,strength,session.settings.gnnTrust,step.config.gravity);}

    std::array<uint64_t,2> gpuGeneration{};
    bool discardElapsed{true};
    int pendingBackend{-1};
    bool pendingResetAll{};

    std::string argument(const char* key,const std::string& fallback="")const{for(size_t i=0;i+1<args.size();++i)if(std::string(args[i])==key)return args[i+1];return fallback;}
    VulkanExample(){
        title="Cloth Studio";settings.overlay=true;width=1920;height=1080;
        apiVersion=VK_API_VERSION_1_1;
        camera.type=Camera::lookat;camera.setPosition({0,0,-8});
        manifest=argument("--demo-manifest");
        if(manifest.empty()){
            wchar_t path[MAX_PATH];GetModuleFileNameW(nullptr,path,MAX_PATH);
            auto location=std::filesystem::path(path).parent_path();
            for(int i=0;i<8;++i){auto candidate=location/".work/demo/demo.json";if(std::filesystem::exists(candidate)){manifest=candidate;break;}
                candidate=location/"demo/demo.json";if(std::filesystem::exists(candidate)){manifest=candidate;break;}location=location.parent_path();}
        }
        maxFrames=static_cast<uint32_t>(std::stoul(argument("--frames","0")));
        capture=argument("--capture-metrics","0")=="1";
        screenshotPath=argument("--screenshot");
    }
    bool useStandaloneUI()const override{return true;}
    ~VulkanExample(){
        if(device)vkDeviceWaitIdle(device);
        if(!maxFrames&&session.assets.animations.size())try{session.saveSettings();}catch(...){}
        if(device){
            screenshotBuffer.destroy();
            for(auto& f:frame){f.scene.destroy();f.palette.destroy();f.cloth.destroy();f.normals.destroy();if(f.queries)vkDestroyQueryPool(device,f.queries,nullptr);}
            for(auto* b:{&bodyVertices,&bodyIds,&bodyWeights,&bodyIndices,&clothIndices,&groundIndices,&csrOffsets,&csrTriangles})b->destroy();
            for(auto& t:textures)if(t.image)t.destroy();
            for(auto p:{surfacePipeline,shadowPipeline,normalsPipeline})if(p)vkDestroyPipeline(device,p,nullptr);
            if(pool)vkDestroyDescriptorPool(device,pool,nullptr);
            for(auto p:{drawLayout,normalLayout})if(p)vkDestroyPipelineLayout(device,p,nullptr);
            for(auto p:{drawSetLayout,normalSetLayout})if(p)vkDestroyDescriptorSetLayout(device,p,nullptr);
            if(shadowFrame)vkDestroyFramebuffer(device,shadowFrame,nullptr);if(shadowPass)vkDestroyRenderPass(device,shadowPass,nullptr);
            if(shadowSampler)vkDestroySampler(device,shadowSampler,nullptr);if(shadowView)vkDestroyImageView(device,shadowView,nullptr);
            if(shadowImage)vkDestroyImage(device,shadowImage,nullptr);if(shadowMemory)vkFreeMemory(device,shadowMemory,nullptr);
        }
    }
    void buffer(vks::Buffer& out,VkDeviceSize size,VkBufferUsageFlags usage,const void* initial=nullptr,bool host=false){
        auto flags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if(host){VK_CHECK_RESULT(vulkanDevice->createBuffer(usage,flags,&out,size,const_cast<void*>(initial)));VK_CHECK_RESULT(out.map());return;}
        vks::Buffer staging;VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,flags,&staging,size,const_cast<void*>(initial)));
        VK_CHECK_RESULT(vulkanDevice->createBuffer(usage|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,&out,size));
        auto cmd=vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);VkBufferCopy copy{0,0,size};vkCmdCopyBuffer(cmd,staging.buffer,out.buffer,1,&copy);
        vulkanDevice->flushCommandBuffer(cmd,queue,true);staging.destroy();
    }
    template<class T>void upload(vks::Buffer& out,const std::vector<T>& a,VkBufferUsageFlags usage){buffer(out,a.size()*sizeof(T),usage,a.data());}
    void createBuffers(){
        const auto& b=session.assets.body;const auto& c=session.assets.cloth;
        std::vector<glm::vec3> low(b.inverseBind.size(),glm::vec3(1e30f)),high(b.inverseBind.size(),glm::vec3(-1e30f));
        for(size_t vertex=0;vertex<b.positions.size();++vertex)for(uint32_t k=0;k<b.influences;++k){auto slot=vertex*b.influences+k;if(b.weights[slot]<=0)continue;
            auto bone=b.boneIds[slot];auto local=glm::vec3(b.inverseBind[bone]*glm::vec4(b.positions[vertex],1));low[bone]=glm::min(low[bone],local);high[bone]=glm::max(high[bone],local);}
        for(uint32_t bone=0;bone<low.size();++bone)if(low[bone].x<=high[bone].x)bodyBounds.push_back({bone,low[bone],high[bone]});
        std::set<std::pair<uint32_t,uint32_t>> edges;const auto& triangles=session.assets.collision.triangles;
        for(size_t t=0;t<triangles.size();t+=3)for(int k=0;k<3;++k)edges.insert(std::minmax(triangles[t+k],triangles[t+(k+1)%3]));
        collisionEdges.assign(edges.begin(),edges.end());
        std::vector<Vertex> vertices;for(size_t i=0;i<b.positions.size();++i)vertices.push_back({glm::vec4(b.positions[i],1),glm::vec4(b.normals[i],0),glm::vec4(b.uv[i],0,0)});
        upload(bodyVertices,vertices,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);upload(bodyIds,b.boneIds,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);upload(bodyWeights,b.weights,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::vector<uint32_t> indices;for(int m=0;m<2;++m){bodyFirst[m]=static_cast<uint32_t>(indices.size());for(size_t i=0;i<b.material.size();++i)if(b.material[i]==m)indices.insert(indices.end(),b.triangles.begin()+i*3,b.triangles.begin()+i*3+3);bodyCount[m]=static_cast<uint32_t>(indices.size())-bodyFirst[m];}
        upload(bodyIndices,indices,VK_BUFFER_USAGE_INDEX_BUFFER_BIT);upload(clothIndices,c.triangles,VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        upload(groundIndices,std::vector<uint32_t>{0,2,1,0,3,2},VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        std::vector<std::vector<uint32_t>> adjacent(c.rest.size());for(uint32_t i=0;i<c.triangles.size();++i)adjacent[c.triangles[i]].push_back(i/3);
        std::vector<uint32_t> offsets{0},incident;for(auto& a:adjacent){incident.insert(incident.end(),a.begin(),a.end());offsets.push_back(static_cast<uint32_t>(incident.size()));}
        upload(csrOffsets,offsets,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);upload(csrTriangles,incident,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        for(auto& f:frame){buffer(f.scene,sizeof(Scene),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,nullptr,true);
            buffer(f.palette,3*b.inverseBind.size()*3*sizeof(glm::vec4),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,nullptr,true);
            buffer(f.cloth,6*c.rest.size()*sizeof(glm::vec4),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,nullptr,true);
            buffer(f.normals,3*c.rest.size()*sizeof(glm::vec4),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};query.queryType=VK_QUERY_TYPE_TIMESTAMP;query.queryCount=21;VK_CHECK_RESULT(vkCreateQueryPool(device,&query,nullptr,&f.queries));}
        for(size_t i=0;i<textures.size();++i){int w{},h{},channels{};auto* pixels=stbi_load(session.assets.textures.at(i).string().c_str(),&w,&h,&channels,4);
            if(!pixels)throw std::runtime_error("Cannot load character texture");textures[i].fromBuffer(pixels,VkDeviceSize(w)*h*4,VK_FORMAT_R8G8B8A8_SRGB,w,h,vulkanDevice,queue);stbi_image_free(pixels);}
        if(!screenshotPath.empty())buffer(screenshotBuffer,VkDeviceSize(width)*height*4,VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,true);
    }
    void createShadow(){
        VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};image.imageType=VK_IMAGE_TYPE_2D;image.format=VK_FORMAT_D32_SFLOAT;image.extent={2048,2048,1};image.mipLevels=1;image.arrayLayers=1;image.samples=VK_SAMPLE_COUNT_1_BIT;image.tiling=VK_IMAGE_TILING_OPTIMAL;image.usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
        VK_CHECK_RESULT(vkCreateImage(device,&image,nullptr,&shadowImage));VkMemoryRequirements requirements;vkGetImageMemoryRequirements(device,shadowImage,&requirements);
        VkMemoryAllocateInfo memory{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};memory.allocationSize=requirements.size;memory.memoryTypeIndex=vulkanDevice->getMemoryType(requirements.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device,&memory,nullptr,&shadowMemory));VK_CHECK_RESULT(vkBindImageMemory(device,shadowImage,shadowMemory,0));
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};view.image=shadowImage;view.viewType=VK_IMAGE_VIEW_TYPE_2D;view.format=image.format;view.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};VK_CHECK_RESULT(vkCreateImageView(device,&view,nullptr,&shadowView));
        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};sampler.magFilter=VK_FILTER_LINEAR;sampler.minFilter=VK_FILTER_LINEAR;sampler.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;sampler.borderColor=VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;sampler.compareEnable=VK_TRUE;sampler.compareOp=VK_COMPARE_OP_LESS_OR_EQUAL;sampler.maxLod=1;VK_CHECK_RESULT(vkCreateSampler(device,&sampler,nullptr,&shadowSampler));
        VkAttachmentDescription attachment{};attachment.format=image.format;attachment.samples=VK_SAMPLE_COUNT_1_BIT;attachment.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;attachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;attachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;attachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;attachment.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;attachment.finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkAttachmentReference depth{0,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};VkSubpassDescription sub{};sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sub.pDepthStencilAttachment=&depth;
        std::array<VkSubpassDependency,2> deps{};deps[0].srcSubpass=VK_SUBPASS_EXTERNAL;deps[0].dstSubpass=0;deps[0].srcStageMask=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;deps[0].dstStageMask=VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;deps[0].srcAccessMask=VK_ACCESS_SHADER_READ_BIT;deps[0].dstAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;deps[0].dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;
        deps[1].srcSubpass=0;deps[1].dstSubpass=VK_SUBPASS_EXTERNAL;deps[1].srcStageMask=VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;deps[1].dstStageMask=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;deps[1].srcAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;deps[1].dstAccessMask=VK_ACCESS_SHADER_READ_BIT;deps[1].dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT;
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};rp.attachmentCount=1;rp.pAttachments=&attachment;rp.subpassCount=1;rp.pSubpasses=&sub;rp.dependencyCount=2;rp.pDependencies=deps.data();VK_CHECK_RESULT(vkCreateRenderPass(device,&rp,nullptr,&shadowPass));
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};fb.renderPass=shadowPass;fb.attachmentCount=1;fb.pAttachments=&shadowView;fb.width=fb.height=2048;fb.layers=1;VK_CHECK_RESULT(vkCreateFramebuffer(device,&fb,nullptr,&shadowFrame));
    }
    void createDescriptors(){
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        auto binding=[&](uint32_t id,VkDescriptorType type,VkShaderStageFlags stages){bindings.push_back(vks::initializers::descriptorSetLayoutBinding(type,stages,id));};
        binding(0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT);
        for(uint32_t i=1;i<=6;++i)binding(i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_SHADER_STAGE_VERTEX_BIT);
        binding(8,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,VK_SHADER_STAGE_FRAGMENT_BIT);binding(9,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,VK_SHADER_STAGE_FRAGMENT_BIT);binding(10,VK_DESCRIPTOR_TYPE_SAMPLER,VK_SHADER_STAGE_FRAGMENT_BIT);binding(11,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,VK_SHADER_STAGE_FRAGMENT_BIT);binding(12,VK_DESCRIPTOR_TYPE_SAMPLER,VK_SHADER_STAGE_FRAGMENT_BIT);
        auto layout=vks::initializers::descriptorSetLayoutCreateInfo(bindings);VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device,&layout,nullptr,&drawSetLayout));
        bindings.clear();for(uint32_t i=0;i<5;++i)binding(i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_SHADER_STAGE_COMPUTE_BIT);layout=vks::initializers::descriptorSetLayoutCreateInfo(bindings);VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device,&layout,nullptr,&normalSetLayout));
        std::vector<VkDescriptorPoolSize> sizes={{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,8},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,64},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,16},{VK_DESCRIPTOR_TYPE_SAMPLER,16}};
        auto info=vks::initializers::descriptorPoolCreateInfo(sizes,16);VK_CHECK_RESULT(vkCreateDescriptorPool(device,&info,nullptr,&pool));
        for(auto& f:frame){auto allocation=vks::initializers::descriptorSetAllocateInfo(pool,&drawSetLayout,1);VK_CHECK_RESULT(vkAllocateDescriptorSets(device,&allocation,&f.draw));allocation.pSetLayouts=&normalSetLayout;VK_CHECK_RESULT(vkAllocateDescriptorSets(device,&allocation,&f.normal));
            std::vector<VkWriteDescriptorSet> writes;
            writes.push_back(vks::initializers::writeDescriptorSet(f.draw,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,0,&f.scene.descriptor));
            std::array<vks::Buffer*,6> buffers={&bodyVertices,&bodyIds,&bodyWeights,&f.palette,&f.cloth,&f.normals};for(uint32_t i=0;i<6;++i)writes.push_back(vks::initializers::writeDescriptorSet(f.draw,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,i+1,&buffers[i]->descriptor));
            VkDescriptorImageInfo bodyTex{VK_NULL_HANDLE,textures[0].view,textures[0].imageLayout},headTex{VK_NULL_HANDLE,textures[1].view,textures[1].imageLayout},colorSampler{textures[0].sampler,VK_NULL_HANDLE,VK_IMAGE_LAYOUT_UNDEFINED};
            VkDescriptorImageInfo shadowTex{VK_NULL_HANDLE,shadowView,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},shadowSampling{shadowSampler,VK_NULL_HANDLE,VK_IMAGE_LAYOUT_UNDEFINED};
            for(auto pair:std::vector<std::pair<uint32_t,VkDescriptorImageInfo*>>{{8,&bodyTex},{9,&headTex},{10,&colorSampler},{11,&shadowTex},{12,&shadowSampling}})
                writes.push_back(vks::initializers::writeDescriptorSet(f.draw,pair.first==10||pair.first==12?VK_DESCRIPTOR_TYPE_SAMPLER:VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,pair.first,pair.second));
            std::array<vks::Buffer*,5> normalBuffers={&f.cloth,&f.normals,&clothIndices,&csrOffsets,&csrTriangles};for(uint32_t i=0;i<5;++i)writes.push_back(vks::initializers::writeDescriptorSet(f.normal,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,i,&normalBuffers[i]->descriptor));
            vkUpdateDescriptorSets(device,static_cast<uint32_t>(writes.size()),writes.data(),0,nullptr);
        }
    }
    void createPipelines(){
        auto push=vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,sizeof(Draw),0);
        auto layout=vks::initializers::pipelineLayoutCreateInfo(&drawSetLayout,1);layout.pushConstantRangeCount=1;layout.pPushConstantRanges=&push;VK_CHECK_RESULT(vkCreatePipelineLayout(device,&layout,nullptr,&drawLayout));
        push=vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT,16,0);layout.pSetLayouts=&normalSetLayout;layout.pPushConstantRanges=&push;VK_CHECK_RESULT(vkCreatePipelineLayout(device,&layout,nullptr,&normalLayout));
        auto compute=vks::initializers::computePipelineCreateInfo(normalLayout);compute.stage=loadShader(getShadersPath()+"mlclothcpu/demo_normals.comp.spv",VK_SHADER_STAGE_COMPUTE_BIT);VK_CHECK_RESULT(vkCreateComputePipelines(device,pipelineCache,1,&compute,nullptr,&normalsPipeline));
        auto input=vks::initializers::pipelineVertexInputStateCreateInfo();auto assembly=vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,0,VK_FALSE);
        auto raster=vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL,VK_CULL_MODE_NONE,VK_FRONT_FACE_COUNTER_CLOCKWISE,0);
        auto blendAttachment=vks::initializers::pipelineColorBlendAttachmentState(0xf,VK_FALSE);auto blend=vks::initializers::pipelineColorBlendStateCreateInfo(1,&blendAttachment);
        auto depth=vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE,VK_TRUE,VK_COMPARE_OP_LESS_OR_EQUAL);auto viewport=vks::initializers::pipelineViewportStateCreateInfo(1,1,0);auto samples=vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT,0);
        std::vector<VkDynamicState> states={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};auto dynamic=vks::initializers::pipelineDynamicStateCreateInfo(states);
        std::array<VkPipelineShaderStageCreateInfo,2> stages={loadShader(getShadersPath()+"mlclothcpu/demo_scene.vert.spv",VK_SHADER_STAGE_VERTEX_BIT),loadShader(getShadersPath()+"mlclothcpu/demo_scene.frag.spv",VK_SHADER_STAGE_FRAGMENT_BIT)};
        auto pipeline=vks::initializers::pipelineCreateInfo(drawLayout,renderPass,0);pipeline.pVertexInputState=&input;pipeline.pInputAssemblyState=&assembly;pipeline.pRasterizationState=&raster;pipeline.pColorBlendState=&blend;pipeline.pDepthStencilState=&depth;pipeline.pViewportState=&viewport;pipeline.pMultisampleState=&samples;pipeline.pDynamicState=&dynamic;pipeline.stageCount=2;pipeline.pStages=stages.data();
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device,pipelineCache,1,&pipeline,nullptr,&surfacePipeline));
        pipeline.renderPass=shadowPass;pipeline.stageCount=1;blend.attachmentCount=0;raster.depthBiasEnable=VK_TRUE;raster.depthBiasConstantFactor=1.25f;raster.depthBiasSlopeFactor=1.75f;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device,pipelineCache,1,&pipeline,nullptr,&shadowPipeline));
    }
    void prepare()override{
        try{session.renderOnly=argument("--render-only","0")=="1";session.gpuMode=!session.renderOnly&&argument("--cpu-reference","0")!="1";session.load(manifest);if(session.renderOnly)session.status="Renderer validation only - physics disabled";
            wchar_t windows[MAX_PATH];if(GetWindowsDirectoryW(windows,MAX_PATH)){auto font=std::filesystem::path(windows)/"Fonts/segoeui.ttf";
                if(std::filesystem::exists(font))ImGui::GetIO().Fonts->AddFontFromFileTTF(font.string().c_str(),18.f);}
            std::ofstream log(session.assets.directory/"startup.log",std::ios::app);log<<"Prepare Vulkan base"<<std::endl;VulkanExampleBase::prepare();
            log<<"Create buffers"<<std::endl;createBuffers();log<<"Create shadow"<<std::endl;createShadow();log<<"Create descriptors"<<std::endl;createDescriptors();log<<"Create pipelines"<<std::endl;createPipelines();log<<"Ready"<<std::endl;
            if(!session.renderOnly)for(auto& solver:gpuPhysics)solver.build(vulkanDevice,queue,pipelineCache,getShadersPath()+"mlclothcpu/demo_physics.comp.spv",session.physicsMesh());
            if(argument("--hybrid-algorithm")!=""){auto algorithm=argument("--hybrid-algorithm");if(algorithm!="gnn"&&algorithm!="mlcloth"&&algorithm!="temporal")throw std::runtime_error("Unknown hybrid algorithm");session.settings.hybridAlgorithm=algorithm=="gnn"?1:algorithm=="temporal"?2:0;session.reset();}
            if(session.settings.hybridAlgorithm!=0){if(!session.gpuMode)throw std::runtime_error("GNN hybrid requires Vulkan compute");try{ensureGnn();}catch(const std::exception& e){if(argument("--hybrid-algorithm")=="gnn"||argument("--hybrid-algorithm")=="temporal")throw;session.settings.hybridAlgorithm=0;session.reset();session.status=std::string("GNN unavailable: ")+e.what();}}
            if(argument("--self-collision")!="")session.settings.physics.enableSelfCollision=argument("--self-collision")=="1";
            if(argument("--body-collision")!="")session.settings.physics.enableCollision=argument("--body-collision")=="1";
            if(argument("--bend-compliance")!="")session.settings.physics.bendCompliance=std::stof(argument("--bend-compliance"));
            if(argument("--tethers")!="")session.settings.physics.enableTethers=argument("--tethers")=="1";
            if(argument("--contact-guide")!="")session.settings.physics.contactAwareGuide=argument("--contact-guide")=="1";
            if(argument("--playback-speed")!=""){float speed=std::stof(argument("--playback-speed"));if(!std::isfinite(speed)||speed<.25f||speed>2)throw std::runtime_error("Invalid playback speed");session.settings.speed=speed;}
            if(argument("--collision-thickness")!=""){float thickness=std::stof(argument("--collision-thickness"));
                if(!std::isfinite(thickness)||thickness<.001f||thickness>.01f)throw std::runtime_error("Collision thickness must be between 1 and 10 mm");session.settings.physics.thickness=thickness;}
            if(argument("--animation")!=""){
                const auto requested=argument("--animation");int index=-1;
                for(size_t i=0;i<session.assets.provenance.at("clips").size();++i)if(session.assets.provenance.at("clips")[i].at("id")==requested)index=static_cast<int>(i);
                if(index<0)throw std::runtime_error("Unknown validation animation: "+requested);for(auto& actor:session.actors)actor.animation=index;session.reset();
            }
            session.settings.showCollision=argument("--show-collision","0")=="1";
            resetGpu();frameTimer=1.f/60;
            if(argument("--validate-gnn","0")=="1")validateGnn();
            if(session.gpuMode&&argument("--validate-asset","0")=="1")validateAssetStep();
            if(argument("--validate-gpu","0")=="1"){
                auto report=d::validateGpuPhysics(vulkanDevice,queue,pipelineCache,getShadersPath()+"mlclothcpu/demo_physics.comp.spv");
                std::ofstream(session.assets.directory/"gpu-validation.json")<<report.dump(2);
            }
            auto& style=ImGui::GetStyle();style.WindowRounding=8;style.FrameRounding=5;style.WindowPadding=ImVec2(16,12);style.ItemSpacing=ImVec2(10,8);
            ImGui::GetIO().FontGlobalScale=1.f;orbitYaw=std::array<float,4>{180,90,0,145}[session.settings.view];
            style.Colors[ImGuiCol_WindowBg]=ImVec4(.055f,.067f,.085f,.94f);
            style.Colors[ImGuiCol_FrameBg]=ImVec4(.13f,.16f,.2f,1);
            style.Colors[ImGuiCol_FrameBgHovered]=ImVec4(.2f,.26f,.33f,1);
            style.Colors[ImGuiCol_FrameBgActive]=ImVec4(.23f,.34f,.45f,1);
            style.Colors[ImGuiCol_Button]=ImVec4(.15f,.2f,.26f,1);
            style.Colors[ImGuiCol_ButtonHovered]=ImVec4(.23f,.34f,.45f,1);
            style.Colors[ImGuiCol_ButtonActive]=ImVec4(.28f,.44f,.6f,1);
            style.Colors[ImGuiCol_Header]=ImVec4(.17f,.24f,.32f,1);
            style.Colors[ImGuiCol_HeaderHovered]=ImVec4(.24f,.34f,.44f,1);
            style.Colors[ImGuiCol_SliderGrab]=ImVec4(.4f,.66f,.85f,1);
            style.Colors[ImGuiCol_SliderGrabActive]=ImVec4(.56f,.8f,1,1);
            style.Colors[ImGuiCol_CheckMark]=ImVec4(.45f,.75f,.94f,1);
            prepared=true;
        }catch(const std::exception& e){std::ofstream("cloth-demo-error.txt")<<e.what();vks::tools::exitFatal(e.what(),-1);}
    }
    void matrices(){
        // The union of transformed per-bone bounds contains the linearly skinned
        // body and follows jumping/root motion without a GPU mesh readback.
        glm::vec3 low(1e30f),high(-1e30f);
        for(int i=0;i<3;++i){if(session.settings.focus>=0&&session.settings.focus!=i)continue;const auto& actor=presentation[i];
            for(auto bound:bodyBounds)for(int k=0;k<8;++k){glm::vec3 corner(k&1?bound.hi.x:bound.lo.x,k&2?bound.hi.y:bound.lo.y,k&4?bound.hi.z:bound.lo.z);
                auto point=glm::vec3(actor.pose.world[bound.bone]*glm::vec4(corner,1))+actor.displayOffset;low=glm::min(low,point);high=glm::max(high,point);}}
        low-=glm::vec3(.12f);high+=glm::vec3(.12f);glm::vec3 wanted=(low+high)*.5f;
        if(session.settings.autoCamera)target=rendered?glm::mix(target,wanted,1-std::exp(-std::max(frameTimer,.001f)*6)):wanted;
        float yaw=glm::radians(orbitYaw),pitch=glm::radians(orbitPitch);auto outward=glm::vec3(std::sin(yaw)*std::cos(pitch),std::sin(pitch),std::cos(yaw)*std::cos(pitch));
        auto right=glm::normalize(glm::cross(glm::vec3(0,1,0),outward)),up=glm::cross(outward,right);float fit=1;
        const float tanVertical=std::tan(glm::radians(21.f)),tanHorizontal=tanVertical*float(width)/height;
        for(int k=0;k<8;++k){glm::vec3 corner(k&1?high.x:low.x,k&2?high.y:low.y,k&4?high.z:low.z);auto delta=corner-target;
            fit=std::max(fit,glm::dot(delta,outward)+std::max(std::abs(glm::dot(delta,right))/(tanHorizontal*.92f),std::abs(glm::dot(delta,up))/(tanVertical*(uiHidden?.9f:.66f))));}
        autoRadius=rendered?glm::mix(autoRadius,fit,1-std::exp(-std::max(frameTimer,.001f)*4)):fit;
        float radius=session.settings.autoCamera?autoRadius*distance/5.8f:distance*(session.settings.focus>=0?.5f:1);
        glm::vec3 eye=target+outward*radius;
        auto projection=glm::perspective(glm::radians(42.f),float(width)/height,.05f,200.f);projection[1][1]*=-1;
        scene.viewProjection=projection*glm::lookAt(eye,target,glm::vec3(0,1,0));scene.eye=glm::vec4(eye,1);scene.lightDirection=glm::vec4(.5f,-1,.35f,0);
        float extent=std::max(3.f,glm::length(high-low)*.6f);auto light=glm::ortho(-extent,extent,-extent,extent,.1f,extent*8);light[1][1]*=-1;
        scene.lightProjection=light*glm::lookAt(target-glm::normalize(glm::vec3(scene.lightDirection))*extent*4.f,target,glm::vec3(0,1,0));
    }
    void updateBuffers(Frame& f){
        for(int i=0;i<3;++i)presentation[i]=session.presentation(i);
        matrices();std::memcpy(f.scene.mapped,&scene,sizeof(scene));auto* palette=static_cast<glm::vec4*>(f.palette.mapped);auto* cloth=static_cast<glm::vec4*>(f.cloth.mapped);
        const auto& body=session.assets.body;for(int actor=0;actor<3;++actor){const auto& a=session.actors[actor];for(size_t b=0;b<body.inverseBind.size();++b){auto m=presentation[actor].pose.world[b]*body.inverseBind[b];for(int r=0;r<3;++r)palette[(actor*body.inverseBind.size()+b)*3+r]=glm::vec4(m[0][r],m[1][r],m[2][r],m[3][r]);}
            for(size_t v=0;v<a.cloth.size();++v){cloth[actor*a.cloth.size()+v]=glm::vec4(a.cloth[v],1);
                cloth[(actor+3)*a.cloth.size()+v]=glm::vec4(a.previousCloth.size()==a.cloth.size()?a.previousCloth[v]:a.cloth[v],1);}}
    }
    void drawScene(VkCommandBuffer cmd,Frame& f,bool shadow){
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,shadow?shadowPipeline:surfacePipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,drawLayout,0,1,&f.draw,0,nullptr);
        Draw draw{};draw.count=5294;draw.shadow=shadow;draw.influences=session.assets.body.influences;draw.boneCount=static_cast<uint32_t>(session.assets.body.inverseBind.size());draw.gray=session.settings.material!=0;
        for(int i=0;i<3;++i){if(session.settings.focus>=0&&session.settings.focus!=i)continue;draw.actor=i;draw.offset=glm::vec4(presentation[i].displayOffset,presentation[i].blend);draw.kind=0;draw.color=glm::vec4(1);
            vkCmdBindIndexBuffer(cmd,bodyIndices.buffer,0,VK_INDEX_TYPE_UINT32);for(uint32_t material=0;material<2;++material){draw.material=material;vkCmdPushConstants(cmd,drawLayout,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(draw),&draw);vkCmdDrawIndexed(cmd,bodyCount[material],1,bodyFirst[material],0,0);}
            draw.kind=1;draw.color=session.settings.material==2?glm::vec4(i==0?.15f:.75f,i==2?.75f:.32f,i==0?.85f:.25f,1):glm::vec4(.055f,.13f,.17f,1);
            if(session.settings.material==1)draw.color=glm::vec4(.65f,.65f,.65f,1);
            vkCmdPushConstants(cmd,drawLayout,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(draw),&draw);vkCmdBindIndexBuffer(cmd,clothIndices.buffer,0,VK_INDEX_TYPE_UINT32);vkCmdDrawIndexed(cmd,static_cast<uint32_t>(session.assets.cloth.triangles.size()),1,0,0,0);
        }
        if(!shadow){draw.kind=2;draw.offset=glm::vec4(target.x,0,target.z,0);draw.color=glm::vec4(1);vkCmdPushConstants(cmd,drawLayout,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(draw),&draw);vkCmdBindIndexBuffer(cmd,groundIndices.buffer,0,VK_INDEX_TYPE_UINT32);vkCmdDrawIndexed(cmd,6,1,0,0,0);}
    }
    void record(Frame& f){
        auto cmd=drawCmdBuffers[currentBuffer];auto begin=vks::initializers::commandBufferBeginInfo();VK_CHECK_RESULT(vkBeginCommandBuffer(cmd,&begin));vkCmdResetQueryPool(cmd,f.queries,0,21);vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,f.queries,0);f.hasStages={};f.hasGnn=false;
        if(session.gpuMode){
            for(const auto& step:session.physicsSteps)if(step.generation==session.actors[step.actor].generation){
                const auto index=step.actor-1;auto& solver=gpuPhysics[index];
                if(step.inferGnn){if(!f.hasGnn)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,f.queries,19);recordGnn(cmd,step,session.settings.gnnStrength);if(!f.hasGnn)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,f.queries,20);f.hasGnn=true;}
                solver.recordStep(cmd,step.dt,step.pins,step.actor==2&&!step.guide.empty()?&step.guide:nullptr,&step.collider,step.config,f.hasStages[index]?VK_NULL_HANDLE:f.queries,2+index*8,session.settings.synchronized&&step.actor==2?&gpuPhysics[0]:nullptr);
                if(!f.hasStages[index])f.hasSelfStages[index]=step.config.enableSelfCollision;
                if(step.checkpoint){solver.recordCheckpoint(cmd,step.checkpoint);if(index==1&&gpuGnn)gpuGnn->recordCheckpoint(cmd,step.checkpoint);}f.hasStages[index]=true;}
            session.physicsSteps.clear();
            VkMemoryBarrier transfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};transfer.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;transfer.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&transfer,0,nullptr,0,nullptr);
            for(uint32_t i=0;i<2;++i){VkBufferCopy copy{0,(i+1)*5294*sizeof(glm::vec4),5294*sizeof(glm::vec4)};vkCmdCopyBuffer(cmd,gpuPhysics[i].positionBuffer().buffer,f.cloth.buffer,1,&copy);
                copy.dstOffset=(i+4)*5294*sizeof(glm::vec4);vkCmdCopyBuffer(cmd,gpuPhysics[i].previousPositionBuffer().buffer,f.cloth.buffer,1,&copy);}
            transfer.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;transfer.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,0,1,&transfer,0,nullptr,0,nullptr);
        }
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,normalsPipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,normalLayout,0,1,&f.normal,0,nullptr);
        struct NormalPush {uint32_t vertices;float blend[3];} normalPush{5294,{presentation[0].blend,presentation[1].blend,presentation[2].blend}};
        static_assert(sizeof(NormalPush)==16);vkCmdPushConstants(cmd,normalLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,&normalPush);vkCmdDispatch(cmd,(5294*3+127)/128,1,1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,f.queries,18);
        VkClearValue clear{};clear.depthStencil={1,0};auto shadow=vks::initializers::renderPassBeginInfo();shadow.renderPass=shadowPass;shadow.framebuffer=shadowFrame;shadow.renderArea.extent={2048,2048};shadow.clearValueCount=1;shadow.pClearValues=&clear;vkCmdBeginRenderPass(cmd,&shadow,VK_SUBPASS_CONTENTS_INLINE);
        VkViewport viewport{0,0,2048,2048,0,1};VkRect2D scissor{{0,0},{2048,2048}};vkCmdSetViewport(cmd,0,1,&viewport);vkCmdSetScissor(cmd,0,1,&scissor);drawScene(cmd,f,true);vkCmdEndRenderPass(cmd);
        std::array<VkClearValue,2> values{};values[0].color={{.055f,.071f,.09f,1}};values[1].depthStencil={1,0};auto pass=vks::initializers::renderPassBeginInfo();pass.renderPass=renderPass;pass.framebuffer=frameBuffers[currentImageIndex];pass.renderArea.extent={width,height};pass.clearValueCount=2;pass.pClearValues=values.data();vkCmdBeginRenderPass(cmd,&pass,VK_SUBPASS_CONTENTS_INLINE);
        viewport={0,0,float(width),float(height),0,1};scissor.extent={width,height};vkCmdSetViewport(cmd,0,1,&viewport);vkCmdSetScissor(cmd,0,1,&scissor);drawScene(cmd,f,false);drawUI(cmd);vkCmdEndRenderPass(cmd);
        if(!screenshotPath.empty()&&maxFrames&&rendered+1==maxFrames){
            auto image=swapChain.images[currentImageIndex];VkImageMemoryBarrier imageBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};imageBarrier.image=image;imageBarrier.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};imageBarrier.srcQueueFamilyIndex=imageBarrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            imageBarrier.oldLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;imageBarrier.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;imageBarrier.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;imageBarrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&imageBarrier);
            VkBufferImageCopy copy{};copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.imageExtent={width,height,1};vkCmdCopyImageToBuffer(cmd,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,screenshotBuffer.buffer,1,&copy);
            imageBarrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;imageBarrier.newLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;imageBarrier.srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT;imageBarrier.dstAccessMask=0;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,nullptr,0,nullptr,1,&imageBarrier);
        }
        vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,f.queries,1);VK_CHECK_RESULT(vkEndCommandBuffer(cmd));
    }
    void operations(){
        if(pendingHybrid>=0){const int requested=pendingHybrid;pendingHybrid=-1;try{VK_CHECK_RESULT(vkDeviceWaitIdle(device));if(requested!=0)ensureGnn(requested);session.settings.hybridAlgorithm=requested;if(requested!=0)session.gpuMode=true;session.reset();discardElapsed=true;}catch(const std::exception& e){session.status=std::string("GNN switch failed; current scene retained: ")+e.what();}}
        if(pendingThreads||pendingSync||pendingClip>=0||pendingReset||pendingSeek||pendingBackend>=0)discardElapsed=true;
        if(pendingBackend>=0){VK_CHECK_RESULT(vkDeviceWaitIdle(device));session.gpuMode=pendingBackend==1;if(!session.gpuMode)session.settings.hybridAlgorithm=0;pendingBackend=-1;session.reset();}
        if(pendingThreads){session.settings.threads=pendingThreads;pendingThreads=0;pendingReset=true;pendingResetAll=true;}
        if(pendingSync){session.synchronize(syncValue);pendingSync=false;}
        if(pendingClip>=0){session.select(pendingActor,pendingClip);pendingClip=-1;}
        if(pendingReset){session.reset(session.settings.synchronized||pendingResetAll?-1:selectedActor);pendingReset=false;pendingResetAll=false;}
        if(pendingSeek){session.seek(session.settings.synchronized?-1:selectedActor,seekTime);pendingSeek=false;}
    }
    void resetGpu(){
        if(!session.gpuMode)return;
        bool changed=false;for(int i=0;i<2;++i)changed|=gpuGeneration[i]!=session.actors[i+1].generation;
        if(!changed)return;VK_CHECK_RESULT(vkDeviceWaitIdle(device));
        for(int i=0;i<2;++i){auto& a=session.actors[i+1];if(gpuGeneration[i]==a.generation)continue;
            std::vector<uint64_t> retained;for(const auto& entry:a.checkpoints)retained.push_back(entry.second.id);gpuPhysics[i].retainCheckpoints(retained);if(i==1&&gpuGnn)gpuGnn->retainCheckpoints(retained);
            if(a.restoreCheckpoint){gpuPhysics[i].restoreCheckpoint(a.restoreCheckpoint);if(i==1&&gpuGnn)gpuGnn->restoreCheckpoint(a.restoreCheckpoint);a.restoreCheckpoint=0;gpuGeneration[i]=a.generation;continue;}
            std::vector<d::Vec3> positions;for(auto v:a.cloth)positions.push_back({v.x,v.y,v.z});
            gpuPhysics[i].reset(positions);if(i==1&&gpuGnn)gpuGnn->resetHistory();
            auto cmd=vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);
            session.warmup(i+1,[&](const d::PhysicsStep& step){gpuPhysics[i].recordStep(cmd,step.dt,step.pins,nullptr,&step.collider,step.config);});
            gpuPhysics[i].recordCheckpoint(cmd,a.checkpoints.at(0).id);if(i==1&&gpuGnn)gpuGnn->recordCheckpoint(cmd,a.checkpoints.at(0).id);
            vulkanDevice->flushCommandBuffer(cmd,queue,true);gpuGeneration[i]=a.generation;
        }
    }
    void render()override{
        if(!prepared||(maxFrames&&rendered>=maxFrames))return;
        auto trace=[&](const char* text){if(rendered<10)std::ofstream(session.assets.directory/"startup.log",std::ios::app)<<rendered<<" "<<text<<std::endl;};
        trace("First render");
        auto started=std::chrono::steady_clock::now();auto& f=frame[currentBuffer];
        VK_CHECK_RESULT(vkWaitForFences(device,1,&waitFences[currentBuffer],VK_TRUE,UINT64_MAX));
        if(f.submitted){uint64_t time[2]{},graphicsStart{};if(vkGetQueryPoolResults(device,f.queries,0,2,sizeof(time),time,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT)==VK_SUCCESS){gpuMs=double(time[1]-time[0])*deviceProperties.limits.timestampPeriod/1e6;
            if(vkGetQueryPoolResults(device,f.queries,18,1,sizeof(graphicsStart),&graphicsStart,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT)==VK_SUCCESS)graphicsMs=double(time[1]-graphicsStart)*deviceProperties.limits.timestampPeriod/1e6;}}
        for(int actor=0;actor<2;++actor)if(f.submitted&&f.hasStages[actor]){uint64_t time[8]{};auto n=f.hasSelfStages[actor]?8:5;if(vkGetQueryPoolResults(device,f.queries,2+8*actor,n,n*sizeof(uint64_t),time,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT)==VK_SUCCESS){
            auto& stages=stepStages[actor];stages={};for(int i=0;i<4;++i)stages[i]=double(time[i+1]-time[i])*deviceProperties.limits.timestampPeriod/1e6;
            if(n==8){stages[4]=double(time[5]-time[2])*deviceProperties.limits.timestampPeriod/1e6;stages[5]=double(time[6]-time[5])*deviceProperties.limits.timestampPeriod/1e6;stages[6]=double(time[7]-time[6])*deviceProperties.limits.timestampPeriod/1e6;}}}
        if(f.submitted&&f.hasGnn){uint64_t time[2]{};if(vkGetQueryPoolResults(device,f.queries,19,2,sizeof(time),time,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT)==VK_SUCCESS){gnnMs=double(time[1]-time[0])*deviceProperties.limits.timestampPeriod/1e6;if(rendered>std::stoul(argument("--metric-warmup","120")))gnnTimes.push_back(gnnMs);}}
        trace("Fence ready");
        bool operationFrame=discardElapsed||session.seeking();discardElapsed=false;
        try{operations();const bool changed=discardElapsed,wasSeeking=session.seeking();resetGpu();operationFrame|=changed||wasSeeking;trace("Operations complete");
            const auto fixed=argument("--fixed-frame-dt");
            session.update(operationFrame?0.:fixed.empty()?frameTimer:std::stod(fixed));discardElapsed=changed||wasSeeking;
            trace("Session updated");updateBuffers(f);trace("Buffers updated");}catch(const std::exception& e){session.settings.paused=true;session.status=e.what();}
        VulkanExampleBase::prepareFrame(false);trace("Image acquired");VK_CHECK_RESULT(vkResetFences(device,1,&waitFences[currentBuffer]));record(f);trace("Commands recorded");
        auto submit=vks::initializers::submitInfo();VkPipelineStageFlags stage=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;submit.waitSemaphoreCount=1;submit.pWaitSemaphores=&presentCompleteSemaphores[currentBuffer];submit.pWaitDstStageMask=&stage;submit.commandBufferCount=1;submit.pCommandBuffers=&drawCmdBuffers[currentBuffer];submit.signalSemaphoreCount=1;submit.pSignalSemaphores=&renderCompleteSemaphores[currentImageIndex];VK_CHECK_RESULT(vkQueueSubmit(queue,1,&submit,waitFences[currentBuffer]));f.submitted=true;
        cpuMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        if(capture&&!operationFrame&&rendered>std::stoul(argument("--metric-warmup","120"))){cpuTimes.push_back(cpuMs);gpuTimes.push_back(gpuMs);frameTimes.push_back(frameTimer*1000);}
        if(!screenshotPath.empty()&&maxFrames&&rendered+1==maxFrames){
            VK_CHECK_RESULT(vkWaitForFences(device,1,&waitFences[currentBuffer],VK_TRUE,UINT64_MAX));
            std::ofstream image(screenshotPath,std::ios::binary);image<<"P6\n"<<width<<" "<<height<<"\n255\n";
            const auto* pixels=static_cast<const unsigned char*>(screenshotBuffer.mapped);const bool bgra=swapChain.colorFormat==VK_FORMAT_B8G8R8A8_UNORM||swapChain.colorFormat==VK_FORMAT_B8G8R8A8_SRGB;
            for(size_t i=0;i<size_t(width)*height;++i){char rgb[3]={char(pixels[i*4+(bgra?2:0)]),char(pixels[i*4+1]),char(pixels[i*4+(bgra?0:2)])};image.write(rgb,3);}
        }
        VulkanExampleBase::submitFrame(true);++rendered;if(capture&&rendered%60==0)writeMetrics();if(maxFrames&&rendered>=maxFrames){
            if(argument("--validate-presentation","0")=="1")validatePresentation(f);
            if(session.gpuMode)for(auto& solver:gpuPhysics)contactDiagnostics.push_back(solver.contactDiagnostics());
            if(gpuGnn&&argument("--dump-gnn","0")=="1"){VK_CHECK_RESULT(vkDeviceWaitIdle(device));gpuGnn->dumpValidation(session.assets.directory/"gnn-validation");}
            if(argument("--dump-physics","0")=="1")dumpPhysics();writeMetrics();PostQuitMessage(0);}
    }
    void validateGnn(){
        if(!gpuGnn||session.settings.hybridAlgorithm==0)throw std::runtime_error("GNN validation requires --hybrid-algorithm gnn");
        const int validationAlgorithm=session.settings.hybridAlgorithm;d::Json report;report["cases"]=d::Json::array();bool passed=true;
        auto require=[&](const char* name,double value,double limit){bool ok=std::isfinite(value)&&value<=limit;passed&=ok;report["cases"].push_back({{"name",name},{"error",value},{"limit",limit},{"passed",ok}});};
        auto difference=[](const std::vector<d::Vec3>& a,const std::vector<d::Vec3>& b){double error=0;for(size_t i=0;i<a.size();i++){double value=d::length(a[i]-b[i]);if(!std::isfinite(value))return std::numeric_limits<double>::infinity();error=std::max(error,value);}return error;};
        require("GNN actor owns no MLCloth stream",session.actors[2].inference?1:0,0);
        std::vector<d::PhysicsStep> steps;for(int i=0;i<24;i++){session.step(2,1.f/session.settings.physicsHz);steps.push_back(std::move(session.physicsSteps.back()));session.physicsSteps.pop_back();}
        const auto initial=session.actors[2].checkpoints.at(0).id;
        auto run=[&](size_t first,size_t last,float strength,bool pure){auto cmd=vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);
            for(size_t i=first;i<last;i++){auto& step=steps[i];if(step.inferGnn)recordGnn(cmd,step,strength);
                gpuPhysics[1].recordStep(cmd,step.dt,step.pins,nullptr,&step.collider,step.config);
                if(pure){auto config=step.config;config.gnnAcceleration=false;gpuPhysics[0].recordStep(cmd,step.dt,step.pins,nullptr,&step.collider,config);}}
            vulkanDevice->flushCommandBuffer(cmd,queue,true);};
        run(0,24,0,true);auto pure=gpuPhysics[0].readback();require("zero strength equals pure XPBD",difference(pure,gpuPhysics[1].readback()),1e-6);
        gpuPhysics[1].restoreCheckpoint(initial);gpuGnn->restoreCheckpoint(initial);run(0,5,1,false);
        const uint64_t partial=0x7fffffffffffffffull;auto cmd=vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);gpuPhysics[1].recordCheckpoint(cmd,partial);gpuGnn->recordCheckpoint(cmd,partial);vulkanDevice->flushCommandBuffer(cmd,queue,true);
        run(5,24,1,false);auto expected=gpuPhysics[1].readback();double contribution=difference(pure,expected);require("learned contribution is finite and nonzero",std::isfinite(contribution)&&contribution>1e-7?0:1,0);report["gnn_max_contribution_m"]=contribution;
        gpuPhysics[1].restoreCheckpoint(partial);gpuGnn->restoreCheckpoint(partial);run(5,24,1,false);require("checkpoint within held acceleration interval",difference(expected,gpuPhysics[1].readback()),0);
        gpuPhysics[1].restoreCheckpoint(initial);gpuGnn->restoreCheckpoint(initial);run(0,24,1,false);require("reset and deterministic replay",difference(expected,gpuPhysics[1].readback()),0);
        // Exercise the same setting operations that the UI invokes, including stream ownership.
        session.synchronize(false);require("unsync retains GNN without ML stream",session.actors[2].inference?1:0,0);
        int clip=(session.actors[2].animation+1)%int(session.assets.animations.size());session.select(2,clip);require("independent animation switch",session.actors[2].animation==clip&&!session.actors[2].inference?0:1,0);
        session.synchronize(true);require("resync restores master animation",session.actors[2].animation==session.actors[0].animation&&!session.actors[2].inference?0:1,0);
        pendingHybrid=0;operations();require("switch back restores ML stream",session.settings.hybridAlgorithm==0&&session.actors[2].inference?0:1,0);
        VK_CHECK_RESULT(vkDeviceWaitIdle(device));gpuGnn.reset();const auto provenance=session.assets.provenance;const auto before=session.actors[2].generation;
        session.assets.provenance[validationAlgorithm==2?"temporal_gnn_model":"gnn_model"]="__missing_gnn_validation_model__.vhood";pendingHybrid=validationAlgorithm;operations();
        require("failed model load retains current scene",session.settings.hybridAlgorithm==0&&session.actors[2].generation==before&&session.actors[2].inference?0:1,0);session.assets.provenance=provenance;
        pendingHybrid=validationAlgorithm;operations();require("switch to GNN releases ML stream",session.settings.hybridAlgorithm!=0&&!session.actors[2].inference?0:1,0);
        resetGpu();report["passed"]=passed;report["performance_qualified"]=false;std::ofstream(session.assets.directory/"gnn-integration-validation.json")<<report.dump(2);
        if(!passed)throw std::runtime_error("GNN integration validation failed");
    }
    void validatePresentation(Frame& f){
        // Explicit validation only. Normal playback keeps both physics endpoints
        // and generated normals on the device, with no cloth readback.
        VK_CHECK_RESULT(vkDeviceWaitIdle(device));const auto n=session.assets.cloth.rest.size();
        const VkDeviceSize pointBytes=6*n*sizeof(glm::vec4),normalBytes=3*n*sizeof(glm::vec4);
        vks::Buffer host;buffer(host,pointBytes+normalBytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,nullptr,true);
        auto cmd=vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        VkBufferCopy copy{0,0,pointBytes};vkCmdCopyBuffer(cmd,f.cloth.buffer,host.buffer,1,&copy);copy.dstOffset=pointBytes;copy.size=normalBytes;vkCmdCopyBuffer(cmd,f.normals.buffer,host.buffer,1,&copy);
        vulkanDevice->flushCommandBuffer(cmd,queue,true);
        const auto* points=static_cast<const glm::vec4*>(host.mapped);const auto* normals=points+6*n;bool passed=true;d::Json report;
        for(size_t actor=0;actor<3;++actor){std::vector<glm::vec3> expected(n,glm::vec3(0)),position(n);double maxError=0;
            for(size_t v=0;v<n;++v)position[v]=glm::mix(glm::vec3(points[(actor+3)*n+v]),glm::vec3(points[actor*n+v]),presentation[actor].blend);
            const auto& triangles=session.assets.cloth.triangles;
            for(size_t t=0;t<triangles.size();t+=3){auto a=triangles[t],b=triangles[t+1],c=triangles[t+2];auto normal=glm::cross(position[b]-position[a],position[c]-position[a]);expected[a]+=normal;expected[b]+=normal;expected[c]+=normal;}
            for(size_t v=0;v<n;++v){auto normal=glm::dot(expected[v],expected[v])>1e-20f?glm::normalize(expected[v]):glm::vec3(0,1,0);
                const double error=glm::length(normal-glm::vec3(normals[actor*n+v]));if(!std::isfinite(error)){maxError=1e30;break;}maxError=std::max(maxError,error);}
            const bool pass=maxError<=.002;passed&=pass;report["actors"].push_back({{"actor",actor},{"blend",presentation[actor].blend},{"display_time",presentation[actor].time},
                {"simulation_time",session.actors[actor].time},{"gpu_normal_max_error",maxError},{"passed",pass}});
        }
        host.destroy();report["passed"]=passed;report["scope"]="GPU normals vs CPU normals of the same interpolated frame endpoints; simulation and playback clock contracts are tested separately.";
        std::ofstream(session.assets.directory/"presentation-validation.json")<<report.dump(2);
        if(!passed)session.status="Display interpolation validation failed";
    }
    void dumpPhysics(){
        d::Json report;report["triangles"]=session.assets.cloth.triangles;report["body_triangles"]=session.assets.body.triangles;
        report["hybrid_algorithm"]=session.settings.hybridAlgorithm==2?"Temporal GNN + XPBD":session.settings.hybridAlgorithm==1?"GNN + XPBD":"MLCloth + XPBD";
        report["pinned"]=session.physicsMesh().pinned;report["collision_mode"]=session.settings.collisionMode;
        report["physics"]={{"tethers",session.settings.physics.enableTethers},{"tether_scale",session.settings.physics.tetherScale},{"contact_aware_guide",session.settings.physics.contactAwareGuide},
            {"hz",session.settings.physicsHz},{"iterations",session.settings.physics.iterations},{"bend_compliance",session.settings.physics.bendCompliance},
            {"stretch_compliance",session.settings.physics.stretchCompliance},{"shear_compliance",session.settings.physics.shearCompliance},
            {"guide_compliance",session.settings.physics.guideCompliance},{"body_collision",session.settings.physics.enableCollision},
            {"self_collision",session.settings.physics.enableSelfCollision},{"thickness",session.settings.physics.thickness},
            {"gravity",session.settings.physics.gravity},{"damping",session.settings.physics.dampingPerSecond},{"friction",session.settings.physics.friction}};
        for(auto v:session.physicsMesh().rest)report["rest"].push_back({v.x,v.y,v.z});
        for(int i=0;i<3;++i){d::Json actor;actor["algorithm"]=i;actor["time"]=session.actors[i].time;actor["animation"]=session.assets.animations[session.actors[i].animation].id;
            if(i&&session.gpuMode){for(auto v:gpuPhysics[i-1].readback())actor["cloth"].push_back({v.x,v.y,v.z});}
            else for(auto v:session.actors[i].cloth)actor["cloth"].push_back({v.x,v.y,v.z});
            std::vector<glm::vec3> body;session.assets.body.skin(session.actors[i].pose,body);for(auto v:body)actor["body"].push_back({v.x,v.y,v.z});
            auto collider=i==0||session.renderOnly?session.diagnosticCollider(i):session.actors[i].collider;
            for(auto v:collider.current)actor["proxy"].push_back({v.x,v.y,v.z});
            actor["capsules"]=d::Json::array();for(auto c:collider.capsules)actor["capsules"].push_back({{"a",{c.currentA.x,c.currentA.y,c.currentA.z}},{"b",{c.currentB.x,c.currentB.y,c.currentB.z}},{"radius",c.radius}});
            report["actors"].push_back(std::move(actor));}
        report["proxy_triangles"]=session.assets.collision.triangles;
        std::ofstream(session.assets.directory/"physics-snapshot.json")<<report.dump();
    }
    void validateAssetStep(){
        d::Json report;report["passed"]=true;
        auto& actor=session.actors[1];auto initial=gpuPhysics[0].readback();
        std::vector<d::Vec3> pins;for(auto v:actor.pinTargets)pins.push_back({v.x,v.y,v.z});
        auto body=actor.collider;body.previous=body.current;
        for(auto& c:body.capsules){c.previousA=c.currentA;c.previousB=c.currentB;}
        d::Json fixture;fixture["triangles"]=session.physicsMesh().triangles;fixture["mass"]=session.physicsMesh().mass;fixture["pinned"]=session.physicsMesh().pinned;
        for(auto v:session.physicsMesh().rest)fixture["rest"].push_back({v.x,v.y,v.z});
        for(auto v:initial)fixture["positions"].push_back({v.x,v.y,v.z});for(auto v:pins)fixture["pins"].push_back({v.x,v.y,v.z});
        fixture["thickness"]=session.settings.physics.thickness;
        for(int scenario=0;scenario<3;++scenario){
            auto config=session.settings.physics;config.gravity=0;config.iterations=scenario==0?2:0;
            config.enableCollision=scenario==1;config.enableSelfCollision=scenario==2;config.guideCompliance=-1;
            actor.solver.captureContacts=scenario==2;gpuPhysics[0].captureContacts=scenario==2;
            actor.solver.reset(initial);gpuPhysics[0].reset(initial);
            auto cmd=vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);
            gpuPhysics[0].recordStep(cmd,1.f/240,pins,nullptr,&body,config);vulkanDevice->flushCommandBuffer(cmd,queue,true);
            actor.solver.step(1.f/240,pins,nullptr,&body,config);auto actual=gpuPhysics[0].readback();
            if(scenario==2){
                auto records=[](const std::vector<d::SelfContactRecord>& contacts){d::Json j=d::Json::array();for(auto c:contacts)j.push_back({{"ids",c.ids},{"weights",c.weights},{"delta",{c.delta.x,c.delta.y,c.delta.z}},{"distance",c.distance}});return j;};
                fixture["cpu_contacts"]=records(actor.solver.contactRecords);fixture["gpu_contacts"]=records(gpuPhysics[0].readContactRecords());
                for(auto v:actual)fixture["gpu"].push_back({v.x,v.y,v.z});std::ofstream(session.assets.directory/"self-contact-fixture.json")<<fixture.dump();}
            std::vector<double> errors;uint32_t worst=0;
            for(uint32_t v=0;v<actual.size();++v){errors.push_back(d::length(actual[v]-actor.solver.positions()[v]));if(errors[v]>errors[worst])worst=v;}
            const double maximum=errors[worst];bool passed=std::isfinite(maximum)&&maximum<=2e-5;
            report["cases"].push_back({{"name",std::array<const char*,3>{"structural","body contact","self contact"}[scenario]},
                {"max_error_m",maximum},{"p95_error_m",percentile(errors,.95)},{"worst_vertex",worst},{"passed",passed}});
            auto expected=actor.solver.positions()[worst],observed=actual[worst],start=initial[worst];
            report["cases"].back()["cpu_position"]={expected.x,expected.y,expected.z};report["cases"].back()["gpu_position"]={observed.x,observed.y,observed.z};
            report["cases"].back()["initial_position"]={start.x,start.y,start.z};report["cases"].back()["pinned"]=session.physicsMesh().pinned[worst];
            if(!passed)report["passed"]=false;
        }
        actor.solver.captureContacts=false;gpuPhysics[0].captureContacts=false;
        {
            auto config=session.settings.physics;config.guideCompliance=-1;config.enableCollision=config.enableSelfCollision=true;
            gpuPhysics[0].reset(initial);
            auto advance=[&](uint32_t steps){auto cmd=vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);
                for(uint32_t i=0;i<steps;++i)gpuPhysics[0].recordStep(cmd,1.f/240,pins,nullptr,&body,config);
                vulkanDevice->flushCommandBuffer(cmd,queue,true);};
            advance(12);const uint64_t checkpoint=~uint64_t(0);
            auto cmd=vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);gpuPhysics[0].recordCheckpoint(cmd,checkpoint);vulkanDevice->flushCommandBuffer(cmd,queue,true);
            advance(48);auto expected=gpuPhysics[0].readback();gpuPhysics[0].restoreCheckpoint(checkpoint);advance(48);auto replay=gpuPhysics[0].readback();
            double maximum=0;for(size_t i=0;i<replay.size();++i){const double error=d::length(replay[i]-expected[i]);if(!std::isfinite(error)){maximum=std::numeric_limits<double>::infinity();break;}maximum=std::max(maximum,error);}
            const bool passed=maximum==0;report["cases"].push_back({{"name","full mesh checkpoint replays 48 contact substeps exactly"},{"max_error_m",maximum},{"passed",passed}});
            if(!passed)report["passed"]=false;
            std::vector<uint64_t> keep;for(const auto& entry:actor.checkpoints)keep.push_back(entry.second.id);gpuPhysics[0].retainCheckpoints(keep);
        }
        gpuPhysics[0].restoreCheckpoint(actor.checkpoints.at(0).id);
        report["vertices"]=initial.size();report["note"]="Full asset single-step comparisons and short static-body checkpoint replay; not long-run quality qualification.";
        std::ofstream(session.assets.directory/"gpu-asset-validation.json")<<report.dump(2);
    }
    static double percentile(std::vector<double> a,double p){if(a.empty())return 0;std::sort(a.begin(),a.end());return a[std::min(a.size()-1,static_cast<size_t>(p*(a.size()-1)))];}
    void writeMetrics(){d::Json report={{"device",deviceProperties.deviceName},{"resolution",{width,height}},{"samples",cpuTimes.size()},{"cpu_p95_ms",percentile(cpuTimes,.95)},{"gpu_p95_ms",percentile(gpuTimes,.95)},{"frame_p95_ms",percentile(frameTimes,.95)},{"frame_p99_ms",percentile(frameTimes,.99)},{"physics_hz",session.settings.physicsHz},{"backend",session.gpuMode?"Vulkan compute":"CPU reference"},{"render_only",session.renderOnly},{"qualified",false},{"synchronized",session.settings.synchronized}};
        report["hybrid_algorithm"]=session.settings.hybridAlgorithm==2?"Temporal GNN + XPBD":session.settings.hybridAlgorithm==1?"GNN + XPBD":"MLCloth + XPBD";
        if(session.settings.hybridAlgorithm!=0){report["gnn_model"]=std::filesystem::weakly_canonical(gnnModelPath()).string();report["gnn_hz"]=30;report["gnn_sampled_inference_ms"]=gnnMs;report["gnn_inference_samples"]=gnnTimes.size();report["gnn_inference_p95_ms"]=percentile(gnnTimes,.95);report["gnn_buffer_allocation_bytes"]=gpuGnn?gpuGnn->allocatedBytes():0;report["gnn_strength"]=session.settings.gnnStrength;report["gnn_trust"]=session.settings.gnnTrust;}
        report["step_upload_integrate_ms"]=stepStages[0][0];report["step_structural_ms"]=stepStages[0][1];report["step_self_contact_ms"]=stepStages[0][2];report["step_body_contact_ms"]=stepStages[0][3];report["self_collision"]=session.settings.physics.enableSelfCollision;
        report["step_self_refit_ms"]=stepStages[0][4];report["step_self_vf_ms"]=stepStages[0][5];report["step_self_ee_ms"]=stepStages[0][6];
        report["gpu_render_ms"]=graphicsMs;report["sampled_substep_stages_ms"]=stepStages;
        report["stage_order"]={"upload and integrate","structure and ML guide","self contact","body contact and velocity","self refit","self VF","self EE"};
        report["shared_ml_inference_ms"]=session.settings.synchronized?session.actors[0].inferenceMs:0.;
        report["independent_ml_inference_ms"]=session.settings.synchronized?d::Json::array():d::Json{session.actors[0].inferenceMs,session.actors[2].inferenceMs};
        report["collision_mode"]=session.settings.collisionMode;report["leg_capsules"]=session.assets.capsules.size();
        report["contact_diagnostics"]=contactDiagnostics;
        if(session.gpuMode)report["candidate_cache"]={gpuPhysics[0].candidateCacheDiagnostics(),gpuPhysics[1].candidateCacheDiagnostics()};
        report["paused"]=session.settings.paused;report["backlog_seconds"]=session.accumulator;report["fixed_frame_dt"]=argument("--fixed-frame-dt");
        report["simulation_time"]=session.actors[0].time;report["status"]=session.status;
        std::ofstream stream(session.assets.directory/"performance.json");stream<<report.dump(2);}
    void keyPressed(uint32_t key)override{if(key==0x20)session.settings.paused=!session.settings.paused;if(key==0x52)pendingReset=true;if(key==0x48)uiHidden=!uiHidden;}
    void mouseMoved(double x,double y,bool& handled)override{
        if(!ImGui::GetIO().WantCaptureMouse&&mouseState.buttons.left){orbitYaw+=float(x-mouseState.position.x)*.25f;orbitPitch=std::clamp(orbitPitch+float(y-mouseState.position.y)*.15f,-15.f,65.f);handled=true;}
    }
    bool animationCombo(const char* label,int& selected){
        std::vector<const char*> names;for(auto& a:session.assets.animations)names.push_back(a.name.c_str());return ImGui::Combo(label,&selected,names.data(),static_cast<int>(names.size()));
    }
    void collisionOverlay(){
        if(!session.settings.showCollision)return;
        auto* list=ImGui::GetOverlayDrawList();const auto& actor=presentation[selectedActor];
        const auto collider=session.diagnosticCollider(selectedActor,true);
        auto line=[&](glm::vec3 a,glm::vec3 b,ImU32 color){
            auto aa=scene.viewProjection*glm::vec4(a+actor.displayOffset,1),bb=scene.viewProjection*glm::vec4(b+actor.displayOffset,1);
            if(aa.w<=.05f||bb.w<=.05f)return;aa/=aa.w;bb/=bb.w;
            list->AddLine(ImVec2((aa.x+1)*width*.5f,(aa.y+1)*height*.5f),ImVec2((bb.x+1)*width*.5f,(bb.y+1)*height*.5f),color);};
        if(session.settings.collisionMode!=2)for(auto [a,b]:collisionEdges)if(a<collider.current.size()&&b<collider.current.size()){
            auto aa=collider.current[a],bb=collider.current[b];line({aa.x,aa.y,aa.z},{bb.x,bb.y,bb.z},IM_COL32(63,220,232,120));}
        for(auto c:collider.capsules){glm::vec3 a(c.currentA.x,c.currentA.y,c.currentA.z),b(c.currentB.x,c.currentB.y,c.currentB.z);
            auto axis=glm::normalize(b-a),u=glm::cross(axis,glm::vec3(1,0,0));if(glm::length(u)<1e-6f)u=glm::cross(axis,glm::vec3(0,1,0));u=glm::normalize(u);auto v=glm::cross(axis,u);
            for(int k=0;k<24;++k){float angle=k*glm::radians(15.f),next=(k+1)*glm::radians(15.f);auto p=u*std::cos(angle)+v*std::sin(angle),q=u*std::cos(next)+v*std::sin(next);
                line(a+p*c.radius,a+q*c.radius,IM_COL32(255,181,63,210));line(b+p*c.radius,b+q*c.radius,IM_COL32(255,181,63,210));
                if(k%6==0)line(a+p*c.radius,b+p*c.radius,IM_COL32(255,181,63,210));
                for(auto tangent:{u,v}){auto p2=axis*std::cos(angle)+tangent*std::sin(angle),q2=axis*std::cos(next)+tangent*std::sin(next);
                    line((k>=6&&k<18?a:b)+p2*c.radius,(k>=6&&k<18?a:b)+q2*c.radius,IM_COL32(255,181,63,150));}}
        }
    }
    void OnUpdateUIOverlay(vks::UIOverlay*)override{
        if(uiHidden)return;const auto flags=ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoCollapse;
        collisionOverlay();
        ImGui::SetNextWindowPos(ImVec2(20,20));ImGui::SetNextWindowSize(ImVec2(float(width)-40,74));ImGui::Begin("Cloth Studio controls",nullptr,flags|ImGuiWindowFlags_NoTitleBar);
        ImGui::TextUnformatted("CLOTH STUDIO");ImGui::SameLine(190);ImGui::PushItemWidth(180);int clip=session.actors[session.settings.synchronized?0:selectedActor].animation;
        if(animationCombo("##animation",clip)){pendingClip=clip;pendingActor=selectedActor;}ImGui::PopItemWidth();ImGui::SameLine();
        if(ImGui::Button(session.settings.paused?"Play":"Pause"))session.settings.paused=!session.settings.paused;ImGui::SameLine();if(ImGui::Button("Reset"))pendingReset=true;
        ImGui::SameLine();bool sync=session.settings.synchronized;if(ImGui::Checkbox("Sync",&sync)){pendingSync=true;syncValue=sync;}
        ImGui::SameLine();if(ImGui::Button("Settings"))session.settings.showSettings=!session.settings.showSettings;ImGui::SameLine();if(ImGui::Button("Performance"))session.settings.showStats=!session.settings.showStats;
        ImGui::SameLine();if(ImGui::Button("Hide UI [H]"))uiHidden=true;
        ImGui::Text("%s",session.status.c_str());ImGui::End();
        const char* labels[]={"MLCloth","XPBD",session.settings.hybridAlgorithm==2?"Temporal GNN + XPBD":session.settings.hybridAlgorithm==1?"GNN + XPBD":"MLCloth + XPBD"};const ImVec4 colors[]={{.3f,.6f,1,1},{1,.65f,.3f,1},{.35f,.85f,.6f,1}};
        for(int i=0;i<3;++i){ImGui::SetNextWindowPos(ImVec2(20+i*(float(width)-40)/3,110));ImGui::SetNextWindowSize(ImVec2((float(width)-60)/3,session.settings.synchronized?52.f:104.f));std::string name="Actor "+std::to_string(i);ImGui::Begin(name.c_str(),nullptr,flags|ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoScrollbar);
            ImGui::PushStyleColor(ImGuiCol_Text,colors[i]);if(ImGui::Selectable(labels[i],selectedActor==i))selectedActor=i;ImGui::PopStyleColor();
            if(!session.settings.synchronized){int a=session.actors[i].animation;ImGui::PushItemWidth(180);if(animationCombo("##motion",a)){pendingClip=a;pendingActor=i;}ImGui::PopItemWidth();ImGui::SameLine();ImGui::Checkbox("Pause",&session.actors[i].paused);}
            ImGui::End();}
        ImGui::SetNextWindowPos(ImVec2(20,float(height)-90));ImGui::SetNextWindowSize(ImVec2(float(width)-40,70));ImGui::Begin("Timeline",nullptr,flags|ImGuiWindowFlags_NoTitleBar);
        auto& actor=session.actors[session.settings.synchronized?0:selectedActor];float t=float(session.settings.loop?std::fmod(actor.time,session.assets.animations[actor.animation].duration):std::min(actor.time,session.assets.animations[actor.animation].duration));
        ImGui::PushItemWidth(float(width)-460);if(ImGui::SliderFloat("##time",&t,0,float(session.assets.animations[actor.animation].duration),"%.2f s")){seekTime=t;pendingSeek=true;}ImGui::PopItemWidth();ImGui::SameLine();ImGui::PushItemWidth(130);ImGui::SliderFloat("Speed",&session.settings.speed,.25f,2.f,"%.2fx");ImGui::PopItemWidth();ImGui::SameLine();if(ImGui::Checkbox("Loop",&session.settings.loop)){pendingReset=true;pendingResetAll=true;}ImGui::End();
        if(session.settings.showSettings){ImGui::SetNextWindowPos(ImVec2(float(width)-370,240),ImGuiSetCond_FirstUseEver);ImGui::SetNextWindowSize(ImVec2(350,520),ImGuiSetCond_FirstUseEver);ImGui::Begin("Settings",&session.settings.showSettings);ImGui::PushItemWidth(170);
            const char* focus[]={"All characters","MLCloth","XPBD","Hybrid"};int f=session.settings.focus+1;if(ImGui::Combo("Focus",&f,focus,4))session.settings.focus=f-1;
            const char* views[]={"Front","Side","Back","Three quarter"};if(ImGui::Combo("View",&session.settings.view,views,4))orbitYaw=std::array<float,4>{180,90,0,145}[session.settings.view];
            ImGui::Checkbox("Follow motion",&session.settings.autoCamera);ImGui::Checkbox("Fixed positions",&session.settings.inPlace);ImGui::SliderFloat("Distance",&distance,3,18);
            ImGui::Checkbox("Space actors by motion range",&session.settings.autoSpacing);ImGui::SliderFloat(session.settings.autoSpacing?"Minimum spacing":"Spacing",&session.settings.spacing,1.5,8);
            const char* materials[]={"Character materials","Neutral gray","Algorithm colors"};ImGui::Combo("Material",&session.settings.material,materials,3);
            if(session.settings.material==0)ImGui::TextUnformatted("Body textures / solid-color cloth");
            if(!session.renderOnly){const char* algorithms[]={"MLCloth + XPBD","GNN + XPBD","Temporal GNN + XPBD"};int hybrid=session.settings.hybridAlgorithm;
                if(ImGui::Combo("Third character",&hybrid,algorithms,3))pendingHybrid=hybrid;
                if(session.settings.hybridAlgorithm!=0){ImGui::TextUnformatted("TinyHOOD 32 x 12 / 30 Hz");bool changed=ImGui::SliderFloat("GNN strength",&session.settings.gnnStrength,0,1);
                    changed|=ImGui::SliderFloat("Prediction trust radius",&session.settings.gnnTrust,.1f,8.f,"%.1fx edge");
                    if(changed){pendingReset=true;pendingResetAll=true;}}
            }
            ImGui::Checkbox("Collision overlay (selected)",&session.settings.showCollision);
            if(session.settings.showCollision)ImGui::TextUnformatted("X-ray: cyan STM / amber capsules");
            const char* quality[]={"120 Hz","240 Hz","480 Hz"};int qualityIndex=session.settings.physicsHz==120?0:session.settings.physicsHz==240?1:2;if(ImGui::Combo("Physics",&qualityIndex,quality,3)){session.settings.physicsHz=120<<qualityIndex;pendingReset=true;pendingResetAll=true;}
            if(!session.renderOnly){const char* backends[]={"CPU reference","Vulkan compute"};int backend=session.gpuMode?1:0;if(ImGui::Combo("Solver",&backend,backends,2)){pendingBackend=backend;session.status="Rebuilding solver state...";}}
            int threadIndex=session.settings.threads==1?0:session.settings.threads==2?1:2;const char* threadNames[]={"1","2","4"};if(ImGui::Combo("ML threads",&threadIndex,threadNames,3))pendingThreads=1<<threadIndex;
            if(ImGui::CollapsingHeader("Physics parameters")){auto& p=session.settings.physics;bool changed=false;
                changed|=ImGui::SliderInt("Iterations",&p.iterations,1,8);
                changed|=ImGui::Checkbox("Body collision",&p.enableCollision);changed|=ImGui::Checkbox("Self collision",&p.enableSelfCollision);
                changed|=ImGui::Checkbox("Limit global stretching",&p.enableTethers);
                if(ImGui::IsItemHovered())ImGui::SetTooltip("Experimental: reduces stretching, but may worsen contact during fast turns.");
                if(p.enableTethers)changed|=ImGui::SliderFloat("Path length allowance",&p.tetherScale,1.f,1.3f,"%.2fx");
                changed|=ImGui::SliderFloat("Thickness (m)",&p.thickness,.001f,.01f,"%.3f");
                changed|=ImGui::SliderFloat("Damping / s",&p.dampingPerSecond,0,4);changed|=ImGui::SliderFloat("Friction",&p.friction,0,1);
                changed|=ImGui::SliderFloat("ML guide compliance",&p.guideCompliance,.00001f,.1f,"%.5f",3);
                changed|=ImGui::Checkbox("Contact-aware ML guidance",&p.contactAwareGuide);
                if(ImGui::IsItemHovered())ImGui::SetTooltip("Reduce coarse attraction into nearby STM, capsules or floor. Hybrid actor only.");
                changed|=ImGui::SliderFloat("Stretch compliance",&p.stretchCompliance,0,.01f,"%.7f",4);
                changed|=ImGui::SliderFloat("Shear compliance",&p.shearCompliance,0,.01f,"%.7f",4);
                changed|=ImGui::Combo("Body representation",&session.settings.collisionMode,"STM + leg capsules\0STM only\0Leg capsules (diagnostic)\0");
                float bendLog=std::log10(std::max(p.bendCompliance,1e-4f));if(ImGui::SliderFloat("Bend log10 compliance",&bendLog,-4,6,"%.2f")){p.bendCompliance=std::pow(10.f,bendLog);changed=true;}
                if(changed){pendingReset=true;pendingResetAll=true;session.status="Applying physics settings and rebuilding checkpoints...";}}

            if(ImGui::Button("Save preferences")){try{session.saveSettings();session.status="Preferences saved";}catch(const std::exception& e){session.status=e.what();}}ImGui::SameLine();if(ImGui::Button("Defaults")){session.settings=d::DemoSettings{};orbitYaw=180;orbitPitch=8;distance=5.8f;pendingReset=true;pendingResetAll=true;}
            ImGui::PopItemWidth();ImGui::End();}
        if(session.settings.showStats){ImGui::SetNextWindowSize(ImVec2(370,260),ImGuiSetCond_FirstUseEver);ImGui::Begin("Performance",&session.settings.showStats);
            ImGui::Text("CPU frame %.2f ms | GPU %.2f ms",cpuMs,gpuMs);ImGui::Text("Physics: %d Hz, %s",session.settings.physicsHz,session.gpuMode?"Vulkan compute":"CPU reference");
            ImGui::Text("Render %.2f ms | Frame %.2f ms",graphicsMs,frameTimer*1000);
            if(session.settings.hybridAlgorithm!=0)ImGui::Text("ML A %.2f ms | GNN %.2f ms / inference",session.actors[0].inferenceMs,gnnMs);
            else if(session.settings.synchronized)ImGui::Text("Shared ML inference %.2f ms",session.actors[0].inferenceMs);
            else ImGui::Text("ML inference A %.2f / C %.2f ms",session.actors[0].inferenceMs,session.actors[2].inferenceMs);
            for(int i=0;i<2;++i){double solve=session.gpuMode?std::accumulate(stepStages[i].begin(),stepStages[i].begin()+4,0.):session.actors[i+1].solveMs;ImGui::Text("%s: sampled step %.2f ms",labels[i+1],solve);}
            if(ImGui::Button(capture?"Stop measurement":"Start measurement")){capture=!capture;if(capture){cpuTimes.clear();gpuTimes.clear();frameTimes.clear();}}
            ImGui::SameLine();if(ImGui::Button("Export metrics"))writeMetrics();ImGui::Text("Captured frames: %u",uint32_t(cpuTimes.size()));ImGui::End();}
    }
};
VULKAN_EXAMPLE_MAIN()
