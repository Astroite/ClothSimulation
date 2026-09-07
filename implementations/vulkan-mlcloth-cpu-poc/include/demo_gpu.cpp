#include "demo_gpu.h"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <cstring>
#include <functional>
#include <map>

namespace mlcloth::demo {
namespace {
struct F4 {float x{},y{},z{},w{};};
struct U4 {uint32_t x{},y{},z{},w{};};
struct Constraint {U4 ids;F4 rest;};
struct Node {F4 lo,hi;U4 links;}; // DFS tree; links={right,escape,primitive,leaf}
struct Push {uint32_t phase{},start{},count{},vertices{};float dt{},alpha{},gravity{},damping{};
             float thickness{},friction{},guide{};uint32_t bodyCount{},treeCount{},edges{},pad0{},pad1{};};
void check(VkResult result){if(result!=VK_SUCCESS)throw std::runtime_error("Vulkan cloth error "+std::to_string(result));}
F4 f4(Vec3 v,float w=0){return {v.x,v.y,v.z,w};}
Vec3 xyz(F4 v){return {v.x,v.y,v.z};}
}
struct GpuPhysics::Impl {
    vks::VulkanDevice* device{};VkQueue queue{};std::array<VkPipeline,17> pipelines{};uint32_t boundPhase{~0u},tetherStart{};VkPipelineLayout layout{};
    VkDescriptorSetLayout setLayout{};VkDescriptorPool pool{};VkDescriptorSet set{};
    // x, velocity, previous, pins, guide, constraints, lambda, patch CSR, patch verts,
    // patch lambda, body points, body previous, body triangles, body tree,
    // cloth triangles, cloth tree, self snapshot, self contact counts, corrections,
    // edges, edge tree, adjacency offsets, adjacency vertices.
    std::array<vks::Buffer,36> buffers;
    Mesh mesh;PhysicsSolver tables;std::vector<Constraint> constraints;
    std::map<uint64_t,std::array<vks::Buffer,3>> checkpoints;
    std::vector<std::pair<uint32_t,uint32_t>> groups;
    std::vector<Node> bodyTree,clothTree,edgeTree;
    std::vector<uint32_t> bodyTopology;
    BodySurfaceTopology bodySurface;
    std::vector<std::pair<uint32_t,uint32_t>> clothLevels,edgeLevels;
    std::vector<uint32_t> clothLevelIds,edgeLevelIds;
    uint32_t n{},patches{},clothDepth{},edgeDepth{};static constexpr uint32_t bodyCapacity=8192,triangleCapacity=16384;
    void make(uint32_t slot,size_t bytes,const void* source=nullptr){
        bytes=std::max<size_t>(bytes,16);std::vector<uint8_t> zero;if(!source){zero.resize(bytes);source=zero.data();}
        vks::Buffer staging;check(device->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&staging,bytes,const_cast<void*>(source)));
        check(device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,&buffers[slot],bytes));
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);VkBufferCopy copy{0,0,bytes};
        vkCmdCopyBuffer(cmd,staging.buffer,buffers[slot].buffer,1,&copy);device->flushCommandBuffer(cmd,queue,true);staging.destroy();
    }
    template<class T>void make(uint32_t slot,const std::vector<T>& values){make(slot,values.size()*sizeof(T),values.empty()?nullptr:values.data());}
    void barrier(VkCommandBuffer cmd){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&b,0,nullptr,0,nullptr);}
    void update(VkCommandBuffer cmd,uint32_t slot,const void* data,size_t bytes){
        if(bytes>buffers[slot].size||bytes%4)throw std::runtime_error("GPU target capacity exceeded");
        for(size_t offset=0;offset<bytes;offset+=65536)vkCmdUpdateBuffer(cmd,buffers[slot].buffer,offset,std::min<size_t>(65536,bytes-offset),static_cast<const uint8_t*>(data)+offset);
    }
    void dispatch(VkCommandBuffer cmd,Push p,uint32_t count){if(!count)return;p.count=count;
        if(boundPhase!=p.phase){vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipelines[p.phase]);boundPhase=p.phase;}
        vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);vkCmdDispatch(cmd,(count+63)/64,1,1);barrier(cmd);}
    std::vector<Node> tree(const std::vector<Vec3>& points,const std::vector<uint32_t>& primitives,uint32_t stride){
        std::vector<Node> result;std::vector<uint32_t> order(primitives.size()/stride);std::iota(order.begin(),order.end(),0);
        std::function<uint32_t(size_t,size_t,uint32_t)> build=[&](size_t first,size_t end,uint32_t depth){
            uint32_t id=static_cast<uint32_t>(result.size());result.emplace_back();
            Vec3 lo{1e30f,1e30f,1e30f},hi{-1e30f,-1e30f,-1e30f};
            for(size_t i=first;i<end;++i)for(uint32_t j=0;j<stride;++j){auto p=points[primitives[order[i]*stride+j]];
                lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};}
            result[id].lo=f4(lo,float(depth));result[id].hi=f4(hi);
            if(end-first==1){result[id].links.z=order[first];result[id].links.w=1;}
            else{Vec3 extent=hi-lo;int axis=extent.y>extent.x?1:0;if(extent.z>(axis==0?extent.x:extent.y))axis=2;
                size_t middle=(first+end)/2;
                auto center=[&](uint32_t primitive){float sum=0;for(uint32_t j=0;j<stride;++j){auto p=points[primitives[primitive*stride+j]];sum+=axis==0?p.x:axis==1?p.y:p.z;}return sum;};
                std::nth_element(order.begin()+first,order.begin()+middle,order.begin()+end,[&](uint32_t a,uint32_t b){return center(a)<center(b);});
                build(first,middle,depth+1);result[id].links.x=build(middle,end,depth+1);}
            result[id].links.y=static_cast<uint32_t>(result.size());return id;
        };if(!order.empty())build(0,order.size(),0);return result;
    }
    void refitBody(const TriangleCollider& body){
        if(bodyTopology!=body.triangles){bodyTopology=body.triangles;bodyTree=tree(body.current,bodyTopology,3);}
        for(size_t i=bodyTree.size();i-->0;){auto& node=bodyTree[i];Vec3 lo{1e30f,1e30f,1e30f},hi{-1e30f,-1e30f,-1e30f};
            auto include=[&](Vec3 p){lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};};
            if(node.links.w)for(uint32_t j=0;j<3;++j){auto v=bodyTopology[node.links.z*3+j];include(body.previous[v]);include(body.current[v]);}
            else{include(xyz(bodyTree[i+1].lo));include(xyz(bodyTree[i+1].hi));include(xyz(bodyTree[node.links.x].lo));include(xyz(bodyTree[node.links.x].hi));}
            node.lo=f4(lo);node.hi=f4(hi);
        }
    }
};
GpuPhysics::GpuPhysics():impl_(std::make_unique<Impl>()){}
GpuPhysics::~GpuPhysics(){auto& s=*impl_;if(!s.device)return;auto d=s.device->logicalDevice;
    for(auto& entry:s.checkpoints)for(auto& b:entry.second)b.destroy();
    for(auto& b:s.buffers)b.destroy();for(auto pipeline:s.pipelines)if(pipeline)vkDestroyPipeline(d,pipeline,nullptr);if(s.layout)vkDestroyPipelineLayout(d,s.layout,nullptr);
    if(s.pool)vkDestroyDescriptorPool(d,s.pool,nullptr);if(s.setLayout)vkDestroyDescriptorSetLayout(d,s.setLayout,nullptr);}
void GpuPhysics::build(vks::VulkanDevice* device,VkQueue queue,VkPipelineCache cache,const std::filesystem::path& shader,const Mesh& mesh){
    std::ofstream trace("gpu-build.log",std::ios::app);trace<<"Begin "<<mesh.rest.size()<<" vertices"<<std::endl;
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};properties.pNext=&subgroup;
    vkGetPhysicalDeviceProperties2(device->physicalDevice,&properties);
    if(subgroup.subgroupSize!=32||!(subgroup.supportedOperations&VK_SUBGROUP_FEATURE_ARITHMETIC_BIT))
        throw std::runtime_error("Cloth contact compute requires 32-lane subgroup support");
    auto& s=*impl_;if(s.device)throw std::runtime_error("GPU solver already built");s.device=device;s.queue=queue;s.mesh=mesh;s.tables.build(mesh);s.n=static_cast<uint32_t>(mesh.rest.size());
    trace<<"Tables ready"<<std::endl;
    std::vector<Constraint> raw;
    for(size_t i=0;i<s.tables.stretchRestLength().size();++i)raw.push_back({{s.tables.stretchPairs()[2*i],s.tables.stretchPairs()[2*i+1],0,0},{s.tables.stretchRestLength()[i],0,0,0}});
    for(auto c:s.tables.shearConstraints())raw.push_back({{c.a,c.b,c.c,0},{c.restDot,1,0,0}});
    for(auto c:s.tables.bendConstraints())raw.push_back({{c.a,c.b,c.c,c.d},{c.restAngle,2,0,0}});
    // Color all participating vertices, including the shared edge of a dihedral.
    // Keep type order identical to the scalar oracle's stretch/shear/bend passes.
    for(uint32_t type=0;type<3;++type){std::vector<std::vector<Constraint>> colors;std::vector<std::vector<uint8_t>> occupied;
        for(auto c:raw){if(uint32_t(c.rest.y)!=type)continue;uint32_t ids[]={c.ids.x,c.ids.y,c.ids.z,c.ids.w};uint32_t arity=type+2;size_t g=0;
            for(;g<colors.size();++g){bool conflict=false;for(uint32_t k=0;k<arity;++k)conflict|=occupied[g][ids[k]]!=0;if(!conflict)break;}
            if(g==colors.size()){colors.emplace_back();occupied.emplace_back(s.n,0);}colors[g].push_back(c);for(uint32_t k=0;k<arity;++k)occupied[g][ids[k]]=1;}
        for(const auto& group:colors){s.groups.emplace_back(static_cast<uint32_t>(s.constraints.size()),static_cast<uint32_t>(group.size()));s.constraints.insert(s.constraints.end(),group.begin(),group.end());}}
    trace<<"Colored "<<s.constraints.size()<<" records in "<<s.groups.size()<<" groups"<<std::endl;
    s.patches=s.tables.patchCount();std::vector<F4> positions;for(size_t i=0;i<mesh.rest.size();++i)positions.push_back(f4(mesh.rest[i],mesh.pinned[i]?0:1/mesh.mass[i]));
    auto records=s.constraints;s.tetherStart=static_cast<uint32_t>(records.size());
    for(const auto& t:s.tables.tethers())records.push_back({{t[0].anchor,t[1].anchor,t[2].anchor,t[3].anchor},{t[0].length,t[1].length,t[2].length,t[3].length}});
    s.make(0,positions);s.make(1,s.n*sizeof(F4));s.make(2,positions);s.make(3,positions);s.make(4,positions);s.make(5,records);s.make(6,s.constraints.size()*sizeof(float));
    s.make(7,s.tables.patchOffsets());s.make(8,s.tables.patchVertices());s.make(9,s.patches*sizeof(F4));
    s.make(10,Impl::bodyCapacity*sizeof(F4));s.make(11,Impl::bodyCapacity*sizeof(F4));s.make(12,Impl::triangleCapacity*3*sizeof(uint32_t));s.make(13,Impl::triangleCapacity*2*sizeof(Node));
    trace<<"Static buffers ready"<<std::endl;
    s.make(14,mesh.triangles);s.clothTree=s.tree(mesh.rest,mesh.triangles,3);trace<<"Cloth tree "<<s.clothTree.size()<<std::endl;s.make(15,s.clothTree);s.make(16,positions);s.make(17,(s.n+1)*sizeof(uint32_t));s.make(18,s.n*256*sizeof(F4));
    s.make(19,s.tables.stretchPairs());s.edgeTree=s.tree(mesh.rest,s.tables.stretchPairs(),2);trace<<"Edge tree "<<s.edgeTree.size()<<std::endl;s.make(20,s.edgeTree);
    for(auto node:s.clothTree)s.clothDepth=std::max(s.clothDepth,uint32_t(node.lo.w));
    for(auto node:s.edgeTree)s.edgeDepth=std::max(s.edgeDepth,uint32_t(node.lo.w));
    s.make(21,s.tables.selfExclusionOffsets());s.make(22,s.tables.selfExclusionVertices());s.make(29,s.tables.selfExclusionDistances());
    s.make(23,32769*sizeof(uint32_t));s.make(24,32768*513*sizeof(uint32_t));
    s.make(25,32769*sizeof(uint32_t));s.make(26,32768*513*sizeof(uint32_t));
    s.make(27,mesh.triangles.size()/3*2*sizeof(F4));s.make(28,s.tables.stretchRestLength().size()*2*sizeof(F4));
    s.make(30,(s.n+s.tables.stretchRestLength().size())*sizeof(SelfContactRecord));
    s.make(31,(16+3*s.n+(s.n+s.tables.stretchRestLength().size())*129)*sizeof(uint32_t));
    s.make(32,(5+Impl::bodyCapacity+6*Impl::triangleCapacity)*sizeof(uint32_t));
    s.make(33,(Impl::bodyCapacity+4*Impl::triangleCapacity)*sizeof(F4));
    s.make(34,s.n*3*sizeof(F4)); // floor/capsule/STM contact planes, regenerated before coarse guidance
    s.make(35,s.n*sizeof(F4)); // held GNN acceleration, checkpointed with position/velocity
    trace<<"Buffers ready"<<std::endl;
    std::vector<VkDescriptorSetLayoutBinding> bindings;for(uint32_t i=0;i<s.buffers.size();++i)bindings.push_back({i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr});
    VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=static_cast<uint32_t>(bindings.size());sl.pBindings=bindings.data();check(vkCreateDescriptorSetLayout(device->logicalDevice,&sl,nullptr,&s.setLayout));
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,static_cast<uint32_t>(bindings.size())};VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pi.maxSets=1;pi.poolSizeCount=1;pi.pPoolSizes=&size;check(vkCreateDescriptorPool(device->logicalDevice,&pi,nullptr,&s.pool));
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=s.pool;ai.descriptorSetCount=1;ai.pSetLayouts=&s.setLayout;check(vkAllocateDescriptorSets(device->logicalDevice,&ai,&s.set));
    std::vector<VkWriteDescriptorSet> writes;for(uint32_t i=0;i<s.buffers.size();++i){VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w.dstSet=s.set;w.dstBinding=i;w.descriptorCount=1;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&s.buffers[i].descriptor;writes.push_back(w);}vkUpdateDescriptorSets(device->logicalDevice,static_cast<uint32_t>(writes.size()),writes.data(),0,nullptr);
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(Push)};VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.setLayoutCount=1;li.pSetLayouts=&s.setLayout;li.pushConstantRangeCount=1;li.pPushConstantRanges=&range;check(vkCreatePipelineLayout(device->logicalDevice,&li,nullptr,&s.layout));
    std::ifstream file(shader,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("Missing GPU cloth shader");auto bytes=static_cast<size_t>(file.tellg());if(bytes%4)throw std::runtime_error("Invalid SPIR-V length");std::vector<uint32_t> words(bytes/4);file.seekg(0);file.read(reinterpret_cast<char*>(words.data()),bytes);
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=bytes;mi.pCode=words.data();VkShaderModule module{};check(vkCreateShaderModule(device->logicalDevice,&mi,nullptr,&module));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.layout=s.layout;ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";
    trace<<"Creating pipeline"<<std::endl;
    VkSpecializationMapEntry entry{0,0,sizeof(uint32_t)};
    for(uint32_t phase=0;phase<s.pipelines.size();++phase){VkSpecializationInfo specialization{1,&entry,sizeof(phase),&phase};ci.stage.pSpecializationInfo=&specialization;
        auto result=vkCreateComputePipelines(device->logicalDevice,cache,1,&ci,nullptr,&s.pipelines[phase]);if(result!=VK_SUCCESS){vkDestroyShaderModule(device->logicalDevice,module,nullptr);check(result);}}
    vkDestroyShaderModule(device->logicalDevice,module,nullptr);trace<<"GPU ready"<<std::endl;
}
void GpuPhysics::reset(const std::vector<Vec3>& positions,const std::vector<Vec3>& velocities){auto& s=*impl_;if(positions.size()!=s.n||(!velocities.empty()&&velocities.size()!=s.n))throw std::runtime_error("GPU reset size mismatch");
    std::vector<F4> x,v;for(uint32_t i=0;i<s.n;++i){x.push_back(f4(positions[i],s.mesh.pinned[i]?0:1/s.mesh.mass[i]));v.push_back(f4(velocities.empty()?Vec3{}:velocities[i]));}
    auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);s.barrier(cmd);vkCmdFillBuffer(cmd,s.buffers[31].buffer,0,32,0);s.update(cmd,0,x.data(),x.size()*sizeof(F4));s.update(cmd,2,x.data(),x.size()*sizeof(F4));s.update(cmd,1,v.data(),v.size()*sizeof(F4));vkCmdFillBuffer(cmd,s.buffers[35].buffer,0,VK_WHOLE_SIZE,0);s.barrier(cmd);s.device->flushCommandBuffer(cmd,s.queue,true);}
void GpuPhysics::recordStep(VkCommandBuffer cmd,float dt,const std::vector<Vec3>& pins,const std::vector<Vec3>* guide,const TriangleCollider* body,const PhysicsConfig& config,VkQueryPool profile,uint32_t firstQuery,const GpuPhysics* sharedBody){
    auto timestamp=[&](uint32_t index){if(profile)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,profile,firstQuery+index);};
    timestamp(0);
    auto& s=*impl_;if(pins.size()!=s.n||(guide&&guide->size()!=s.n)||!std::isfinite(dt)||dt<=0)throw std::runtime_error("Invalid GPU step input");
    if(!std::isfinite(config.tetherScale)||config.tetherScale<1)throw std::runtime_error("Invalid GPU tether scale");
    s.barrier(cmd);std::vector<F4> points;for(auto v:pins)points.push_back(f4(v));s.update(cmd,3,points.data(),points.size()*sizeof(F4));
    if(guide){points.clear();for(auto v:*guide)points.push_back(f4(v));s.update(cmd,4,points.data(),points.size()*sizeof(F4));}
    Push p{};p.vertices=s.n;p.dt=dt;p.gravity=config.gravity;p.damping=config.dampingPerSecond;p.thickness=config.thickness;p.friction=config.friction;
    p.pad0=config.enableCollision?1:0;p.pad1=(captureContacts?1:0)|(cacheSelfContacts?0:2)|(config.gnnAcceleration?4:0);
    if(body&&config.enableCollision)p.edges=static_cast<uint32_t>(body->capsules.size());
    if(body&&config.enableCollision&&sharedBody){
        // Both actors receive the same time-stamped body. Copy device buffers after its
        // first solve, avoiding duplicate CPU skinning/BVH fitting and host upload.
        const auto& source=*sharedBody->impl_;s.barrier(cmd);
        const uint32_t slots[]={10,11,12,13,33,32};
        const size_t sizes[]={(body->current.size()+2*body->capsules.size())*sizeof(F4),(body->previous.size()+2*body->capsules.size())*sizeof(F4),body->triangles.size()*sizeof(uint32_t),source.bodyTree.size()*sizeof(Node),(body->current.size()+4*body->triangles.size()/3)*sizeof(F4),(5+body->current.size()+2*body->triangles.size())*sizeof(uint32_t)};
        for(uint32_t i=0;i<6;++i){if(sizes[i]>s.buffers[slots[i]].size)throw std::runtime_error("Shared body exceeds capacity");
            VkBufferCopy copy{0,0,sizes[i]};if(sizes[i])vkCmdCopyBuffer(cmd,source.buffers[slots[i]].buffer,s.buffers[slots[i]].buffer,1,&copy);}
        p.bodyCount=static_cast<uint32_t>(source.bodyTree.size());
        s.bodySurface=BodySurfaceTopology{}; // shared incidence replaces the cached local GPU table
    }
    else if(body&&config.enableCollision){if(body->current.size()+2*body->capsules.size()>Impl::bodyCapacity||body->capsules.size()>64||body->triangles.size()/3>Impl::triangleCapacity||body->previous.size()!=body->current.size())throw std::runtime_error("Invalid GPU body collider");for(auto i:body->triangles)if(i>=body->current.size())throw std::runtime_error("Invalid GPU body triangle");
        points.clear();for(auto v:body->current)points.push_back(f4(v));
        for(auto c:body->capsules){auto finite=[](Vec3 v){return std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z);};
            if(!std::isfinite(c.radius)||c.radius<=0||!finite(c.currentA)||!finite(c.currentB)||!finite(c.previousA)||!finite(c.previousB))throw std::runtime_error("Invalid GPU capsule");
            points.push_back(f4(c.currentA,c.radius));points.push_back(f4(c.currentB,c.radius));}
        s.update(cmd,10,points.data(),points.size()*sizeof(F4));points.clear();for(auto v:body->previous)points.push_back(f4(v));
        for(auto c:body->capsules){points.push_back(f4(c.previousA,c.radius));points.push_back(f4(c.previousB,c.radius));}s.update(cmd,11,points.data(),points.size()*sizeof(F4));
        if(s.bodySurface.vertices!=body->current.size()||s.bodySurface.triangles!=body->triangles){
            s.bodySurface.build(static_cast<uint32_t>(body->current.size()),body->triangles);auto topology=s.bodySurface.packed();s.update(cmd,32,topology.data(),topology.size()*sizeof(uint32_t));}
        s.refitBody(*body);s.update(cmd,12,body->triangles.data(),body->triangles.size()*sizeof(uint32_t));s.update(cmd,13,s.bodyTree.data(),s.bodyTree.size()*sizeof(Node));p.bodyCount=static_cast<uint32_t>(s.bodyTree.size());}
    s.barrier(cmd);s.boundPhase=~0u;vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,s.layout,0,1,&s.set,0,nullptr);
    if(body&&config.enableCollision&&!sharedBody&&!body->triangles.empty()){
        p.phase=15;p.start=static_cast<uint32_t>(body->current.size());s.dispatch(cmd,p,std::max(p.start,static_cast<uint32_t>(body->triangles.size()/3)));}
    p.phase=0;s.dispatch(cmd,p,s.n);vkCmdFillBuffer(cmd,s.buffers[6].buffer,0,VK_WHOLE_SIZE,0);vkCmdFillBuffer(cmd,s.buffers[9].buffer,0,VK_WHOLE_SIZE,0);s.barrier(cmd);
    const bool contactGuide=guide&&config.guideCompliance>=0&&config.enableCollision&&config.contactAwareGuide;
    if(contactGuide){p.phase=16;p.start=body?static_cast<uint32_t>(body->current.size()):0;s.dispatch(cmd,p,s.n);}
    timestamp(1);
    for(int iteration=0;iteration<config.iterations;++iteration){for(auto group:s.groups){auto type=uint32_t(s.constraints[group.first].rest.y);p.phase=1;p.start=group.first;p.alpha=(type==0?config.stretchCompliance:type==1?config.shearCompliance:config.bendCompliance)/(dt*dt);s.dispatch(cmd,p,group.second);}
        if(guide&&config.guideCompliance>=0){p.phase=2;p.start=0;p.guide=contactGuide?1.f:0.f;p.alpha=config.guideCompliance/(dt*dt);s.dispatch(cmd,p,s.patches);}
        if(config.enableTethers){p.phase=14;p.start=s.tetherStart;p.alpha=config.tetherScale;s.dispatch(cmd,p,s.n);}}
    timestamp(2);
    if(config.enableSelfCollision){
        vkCmdFillBuffer(cmd,s.buffers[30].buffer,0,VK_WHOLE_SIZE,0);
        VkBufferCopy copy{0,0,s.n*sizeof(F4)};vkCmdCopyBuffer(cmd,s.buffers[0].buffer,s.buffers[16].buffer,1,&copy);
        vkCmdFillBuffer(cmd,s.buffers[17].buffer,0,VK_WHOLE_SIZE,0);
        vkCmdFillBuffer(cmd,s.buffers[31].buffer,0,sizeof(uint32_t),0);s.barrier(cmd);
        if(cacheSelfContacts){p.phase=11;s.dispatch(cmd,p,s.n);
            p.phase=12;s.dispatch(cmd,p,s.n+static_cast<uint32_t>(s.tables.stretchRestLength().size()));}
        else{uint32_t rebuild=1;s.update(cmd,31,&rebuild,sizeof(rebuild));s.barrier(cmd);}
        vkCmdFillBuffer(cmd,s.buffers[23].buffer,0,VK_WHOLE_SIZE,0);vkCmdFillBuffer(cmd,s.buffers[25].buffer,0,VK_WHOLE_SIZE,0);s.barrier(cmd);
        p.phase=9;s.dispatch(cmd,p,static_cast<uint32_t>(s.mesh.triangles.size()/3));
        p.phase=10;s.dispatch(cmd,p,static_cast<uint32_t>(s.tables.stretchRestLength().size()));
        for(uint32_t kind=0;kind<2;++kind){p.guide=float(kind);p.phase=4;uint32_t size=static_cast<uint32_t>(kind?s.edgeTree.size():s.clothTree.size());s.dispatch(cmd,p,size);
            auto depth=kind?s.edgeDepth:s.clothDepth;for(uint32_t level=depth;level-->0;){p.phase=5;p.start=level;s.dispatch(cmd,p,size);}}
        timestamp(5);
        p.phase=6;p.treeCount=static_cast<uint32_t>(s.clothTree.size());s.dispatch(cmd,p,s.n*32);timestamp(6);
        p.phase=7;p.treeCount=static_cast<uint32_t>(s.edgeTree.size());s.dispatch(cmd,p,static_cast<uint32_t>(s.tables.stretchRestLength().size())*32);timestamp(7);
        if(cacheSelfContacts){p.phase=13;s.dispatch(cmd,p,s.n);}
        p.phase=8;p.start=static_cast<uint32_t>(s.tables.stretchRestLength().size());s.dispatch(cmd,p,s.n);
    }
    timestamp(3);
    p.phase=3;p.start=body?static_cast<uint32_t>(body->current.size()):0;s.dispatch(cmd,p,s.n);timestamp(4);
}
void GpuPhysics::recordCheckpoint(VkCommandBuffer cmd,uint64_t id){
    auto& s=*impl_;if(!id||s.checkpoints.count(id))return;
    auto& buffers=s.checkpoints[id];
    for(auto& b:buffers)check(s.device->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,&b,s.n*sizeof(F4)));
    s.barrier(cmd);VkBufferCopy copy{0,0,s.n*sizeof(F4)};
    for(uint32_t i=0;i<3;++i)vkCmdCopyBuffer(cmd,s.buffers[i==2?35:i].buffer,buffers[i].buffer,1,&copy);
    s.barrier(cmd);
}
void GpuPhysics::restoreCheckpoint(uint64_t id){
    auto& s=*impl_;auto found=s.checkpoints.find(id);if(found==s.checkpoints.end())throw std::runtime_error("GPU checkpoint is unavailable");
    auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);s.barrier(cmd);vkCmdFillBuffer(cmd,s.buffers[31].buffer,0,32,0);VkBufferCopy copy{0,0,s.n*sizeof(F4)};
    for(uint32_t i=0;i<3;++i)vkCmdCopyBuffer(cmd,found->second[i].buffer,s.buffers[i==2?35:i].buffer,1,&copy);
    vkCmdCopyBuffer(cmd,found->second[0].buffer,s.buffers[2].buffer,1,&copy);
    s.barrier(cmd);s.device->flushCommandBuffer(cmd,s.queue,true);
}
void GpuPhysics::retainCheckpoints(const std::vector<uint64_t>& ids){
    auto& s=*impl_;for(auto it=s.checkpoints.begin();it!=s.checkpoints.end();){
        if(std::find(ids.begin(),ids.end(),it->first)!=ids.end()){++it;continue;}
        for(auto& b:it->second)b.destroy();it=s.checkpoints.erase(it);
    }
}
const vks::Buffer& GpuPhysics::positionBuffer()const{return impl_->buffers[0];}
const vks::Buffer& GpuPhysics::previousPositionBuffer()const{return impl_->buffers[2];}
const vks::Buffer& GpuPhysics::velocityBuffer()const{return impl_->buffers[1];}
const vks::Buffer& GpuPhysics::accelerationBuffer()const{return impl_->buffers[35];}
std::array<uint32_t,7> GpuPhysics::contactDiagnostics(){auto& s=*impl_;std::array<uint32_t,7> result{};
    for(uint32_t index=0;index<3;++index){uint32_t slot=index==0?23:index==1?25:17;auto bytes=s.buffers[slot].size;vks::Buffer host;
        check(s.device->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&host,bytes));
        auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);s.barrier(cmd);VkBufferCopy copy{0,0,bytes};vkCmdCopyBuffer(cmd,s.buffers[slot].buffer,host.buffer,1,&copy);s.device->flushCommandBuffer(cmd,s.queue,true);check(host.map());
        auto data=static_cast<const uint32_t*>(host.mapped);uint32_t n=index<2?32768:s.n;
        for(uint32_t i=0;i<n;++i){result[index*2]=std::max(result[index*2],data[i]);if(index<2&&data[i]>512)++result[index*2+1];}
        if(index<2)result[6]+=data[n];else result[5]=data[n];host.destroy();}
    return result;
}
std::vector<Vec3> GpuPhysics::readback(){auto& s=*impl_;vks::Buffer host;check(s.device->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&host,s.n*sizeof(F4)));
    auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);s.barrier(cmd);VkBufferCopy copy{0,0,s.n*sizeof(F4)};vkCmdCopyBuffer(cmd,s.buffers[0].buffer,host.buffer,1,&copy);s.device->flushCommandBuffer(cmd,s.queue,true);check(host.map());std::vector<Vec3> result;auto* p=static_cast<const F4*>(host.mapped);for(uint32_t i=0;i<s.n;++i)result.push_back(xyz(p[i]));host.destroy();return result;}
}

namespace mlcloth::demo {
std::vector<SelfContactRecord> GpuPhysics::readContactRecords(){
    auto& s=*impl_;const auto bytes=s.buffers[30].size;vks::Buffer host;
    check(s.device->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&host,bytes));
    auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);s.barrier(cmd);VkBufferCopy copy{0,0,bytes};
    vkCmdCopyBuffer(cmd,s.buffers[30].buffer,host.buffer,1,&copy);s.device->flushCommandBuffer(cmd,s.queue,true);check(host.map());
    std::vector<SelfContactRecord> result(bytes/sizeof(SelfContactRecord));std::memcpy(result.data(),host.mapped,bytes);host.destroy();
    result.erase(std::remove_if(result.begin(),result.end(),[](const auto& r){return r.distance==0;}),result.end());return result;
}
}

namespace mlcloth::demo {
std::array<uint32_t,3> GpuPhysics::candidateCacheDiagnostics(){
    auto& s=*impl_;vks::Buffer host;const size_t bytes=3*sizeof(uint32_t);
    check(s.device->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&host,bytes));
    auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);s.barrier(cmd);VkBufferCopy copy{2*sizeof(uint32_t),0,bytes};
    vkCmdCopyBuffer(cmd,s.buffers[31].buffer,host.buffer,1,&copy);s.device->flushCommandBuffer(cmd,s.queue,true);check(host.map());
    std::array<uint32_t,3> result;std::memcpy(result.data(),host.mapped,bytes);host.destroy();return result;
}
}
