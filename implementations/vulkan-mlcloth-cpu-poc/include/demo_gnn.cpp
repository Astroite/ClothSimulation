#include "demo_gnn.h"
#include "fine15_gpu_layout.h"
#include <numeric>
#include <set>
#include <functional>
#include <fstream>

namespace mlcloth::demo {
namespace {
struct F4{float x,y,z,w;};struct U4{uint32_t x,y,z,w;};struct Tree{F4 lo,hi;U4 links;};
F4 f4(Vec3 p,float w=0){return {p.x,p.y,p.z,w};} Vec3 xyz(F4 p){return {p.x,p.y,p.z};}
void check(VkResult r){if(r!=VK_SUCCESS)throw std::runtime_error("GNN Vulkan error "+std::to_string(r));}
enum Slot{Weights,Table,Norms,Rest,Triangles,Offsets,Incident,Senders,Receivers,MeshCsr,
    Pins,Body,Future,Normals,TreeBuffer,Nearest,Active,Bits,NodeFeatures,MeshFeatures,DirectFeatures,InverseFeatures,
    Node0,Node1,Mesh0,Mesh1,Direct0,Direct1,Inverse0,Inverse1,Reverse,Begin,Count,MinEdge,Raw,ValidationX,ValidationV,History,HistoryInput,PreviousBody,TemporalWeights,HistoryCaptureX,HistoryCaptureV,SlotCount};
}
struct GpuGnn::Impl {
    vks::VulkanDevice* device{};VkQueue queue{};VkDescriptorPool pool{};uint32_t n{},m{},e{},blocks{},embedding{},words{};
    uint32_t capacity=8192;bool temporal{},currentOnly{};uint64_t ticks{};double lastTime=-1;
    std::vector<uint64_t> surfaceIds;
    struct Observation{uint64_t tick{};bool bodyValid{};std::vector<Vec3> pins,delta,normals;};
    std::array<Observation,4> observations;BodySurfaceFrame previousObservation;
    struct Snapshot{vks::Buffer history,previous;uint64_t ticks{};double time=-1;std::vector<uint64_t> ids;};
    std::map<uint64_t,Snapshot> snapshots;
    std::array<vks::Buffer,SlotCount> buffers{};
    struct Kernel{VkDescriptorSetLayout setLayout{};VkPipelineLayout layout{};VkPipeline pipeline{};std::vector<VkDescriptorSet> sets;};
    std::array<Kernel,8> kernels;
    const GpuPhysics* physics{};
    void barrier(VkCommandBuffer cmd){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&b,0,nullptr,0,nullptr);}
    void make(uint32_t slot,size_t bytes,const void* data=nullptr){bytes=std::max<size_t>(bytes,16);auto& b=buffers[slot];check(device->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,&b,bytes));
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);vkCmdFillBuffer(cmd,b.buffer,0,VK_WHOLE_SIZE,0);device->flushCommandBuffer(cmd,queue,true);
        if(data){vks::Buffer stage;check(device->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&stage,bytes,const_cast<void*>(data)));cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);VkBufferCopy c{0,0,bytes};vkCmdCopyBuffer(cmd,stage.buffer,b.buffer,1,&c);device->flushCommandBuffer(cmd,queue,true);stage.destroy();}}
    template<class T>void make(uint32_t slot,const std::vector<T>& v){if(v.empty()){make(slot,16);return;} // pad small host arrays too
        std::vector<uint8_t> padded(std::max<size_t>(16,v.size()*sizeof(T)),0);std::memcpy(padded.data(),v.data(),v.size()*sizeof(T));make(slot,padded.size(),padded.data());}
    template<class T>void upload(VkCommandBuffer cmd,uint32_t slot,const std::vector<T>& v){size_t bytes=v.size()*sizeof(T);if(bytes>buffers[slot].size)throw std::runtime_error("GNN input exceeds capacity");
        for(size_t offset=0;offset<bytes;offset+=65536)vkCmdUpdateBuffer(cmd,buffers[slot].buffer,offset,std::min<size_t>(65536,bytes-offset),reinterpret_cast<const uint8_t*>(v.data())+offset);}
    void kernel(uint32_t index,VkPipelineCache cache,const std::filesystem::path& path,const std::vector<std::vector<const vks::Buffer*>>& mappings){auto& k=kernels[index];std::vector<VkDescriptorSetLayoutBinding> bindings;
        for(uint32_t i=0;i<mappings[0].size();i++)bindings.push_back({i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr});
        VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=static_cast<uint32_t>(bindings.size());sl.pBindings=bindings.data();check(vkCreateDescriptorSetLayout(device->logicalDevice,&sl,nullptr,&k.setLayout));
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,32};VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.setLayoutCount=1;li.pSetLayouts=&k.setLayout;li.pushConstantRangeCount=1;li.pPushConstantRanges=&range;check(vkCreatePipelineLayout(device->logicalDevice,&li,nullptr,&k.layout));
        for(const auto& mapping:mappings){VkDescriptorSet set{};VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=pool;ai.descriptorSetCount=1;ai.pSetLayouts=&k.setLayout;check(vkAllocateDescriptorSets(device->logicalDevice,&ai,&set));k.sets.push_back(set);
            std::vector<VkWriteDescriptorSet> writes;for(uint32_t i=0;i<mapping.size();i++){VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w.dstSet=set;w.dstBinding=i;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.descriptorCount=1;w.pBufferInfo=&mapping[i]->descriptor;writes.push_back(w);}vkUpdateDescriptorSets(device->logicalDevice,static_cast<uint32_t>(writes.size()),writes.data(),0,nullptr);}
        auto data=vgnn::readFile(path);if(data.empty()||data.size()%4)throw std::runtime_error("Invalid GNN shader");std::vector<uint32_t> code(data.size()/4);std::memcpy(code.data(),data.data(),data.size());VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=data.size();mi.pCode=code.data();VkShaderModule module{};check(vkCreateShaderModule(device->logicalDevice,&mi,nullptr,&module));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.layout=k.layout;ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";auto r=vkCreateComputePipelines(device->logicalDevice,cache,1,&ci,nullptr,&k.pipeline);vkDestroyShaderModule(device->logicalDevice,module,nullptr);check(r);}
    void dispatch(VkCommandBuffer cmd,uint32_t kernel,uint32_t set,const void* push,uint32_t size,uint32_t groups){if(!groups)return;auto& k=kernels[kernel];vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,k.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,k.layout,0,1,&k.sets[set],0,nullptr);vkCmdPushConstants(cmd,k.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,size,push);vkCmdDispatch(cmd,groups,1,1);barrier(cmd);}
};
GpuGnn::GpuGnn():impl_(std::make_unique<Impl>()){}
GpuGnn::~GpuGnn(){auto& s=*impl_;if(!s.device)return;auto d=s.device->logicalDevice;for(auto& k:s.kernels){if(k.pipeline)vkDestroyPipeline(d,k.pipeline,nullptr);if(k.layout)vkDestroyPipelineLayout(d,k.layout,nullptr);if(k.setLayout)vkDestroyDescriptorSetLayout(d,k.setLayout,nullptr);}if(s.pool)vkDestroyDescriptorPool(d,s.pool,nullptr);for(auto& b:s.buffers)b.destroy();for(auto& [id,v]:s.snapshots){v.history.destroy();v.previous.destroy();}}
bool GpuGnn::ready()const{return impl_->kernels[5].pipeline!=VK_NULL_HANDLE;}
void GpuGnn::build(vks::VulkanDevice* device,VkQueue queue,VkPipelineCache cache,const std::filesystem::path& shaders,const std::filesystem::path& weights,const Mesh& mesh,const GpuPhysics& physics,bool temporal,uint32_t bodyCapacity){
    auto asset=vhood::loadTensorAsset(weights,temporal?"VTHOOD01":"VHOOD001");auto arch=vhood::inferTinyArchitecture(asset);if(arch.latent!=32)throw std::runtime_error("Demo GNN requires a 32-channel TinyHOOD model");auto model=vhood::buildTinyGpuModel(asset);
    auto& s=*impl_;if(s.device)throw std::runtime_error("GNN already built");s.device=device;s.queue=queue;s.physics=&physics;s.n=static_cast<uint32_t>(mesh.rest.size());s.words=(s.n+31)/32;s.blocks=arch.blocks;s.embedding=model.embeddingOffset;s.temporal=temporal;s.capacity=std::max(1u,bodyCapacity);
    if(temporal){
        auto c=vhood::tensorFloats(asset.require("temporal.contract",{9}));const float expected[9]={1,4,7,28,32,12,30,c[7],1};
        if(!std::equal(c.begin(),c.end(),expected)||(c[7]!=0&&c[7]!=1)||arch.blocks!=12)throw std::runtime_error("Unsupported temporal contract");s.currentOnly=c[7]!=0;
        std::vector<float> tw;auto append=[&](const char* name,uint32_t rows,uint32_t cols,bool transpose){const auto& view=cols==0?asset.require(name,{rows}):asset.require(name,{rows,cols});auto x=vhood::tensorFloats(view);
            for(float v:x)if(!std::isfinite(v))throw std::runtime_error("Nonfinite temporal weight");
            if(transpose){for(uint32_t c=0;c<cols;c++)for(uint32_t r=0;r<rows;r++)tw.push_back(x[r*cols+c]);}else tw.insert(tw.end(),x.begin(),x.end());};
        append("temporal.layers.0.weight",32,28,true);append("temporal.layers.0.bias",32,0,false);append("temporal.layers.2.weight",32,32,true);append("temporal.layers.2.bias",32,0,false);s.make(TemporalWeights,tw);
    }
    s.make(Weights,model.weights);s.make(Table,model.mlps);s.make(Norms,model.normalizers);
    std::vector<F4> rest;for(size_t i=0;i<mesh.rest.size();i++)rest.push_back(f4(mesh.rest[i],mesh.mass[i]));s.make(Rest,rest);s.make(Triangles,mesh.triangles);
    std::vector<std::vector<uint32_t>> incident(s.n);std::set<std::pair<uint32_t,uint32_t>> edges;
    for(uint32_t t=0;t<mesh.triangles.size()/3;t++)for(uint32_t k=0;k<3;k++){uint32_t a=mesh.triangles[t*3+k],b=mesh.triangles[t*3+(k+1)%3];incident[a].push_back(t);edges.insert({a,b});edges.insert({b,a});}
    std::vector<uint32_t> off{0},inc,send,recv,csr(s.n+1,0);std::vector<float> minEdge(s.n,1e30f);
    for(const auto& list:incident){inc.insert(inc.end(),list.begin(),list.end());off.push_back(static_cast<uint32_t>(inc.size()));}
    // pair first is receiver: stable receiver CSR, sender order inside each row.
    for(auto [b,a]:edges){send.push_back(a);recv.push_back(b);csr[b+1]++;minEdge[b]=std::min(minEdge[b],length(mesh.rest[a]-mesh.rest[b]));}
    std::partial_sum(csr.begin(),csr.end(),csr.begin());s.e=static_cast<uint32_t>(send.size());s.make(Offsets,off);s.make(Incident,inc);s.make(Senders,send);s.make(Receivers,recv);s.make(MeshCsr,csr);s.make(MinEdge,minEdge);
    s.make(Pins,s.n*sizeof(F4));for(auto slot:{Body,Future,Normals})s.make(slot,s.capacity*sizeof(F4));s.make(TreeBuffer,2*s.capacity*sizeof(Tree));
    s.make(Nearest,s.n*4);s.make(Active,s.capacity*4);s.make(Bits,s.capacity*s.words*4);
    s.make(NodeFeatures,(s.n+s.capacity)*20*4);s.make(MeshFeatures,s.e*12*4);s.make(DirectFeatures,s.n*9*4);s.make(InverseFeatures,s.n*9*4);
    for(auto slot:{Node0,Node1})s.make(slot,(s.n+s.capacity)*32*4);for(auto slot:{Mesh0,Mesh1})s.make(slot,s.e*32*4);for(auto slot:{Direct0,Direct1,Inverse0,Inverse1})s.make(slot,s.n*32*4);
    s.make(Reverse,s.n*4);s.make(Begin,s.capacity*4);s.make(Count,s.capacity*4);s.make(Raw,s.n*sizeof(F4));
    s.make(ValidationX,s.n*sizeof(F4));s.make(ValidationV,s.n*sizeof(F4));
    if(s.temporal){s.make(History,(s.n+s.capacity)*28*4);s.make(HistoryInput,(s.n+s.capacity)*28*4);s.make(PreviousBody,s.capacity*sizeof(F4));
        s.make(HistoryCaptureX,4*s.n*sizeof(F4));s.make(HistoryCaptureV,4*s.n*sizeof(F4));}
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,256};VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pi.maxSets=16;pi.poolSizeCount=1;pi.pPoolSizes=&ps;check(vkCreateDescriptorPool(device->logicalDevice,&pi,nullptr,&s.pool));
    auto b=[&](Slot slot){return &s.buffers[slot];};
    s.kernel(0,cache,shaders/"demo_gnn_features.comp.spv",{{b(Weights),b(Norms),&physics.positionBuffer(),&physics.velocityBuffer(),b(Pins),b(Rest),b(Triangles),b(Offsets),b(Incident),b(Senders),b(Receivers),b(Body),b(Future),b(Normals),b(TreeBuffer),b(Nearest),b(Active),b(Bits),b(NodeFeatures),b(MeshFeatures),b(DirectFeatures),b(InverseFeatures)}});
    s.kernel(1,cache,shaders/"demo_gnn_reverse.comp.spv",{{b(Bits),b(Count),b(Begin),b(Reverse)}});
    s.kernel(2,cache,shaders/"tinyhood_encode.comp.spv",{{b(Weights),b(Table),b(NodeFeatures),b(MeshFeatures),b(DirectFeatures),b(InverseFeatures),b(Node0),b(Mesh0),b(Direct0),b(Inverse0)}});
    s.kernel(3,cache,shaders/"tinyhood_edge_update.comp.spv",{{b(Weights),b(Table),b(Node0),b(Mesh0),b(Direct0),b(Inverse0),b(Mesh1),b(Direct1),b(Inverse1),b(Senders),b(Receivers),b(Nearest)},
        {b(Weights),b(Table),b(Node1),b(Mesh1),b(Direct1),b(Inverse1),b(Mesh0),b(Direct0),b(Inverse0),b(Senders),b(Receivers),b(Nearest)}});
    s.kernel(4,cache,shaders/"tinyhood_node_update.comp.spv",{{b(Weights),b(Table),b(Node0),b(Node1),b(Mesh0),b(Mesh1),b(Direct0),b(Direct1),b(Inverse0),b(Inverse1),b(MeshCsr),b(Nearest),b(Active),b(Reverse),b(Begin),b(Count)},
        {b(Weights),b(Table),b(Node1),b(Node0),b(Mesh1),b(Mesh0),b(Direct1),b(Direct0),b(Inverse1),b(Inverse0),b(MeshCsr),b(Nearest),b(Active),b(Reverse),b(Begin),b(Count)}});
    s.kernel(5,cache,shaders/"demo_gnn_decode.comp.spv",{{b(Weights),b(Table),b(Norms),b(s.blocks%2?Node1:Node0),b(MinEdge),&physics.accelerationBuffer(),b(Raw)}});
    if(s.temporal){s.kernel(6,cache,shaders/"temporal_history.comp.spv",{{&physics.velocityBuffer(),b(Body),b(PreviousBody),b(NodeFeatures),b(Norms),b(History),b(HistoryInput)}});
        s.kernel(7,cache,shaders/"temporal_encode.comp.spv",{{b(TemporalWeights),b(HistoryInput),b(Node0)}});}
}
void GpuGnn::record(VkCommandBuffer cmd,const std::vector<Vec3>& pins,const TriangleCollider& body,const TriangleCollider& future,float strength,float trust,float gravity){auto& s=*impl_;
    if(s.temporal){record(cmd,pins,sampleBodySurface(body,future,double(s.ticks)/30),strength,trust,gravity);return;}
    if(body.current.size()!=future.current.size()||body.capsules.size()!=future.capsules.size())throw std::runtime_error("Body shape mismatch");
    std::vector<F4> cur,next,normal;std::vector<Vec3> normals(body.current.size());
    for(size_t t=0;t<body.triangles.size();t+=3){auto a=body.triangles[t],b=body.triangles[t+1],c=body.triangles[t+2];auto n=cross(body.current[b]-body.current[a],body.current[c]-body.current[a]);normals[a]+=n;normals[b]+=n;normals[c]+=n;}
    for(size_t i=0;i<body.current.size();i++){cur.push_back(f4(body.current[i]));next.push_back(f4(future.current[i]));normal.push_back(f4(normalize(normals[i])));}
    // Analytic capsules also contribute network world-edge samples; no old proxy asset.
    for(size_t i=0;i<body.capsules.size();i++){const auto& a=body.capsules[i];const auto& b=future.capsules[i];auto axis=normalize(a.currentB-a.currentA),axisNext=normalize(b.currentB-b.currentA);
        Vec3 seed=std::abs(axis.x)<.8f?Vec3{1,0,0}:Vec3{0,0,1};auto u=normalize(cross(axis,seed)),v=cross(axis,u),un=normalize(cross(axisNext,seed)),vn=cross(axisNext,un);
        for(int ring=0;ring<5;ring++)for(int k=0;k<8;k++){float angle=float(k)*6.28318530718f/8,t=float(ring)/4;auto n=u*std::cos(angle)+v*std::sin(angle),nn=un*std::cos(angle)+vn*std::sin(angle);
            cur.push_back(f4(a.currentA*(1-t)+a.currentB*t+n*a.radius));next.push_back(f4(b.currentA*(1-t)+b.currentB*t+nn*b.radius));normal.push_back(f4(n));}}

    BodySurfaceFrame frame;frame.time=double(s.ticks)/30;for(size_t i=0;i<cur.size();i++){frame.ids.push_back(i);frame.positions.push_back(xyz(cur[i]));frame.targets.push_back(xyz(next[i]));frame.normals.push_back(xyz(normal[i]));}
    record(cmd,pins,frame,strength,trust,gravity);
}
void GpuGnn::record(VkCommandBuffer cmd,const std::vector<Vec3>& pins,const BodySurfaceFrame& frame,float strength,float trust,float gravity){auto& s=*impl_;
    frame.validate();if(!ready()||pins.size()!=s.n||!std::isfinite(strength)||strength<0||strength>1||!std::isfinite(trust)||trust<=0||!std::isfinite(gravity))throw std::runtime_error("Invalid GNN inference inputs");
    if(s.temporal&&s.ticks){if(frame.ids!=s.surfaceIds)throw std::runtime_error("Surface IDs changed without reset");
        if(std::abs(frame.time-s.lastTime-1.0/30)>1e-6)throw std::runtime_error("Temporal inference must advance exactly one 30 Hz tick");}
    s.surfaceIds=frame.ids;s.lastTime=frame.time;
    std::vector<F4> cur,next,normal,target;for(size_t i=0;i<frame.ids.size();i++){cur.push_back(f4(frame.positions[i]));next.push_back(f4(frame.targets.empty()?frame.positions[i]:frame.targets[i]));normal.push_back(f4(frame.normals[i]));}
    s.m=static_cast<uint32_t>(cur.size());if(s.m>s.capacity)throw std::runtime_error("GNN body proxy exceeds preprocessed capacity");for(auto p:pins)target.push_back(f4(p));
    std::vector<uint32_t> order(s.m);std::iota(order.begin(),order.end(),0);std::vector<Tree> tree;
    std::function<uint32_t(size_t,size_t)> build=[&](size_t first,size_t last){uint32_t id=static_cast<uint32_t>(tree.size());tree.push_back({});Vec3 lo{1e30f,1e30f,1e30f},hi{-1e30f,-1e30f,-1e30f};
        for(size_t i=first;i<last;i++){auto p=cur[order[i]];lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};}tree[id].lo=f4(lo);tree[id].hi=f4(hi);
        if(last-first==1){tree[id].links.z=order[first];tree[id].links.w=1;}else{auto extent=hi-lo;int axis=extent.y>extent.x?1:0;if(extent.z>(axis?extent.y:extent.x))axis=2;auto value=[&](uint32_t i){return axis==0?cur[i].x:axis==1?cur[i].y:cur[i].z;};size_t middle=(first+last)/2;
            std::nth_element(order.begin()+first,order.begin()+middle,order.begin()+last,[&](uint32_t a,uint32_t b){return value(a)==value(b)?a<b:value(a)<value(b);});build(first,middle);build(middle,last);}tree[id].links.y=static_cast<uint32_t>(tree.size());return id;};if(s.m)build(0,s.m);
    s.barrier(cmd);s.upload(cmd,Pins,target);s.upload(cmd,Body,cur);s.upload(cmd,Future,next);s.upload(cmd,Normals,normal);s.upload(cmd,TreeBuffer,tree);
    if(captureValidation){VkBufferCopy c{0,0,s.n*sizeof(F4)};vkCmdCopyBuffer(cmd,s.physics->positionBuffer().buffer,s.buffers[ValidationX].buffer,1,&c);vkCmdCopyBuffer(cmd,s.physics->velocityBuffer().buffer,s.buffers[ValidationV].buffer,1,&c);}
    if(captureValidation&&s.temporal){const auto slot=s.ticks%4;VkBufferCopy c{0,slot*s.n*sizeof(F4),s.n*sizeof(F4)};
        vkCmdCopyBuffer(cmd,s.physics->positionBuffer().buffer,s.buffers[HistoryCaptureX].buffer,1,&c);vkCmdCopyBuffer(cmd,s.physics->velocityBuffer().buffer,s.buffers[HistoryCaptureV].buffer,1,&c);
        auto& o=s.observations[slot];o.tick=s.ticks+1;o.pins=pins;o.normals=frame.normals;o.delta.assign(s.m,{});
        o.bodyValid=s.ticks==0||(frame.ids==s.previousObservation.ids&&std::abs(frame.time-s.previousObservation.time-1./30)<1e-6);
        if(s.ticks&&o.bodyValid)for(size_t i=0;i<s.m;++i)o.delta[i]=frame.positions[i]-s.previousObservation.positions[i];s.previousObservation=frame;
    }
    for(auto slot:{Bits,Active,Count,Direct0,Direct1,Inverse0,Inverse1})vkCmdFillBuffer(cmd,s.buffers[slot].buffer,0,VK_WHOLE_SIZE,0);s.barrier(cmd);
    uint32_t feature[8]={s.n,s.m,s.e,s.embedding,static_cast<uint32_t>(tree.size()),s.words,0,0};s.dispatch(cmd,0,0,feature,sizeof(feature),(std::max(s.n+s.m,s.e)+127)/128);
    for(uint32_t phase=0;phase<2;phase++){U4 push{phase,s.m,s.words,0};s.dispatch(cmd,1,0,&push,sizeof(push),s.m);}
    for(uint32_t kind=0;kind<4;kind++){U4 p{std::min(kind,2u),kind,kind==0?s.n+s.m:kind==1?s.e:s.n,kind==0?20u:kind==1?12u:9u};s.dispatch(cmd,2,0,&p,sizeof(p),p.z);}
    if(s.temporal){if(s.ticks>UINT32_MAX)throw std::runtime_error("Temporal tick counter exhausted");U4 h{s.n,s.m,static_cast<uint32_t>(s.ticks),s.currentOnly?1u:0u};s.dispatch(cmd,6,0,&h,sizeof(h),(s.n+s.m+127)/128);U4 e{s.n+s.m,0,0,0};s.dispatch(cmd,7,0,&e,sizeof(e),s.n+s.m);}
    ++s.ticks;
    for(uint32_t block=0;block<s.blocks;block++){for(uint32_t kind=0;kind<3;kind++){U4 p{block,kind,kind==0?s.e:s.n,s.n};s.dispatch(cmd,3,block%2,&p,sizeof(p),p.z);}U4 p{block,s.n+s.m,s.n,s.e};s.dispatch(cmd,4,block%2,&p,sizeof(p),p.y);}
    struct Decode{uint32_t n,decoder;float strength,trust,gravity;uint32_t pad[3];}decode{s.n,3+s.blocks*3,strength,trust,gravity,{0,0,0}};s.dispatch(cmd,5,0,&decode,sizeof(decode),s.n);
}
bool GpuGnn::temporal()const{return impl_->temporal;}
uint64_t GpuGnn::allocatedBytes()const{const auto& s=*impl_;uint64_t total=0;if(!s.device)return 0;
    auto add=[&](const vks::Buffer& b){if(b.buffer){VkMemoryRequirements r{};vkGetBufferMemoryRequirements(s.device->logicalDevice,b.buffer,&r);total+=r.size;}};
    for(const auto& b:s.buffers)add(b);for(const auto& item:s.snapshots){add(item.second.history);add(item.second.previous);}return total;
}
void GpuGnn::resetHistory(){auto& s=*impl_;s.ticks=0;s.lastTime=-1;s.surfaceIds.clear();
    s.observations={};s.previousObservation={};
    for(auto& [id,v]:s.snapshots){v.history.destroy();v.previous.destroy();}s.snapshots.clear();
    if(!s.temporal)return;auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);
    for(auto slot:{History,HistoryInput,PreviousBody})vkCmdFillBuffer(cmd,s.buffers[slot].buffer,0,VK_WHOLE_SIZE,0);
    s.barrier(cmd);s.device->flushCommandBuffer(cmd,s.queue,true);
}
void GpuGnn::recordCheckpoint(VkCommandBuffer cmd,uint64_t id){auto& s=*impl_;if(!s.temporal)return;auto& v=s.snapshots[id];
    auto copy=[&](vks::Buffer& dst,Slot slot){auto& src=s.buffers[slot];if(!dst.buffer)check(s.device->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,&dst,src.size));VkBufferCopy c{0,0,src.size};vkCmdCopyBuffer(cmd,src.buffer,dst.buffer,1,&c);};
    s.barrier(cmd);copy(v.history,History);copy(v.previous,PreviousBody);s.barrier(cmd);v.ticks=s.ticks;v.time=s.lastTime;v.ids=s.surfaceIds;
}
void GpuGnn::restoreCheckpoint(uint64_t id){auto& s=*impl_;if(!s.temporal)return;const auto it=s.snapshots.find(id);
    if(it==s.snapshots.end())throw std::runtime_error("Missing temporal checkpoint");const auto& v=it->second;
    auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);s.barrier(cmd);
    auto copy=[&](const vks::Buffer& src,Slot slot){VkBufferCopy c{0,0,src.size};vkCmdCopyBuffer(cmd,src.buffer,s.buffers[slot].buffer,1,&c);};copy(v.history,History);copy(v.previous,PreviousBody);s.barrier(cmd);s.device->flushCommandBuffer(cmd,s.queue,true);
    s.ticks=v.ticks;s.lastTime=v.time;s.surfaceIds=v.ids;
    s.observations={};s.previousObservation={};
}
void GpuGnn::retainCheckpoints(const std::vector<uint64_t>& ids){auto& s=*impl_;for(auto it=s.snapshots.begin();it!=s.snapshots.end();){
    if(std::find(ids.begin(),ids.end(),it->first)==ids.end()){it->second.history.destroy();it->second.previous.destroy();it=s.snapshots.erase(it);}else ++it;}}
void GpuGnn::dumpValidation(const std::filesystem::path& path){auto& s=*impl_;std::filesystem::create_directories(path);
    const std::vector<std::pair<const char*,std::pair<Slot,size_t>>> files={{"nodes.f32",{NodeFeatures,(s.n+s.m)*20*4}},{"mesh.f32",{MeshFeatures,s.e*12*4}},{"direct.f32",{DirectFeatures,s.n*9*4}},{"inverse.f32",{InverseFeatures,s.n*9*4}},{"senders.u32",{Senders,s.e*4}},{"receivers.u32",{Receivers,s.e*4}},{"nearest.u32",{Nearest,s.n*4}},{"reverse.u32",{Reverse,s.n*4}},{"begin.u32",{Begin,s.m*4}},{"count.u32",{Count,s.m*4}},{"raw.f32",{Raw,s.n*16}},
        {"x.f32",{ValidationX,s.n*16}},{"velocity.f32",{ValidationV,s.n*16}},{"pins.f32",{Pins,s.n*16}},{"rest.f32",{Rest,s.n*16}},
        {"body.f32",{Body,s.m*16}},{"future.f32",{Future,s.m*16}},{"normals.f32",{Normals,s.m*16}},{"triangles.u32",{Triangles,s.buffers[Triangles].size}},{"normalizers.f32",{Norms,s.buffers[Norms].size}}};
    for(auto& [name,data]:files){if(!data.second)continue;vks::Buffer host;check(s.device->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&host,data.second));auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);s.barrier(cmd);VkBufferCopy copy{0,0,data.second};vkCmdCopyBuffer(cmd,s.buffers[data.first].buffer,host.buffer,1,&copy);s.device->flushCommandBuffer(cmd,s.queue,true);check(host.map());std::ofstream out(path/name,std::ios::binary);out.write(static_cast<const char*>(host.mapped),data.second);host.destroy();}
    if(s.temporal){for(auto slot:{History,HistoryInput}){vks::Buffer host;size_t bytes=(s.n+s.m)*28*4;check(s.device->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&host,bytes));auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);s.barrier(cmd);VkBufferCopy c{0,0,bytes};vkCmdCopyBuffer(cmd,s.buffers[slot].buffer,host.buffer,1,&c);s.device->flushCommandBuffer(cmd,s.queue,true);check(host.map());std::ofstream f(path/(slot==History?"history.f32":"history-input.f32"),std::ios::binary);f.write(static_cast<const char*>(host.mapped),bytes);f.close();host.destroy();}
        for(auto slot:{HistoryCaptureX,HistoryCaptureV}){vks::Buffer host;const size_t bytes=4*s.n*sizeof(F4);check(s.device->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,&host,bytes));auto cmd=s.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);s.barrier(cmd);VkBufferCopy c{0,0,bytes};vkCmdCopyBuffer(cmd,s.buffers[slot].buffer,host.buffer,1,&c);s.device->flushCommandBuffer(cmd,s.queue,true);check(host.map());std::ofstream f(path/(slot==HistoryCaptureX?"history-x.f32":"history-v.f32"),std::ios::binary);f.write(static_cast<const char*>(host.mapped),bytes);f.close();host.destroy();}
        std::ofstream capture(path/"history-capture.txt");std::ofstream p(path/"history-pins.f32",std::ios::binary),d(path/"history-body-delta.f32",std::ios::binary),n(path/"history-body-normal.f32",std::ios::binary);
        for(const auto& o:s.observations){capture<<o.tick<<" "<<o.bodyValid<<"\n";auto put=[](std::ofstream& f,const std::vector<Vec3>& values,size_t count){std::vector<Vec3> copy=values;copy.resize(count);f.write(reinterpret_cast<const char*>(copy.data()),copy.size()*sizeof(Vec3));};put(p,o.pins,s.n);put(d,o.delta,s.m);put(n,o.normals,s.m);}
        std::ofstream(path/"temporal.txt")<<s.ticks<<" "<<s.currentOnly<<"\n";}
    std::ofstream(path/"shape.txt")<<s.n<<" "<<s.m<<" "<<s.e<<" "<<s.blocks<<"\n";
}
}
