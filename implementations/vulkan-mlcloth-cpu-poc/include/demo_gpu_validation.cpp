#include "demo_gpu_validation.h"
#include <algorithm>
namespace mlcloth::demo {
nlohmann::json validateGpuPhysics(vks::VulkanDevice* device,VkQueue queue,VkPipelineCache cache,const std::filesystem::path& shader){
    nlohmann::json report;report["cases"]=nlohmann::json::array();bool passed=true;
    auto require=[&](const char* name,double error,double limit){bool pass=std::isfinite(error)&&error<=limit;passed&=pass;report["cases"].push_back({{"name",name},{"error",error},{"limit",limit},{"pass",pass}});};
    Mesh mesh;mesh.rest={{0,1,0},{.1f,1,0},{0,1,.1f},{.1f,1,.1f}};mesh.triangles={0,2,1,1,2,3};mesh.mass={.01f,.01f,.01f,.01f};mesh.pinned={0,0,0,0};
    GpuPhysics gpu;gpu.build(device,queue,cache,shader,mesh);PhysicsSolver cpu;cpu.build(mesh);
    PhysicsConfig config;config.dampingPerSecond=.7f;config.stretchCompliance=1e-7f;config.shearCompliance=1e-6f;config.bendCompliance=1e-4f;
    auto step=[&](float dt,const std::vector<Vec3>& pins,const std::vector<Vec3>* guide,const TriangleCollider* body){
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);gpu.recordStep(cmd,dt,pins,guide,body,config);device->flushCommandBuffer(cmd,queue,true);};
    auto maxDifference=[&](const std::vector<Vec3>& a,const std::vector<Vec3>& b){double error=0;for(size_t i=0;i<a.size();++i)error=std::max(error,double(length(a[i]-b[i])));return error;};
    cpu.reset(mesh.rest);gpu.reset(mesh.rest);
    for(int i=0;i<24;++i){step(1.f/240,mesh.rest,nullptr,nullptr);cpu.step(1.f/240,mesh.rest,nullptr,nullptr,config);}
    require("free fall CPU/GPU 24 steps metres",maxDifference(gpu.readback(),cpu.positions()),1e-5);
    auto deformed=mesh.rest;deformed[3].y+=.03f;gpu.reset(deformed);cpu.reset(deformed);config.gravity=0;
    step(1.f/240,mesh.rest,nullptr,nullptr);cpu.step(1.f/240,mesh.rest,nullptr,nullptr,config);
    require("deformed single step CPU/GPU metres",maxDifference(gpu.readback(),cpu.positions()),2e-5);
    {
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);gpu.recordCheckpoint(cmd,1);device->flushCommandBuffer(cmd,queue,true);
        for(int i=0;i<8;++i)step(1.f/240,mesh.rest,nullptr,nullptr);auto expected=gpu.readback();
        gpu.restoreCheckpoint(1);for(int i=0;i<8;++i)step(1.f/240,mesh.rest,nullptr,nullptr);
        require("device checkpoint restores position and velocity",maxDifference(expected,gpu.readback()),1e-7);
        gpu.retainCheckpoints({});
    }
    gpu.reset(mesh.rest);config.guideCompliance=.001f;auto guide=mesh.rest;for(auto& v:guide)v.x+=.01f;
    step(1.f/240,mesh.rest,&guide,nullptr);auto guided=gpu.readback();
    require("coarse guide moves toward target",std::max(0.,double(.00001f-guided[0].x)),1e-7);
    require("coarse guide retains relative shape",length((guided[1]-guided[0])-(mesh.rest[1]-mesh.rest[0])),1e-5);
    config.guideCompliance=-1;config.enableCollision=true;config.friction=.4f;
    TriangleCollider body;body.previous={{-1,.9f,-1},{-1,.9f,1},{1,.9f,-1},{1,.9f,1}};body.current=body.previous;for(auto& v:body.current)v.y=1.02f;body.triangles={0,1,2,2,1,3};
    gpu.reset(mesh.rest);step(1.f/240,mesh.rest,nullptr,&body);auto contact=gpu.readback();double penetration=0;for(auto v:contact)penetration=std::max(penetration,double(1.023f-v.y));
    require("swept body plane penetration metres",penetration,1e-5);
    {
        GpuPhysics shared;shared.build(device,queue,cache,shader,mesh);shared.reset(mesh.rest);gpu.reset(mesh.rest);
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);
        gpu.recordStep(cmd,1.f/240,mesh.rest,nullptr,&body,config);
        shared.recordStep(cmd,1.f/240,mesh.rest,nullptr,&body,config,VK_NULL_HANDLE,0,&gpu);
        device->flushCommandBuffer(cmd,queue,true);
        require("shared synchronized body retains swept collision",maxDifference(shared.readback(),gpu.readback()),1e-7);
    }
    {
        Mesh m;m.rest={{0,1,.2f},{1,1,0},{1,1,.1f}};m.triangles={0,1,2};m.mass={1,1,1};m.pinned={0,1,1};
        GpuPhysics solver;solver.build(device,queue,cache,shader,m);PhysicsSolver reference;reference.build(m);
        PhysicsConfig c;c.gravity=0;c.iterations=0;c.dampingPerSecond=0;c.enableCollision=true;
        TriangleCollider collider;collider.capsules={{{0,.8f,0},{0,1.2f,0},{0,.8f,0},{0,1.2f,0},.05f}};
        auto advance=[&](){auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,nullptr,&collider,c);device->flushCommandBuffer(cmd,queue,true);reference.step(1.f/240,m.rest,nullptr,&collider,c);};
        auto reset=[&](Vec3 position,Vec3 velocity=Vec3{}){auto points=m.rest;points[0]=position;std::vector<Vec3> speeds(3);speeds[0]=velocity;solver.reset(points,speeds);reference.reset(points,speeds);};
        reset(m.rest[0],{0,0,-96});advance();
        require("capsule fast crossing CPU/GPU metres",maxDifference(solver.readback(),reference.positions()),1e-6);
        require("capsule fast crossing arrival side",std::max(0.,double(.053f-solver.readback()[0].z)),1e-5);
        const auto crossing=solver.readback()[0];const Vec3 shift{.4f,.3f,-.2f};
        collider.capsules[0].currentA+=shift;collider.capsules[0].currentB+=shift;
        reset(m.rest[0],Vec3{0,0,-96}+shift*240);advance();
        require("common translation preserves swept capsule",length(solver.readback()[0]-shift-crossing),1e-5);
        collider.capsules[0].currentA=collider.capsules[0].previousA;collider.capsules[0].currentB=collider.capsules[0].previousB;
        reset({0,1,.01f});advance();auto recovered=solver.readback();advance();
        require("capsule embedded recovery retains zero velocity",maxDifference(recovered,solver.readback()),1e-6);
        require("capsule embedded recovery CPU/GPU metres",maxDifference(solver.readback(),reference.positions()),1e-6);
        collider.capsules={{{-.1f,.8f,0},{-.1f,1.2f,0},{.1f,.8f,0},{.1f,1.2f,0},.03f}};
        reset({0,1,0});advance();
        require("moving capsule swept CPU/GPU metres",maxDifference(solver.readback(),reference.positions()),1e-6);
        require("moving capsule pushes particle beyond final surface",std::max(0.,double(.133f-solver.readback()[0].x)),1e-5);
        // Exercise the endpoint offset after a triangle mesh and synchronized device copy.
        collider.current={{-1,.5f,-1},{-1,.5f,1},{1,.5f,-1}};collider.previous=collider.current;collider.triangles={0,1,2};
        reset({0,1,0});advance();
        require("STM and capsule packed together CPU/GPU metres",maxDifference(solver.readback(),reference.positions()),1e-6);
        GpuPhysics shared;shared.build(device,queue,cache,shader,m);auto points=m.rest;points[0]={0,1,0};shared.reset(points);
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);shared.recordStep(cmd,1.f/240,m.rest,nullptr,&collider,c,VK_NULL_HANDLE,0,&solver);device->flushCommandBuffer(cmd,queue,true);
        require("shared STM and capsules retain identical collision",maxDifference(solver.readback(),shared.readback()),1e-6);
        collider.current.clear();collider.previous.clear();collider.triangles.clear();
        collider.capsules={{{-.2f,.01f,0},{.2f,.01f,0},{-.2f,.01f,0},{.2f,.01f,0},.05f}};
        reset({0,.002f,0});advance();advance();
        require("capsule embedded recovery and floor CPU/GPU metres",maxDifference(solver.readback(),reference.positions()),1e-6);
    }
    {
        Mesh m;m.rest={{-.1f,1,0},{.1f,1,0},{-.1f,1,.1f},{-.02f,1.0005f,.03f}};
        m.triangles={0,2,1};m.mass={1,1,1,1};m.pinned={1,1,1,0};
        GpuPhysics solver;solver.build(device,queue,cache,shader,m);solver.reset(m.rest);
        PhysicsConfig self;self.gravity=0;self.iterations=0;self.enableSelfCollision=true;self.dampingPerSecond=0;
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,nullptr,nullptr,self);device->flushCommandBuffer(cmd,queue,true);
        auto solved=solver.readback();require("self VF separation metres",std::max(0.,double(1.003f-solved[3].y)),1e-5);
        PhysicsSolver reference;reference.build(m);reference.reset(m.rest);reference.step(1.f/240,m.rest,nullptr,nullptr,self);
        require("self VF CPU/GPU metres",maxDifference(solved,reference.positions()),1e-6);
        require("self VF pins invariant",maxDifference({solved[0],solved[1],solved[2]},{m.rest[0],m.rest[1],m.rest[2]}),1e-7);
        auto crossing=m.rest;crossing[3].y=1.05f;std::vector<Vec3> velocity(4);velocity[3].y=-24;
        solver.reset(crossing,velocity);reference.reset(crossing,velocity);
        cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,nullptr,nullptr,self);device->flushCommandBuffer(cmd,queue,true);
        reference.step(1.f/240,m.rest,nullptr,nullptr,self);auto swept=solver.readback();
        require("swept self VF separation metres",std::max(0.,double(1.003f-swept[3].y)),1e-5);
        require("swept self VF CPU/GPU metres",maxDifference(swept,reference.positions()),1e-6);
    }
    {
        Mesh m;m.rest={{-.1f,1,0},{.1f,1,0},{-.1f,1,.01f},{0,1.001f,-.1f},{0,1.001f,.1f},{.01f,1.001f,-.1f}};
        m.triangles={0,1,2,3,4,5};m.mass.assign(6,1);m.pinned.assign(6,0);
        GpuPhysics solver;solver.build(device,queue,cache,shader,m);solver.reset(m.rest);
        PhysicsConfig self;self.gravity=0;self.iterations=0;self.enableSelfCollision=true;self.dampingPerSecond=0;
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,nullptr,nullptr,self);device->flushCommandBuffer(cmd,queue,true);
        auto solved=solver.readback();double gap=(solved[3].y+solved[4].y-solved[0].y-solved[1].y)*.5;
        require("non-adjacent self EE increases separation",std::max(0.,.0015-gap),1e-5);
        PhysicsSolver reference;reference.build(m);reference.reset(m.rest);reference.step(1.f/240,m.rest,nullptr,nullptr,self);
        require("self EE CPU/GPU metres",maxDifference(solved,reference.positions()),1e-6);
    }
    {
        Mesh m;m.rest={{-.1f,1.05f,-.1f},{.1f,1.05f,-.1f},{0,1.05f,.1f}};m.triangles={0,1,2};m.mass.assign(3,1);m.pinned.assign(3,0);
        TriangleCollider box;box.current={{-1,1,-1},{1,1,-1},{1,1,1},{-1,1,1},{-1,2,-1},{1,2,-1},{1,2,1},{-1,2,1}};box.previous=box.current;
        box.triangles={0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,3,2,6,3,6,7,0,3,7,0,7,4,1,5,6,1,6,2};
        GpuPhysics solver;solver.build(device,queue,cache,shader,m);solver.reset(m.rest);PhysicsSolver reference;reference.build(m);reference.reset(m.rest);
        PhysicsConfig contact;contact.gravity=0;contact.iterations=0;contact.enableCollision=true;contact.dampingPerSecond=0;
        auto advance=[&](){auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,nullptr,&box,contact);device->flushCommandBuffer(cmd,queue,true);};
        advance();reference.step(1.f/240,m.rest,nullptr,&box,contact);auto recovered=solver.readback();
        require("embedded particle recovery CPU/GPU metres",maxDifference(recovered,reference.positions()),1e-6);
        double depth=0;for(auto v:recovered)depth=std::max(depth,double(v.y-.997f));require("embedded particles leave closed box",depth,1e-5);
        advance();require("embedded recovery does not inject launch velocity",maxDifference(recovered,solver.readback()),1e-6);
    }
    {
        Mesh m;m.rest={{.1f,1,.1f},{2,1,2},{2.1f,1,2}};m.triangles={0,1,2};m.mass={1,1,1};m.pinned={0,1,1};
        TriangleCollider corner;corner.current={{-1,.5f,-1},{1,.5f,-1},{1,.5f,0},{0,.5f,0},{0,.5f,1},{-1,.5f,1}};
        for(int i=0;i<6;++i){auto p=corner.current[i];p.y=1.5f;corner.current.push_back(p);}corner.previous=corner.current;
        const uint32_t cap[]={0,1,3,1,2,3,0,3,5,3,4,5};
        for(int i=0;i<12;i+=3)corner.triangles.insert(corner.triangles.end(),{cap[i],cap[i+1],cap[i+2],cap[i]+6,cap[i+2]+6,cap[i+1]+6});
        for(uint32_t a=0;a<6;++a){auto b=(a+1)%6;corner.triangles.insert(corner.triangles.end(),{a,a+6,b,b,a+6,b+6});}
        PhysicsConfig c;c.gravity=0;c.iterations=0;c.dampingPerSecond=0;c.enableCollision=true;
        GpuPhysics solver;solver.build(device,queue,cache,shader,m);PhysicsSolver reference;reference.build(m);std::vector<Vec3> speed(3);speed[0]={-48,0,-48};
        solver.reset(m.rest,speed);reference.reset(m.rest,speed);
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,nullptr,&corner,c);device->flushCommandBuffer(cmd,queue,true);
        reference.step(1.f/240,m.rest,nullptr,&corner,c);auto result=solver.readback();
        require("concave body corner swept CPU/GPU metres",maxDifference(result,reference.positions()),2e-6);
        require("concave body corner exits both faces",std::max(0.,double(.003f-std::min(result[0].x,result[0].z))),1e-5);
    }
    {
        Mesh m;m.rest={{-.1f,1,-.1f},{.1f,1,-.1f},{0,1,.1f},{0,1.05f,0}};m.triangles={0,2,1};m.mass.assign(4,1);m.pinned={1,1,1,0};
        GpuPhysics baseline,translated;baseline.build(device,queue,cache,shader,m);translated.build(device,queue,cache,shader,m);
        PhysicsConfig c;c.gravity=0;c.iterations=0;c.dampingPerSecond=0;c.enableSelfCollision=true;
        std::vector<Vec3> speed(4);speed[3]={0,-24,0};baseline.reset(m.rest,speed);const Vec3 shift{.4f,.3f,-.2f};speed[3]+=shift*240;translated.reset(m.rest,speed);
        auto pins=m.rest;for(auto& p:pins)p+=shift;
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);baseline.recordStep(cmd,1.f/240,m.rest,nullptr,nullptr,c);translated.recordStep(cmd,1.f/240,pins,nullptr,nullptr,c);device->flushCommandBuffer(cmd,queue,true);
        auto result=translated.readback();for(auto& p:result)p-=shift;
        require("common translation preserves swept self VF",maxDifference(result,baseline.readback()),1e-5);
        require("relative bound retains swept self VF arrival side",std::max(0.,double(1.003f-result[3].y)),1e-5);
    }
    {
        // Cached broadphase must preserve incoming contacts, root translation and replay.
        Mesh m;m.rest={{-.1f,1,-.1f},{.1f,1,-.1f},{0,1,.1f},{0,1.04f,0}};
        m.triangles={0,2,1};m.mass.assign(4,1);m.pinned={1,1,1,0};
        GpuPhysics cached,uncached;cached.cacheSelfContacts=true;cached.build(device,queue,cache,shader,m);uncached.build(device,queue,cache,shader,m);
        uncached.cacheSelfContacts=false;std::vector<Vec3> velocity(4);velocity[3].y=-.4f;
        cached.reset(m.rest,velocity);uncached.reset(m.rest,velocity);
        PhysicsConfig self;self.gravity=0;self.iterations=0;self.enableSelfCollision=true;self.dampingPerSecond=0;
        double error=0;
        for(int i=0;i<48;++i){auto pins=m.rest;for(auto& v:pins)v.x=.0005f*i+v.x;
            auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);
            cached.recordStep(cmd,1.f/240,pins,nullptr,nullptr,self);uncached.recordStep(cmd,1.f/240,pins,nullptr,nullptr,self);
            if(i==12)cached.recordCheckpoint(cmd,7);device->flushCommandBuffer(cmd,queue,true);
            error=std::max(error,maxDifference(cached.readback(),uncached.readback()));}
        require("candidate cache preserves incoming swept contacts",error,2e-6);
        auto expected=cached.readback();cached.restoreCheckpoint(7);
        for(int i=13;i<48;++i){auto pins=m.rest;for(auto& v:pins)v.x=.0005f*i+v.x;
            auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);cached.recordStep(cmd,1.f/240,pins,nullptr,nullptr,self);device->flushCommandBuffer(cmd,queue,true);}
        require("checkpoint invalidates derived candidate cache",maxDifference(expected,cached.readback()),2e-6);
    }
    {
        Mesh m;for(uint32_t i=0;i<150;++i){float y=1+float(i)*.00001f;uint32_t v=static_cast<uint32_t>(m.rest.size());
            m.rest.insert(m.rest.end(),{{-.1f,y,-.1f},{.1f,y,-.1f},{0,y,.1f}});m.triangles.insert(m.triangles.end(),{v,v+2,v+1});}
        m.rest.push_back({0,1.002f,0});m.mass.assign(m.rest.size(),1);m.pinned.assign(m.rest.size(),1);m.pinned.back()=0;
        GpuPhysics cached,uncached;cached.cacheSelfContacts=true;cached.build(device,queue,cache,shader,m);uncached.build(device,queue,cache,shader,m);
        cached.reset(m.rest);uncached.reset(m.rest);PhysicsConfig self;self.gravity=0;self.iterations=0;self.enableSelfCollision=true;self.dampingPerSecond=1000;
        double error=0;for(int i=0;i<3;++i){auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);
            cached.recordStep(cmd,1.f/240,m.rest,nullptr,nullptr,self);uncached.recordStep(cmd,1.f/240,m.rest,nullptr,nullptr,self);
            device->flushCommandBuffer(cmd,queue,true);error=std::max(error,maxDifference(cached.readback(),uncached.readback()));}
        auto diagnostics=cached.candidateCacheDiagnostics();
        require("candidate overflow uses current BVH",error,2e-6);
        require("candidate overflow fixture exercised reuse and fallback",diagnostics[1]>0&&diagnostics[2]>0?0:1,0);
    }
    {
        Mesh m;m.rest={{0,1,0},{1,1,0},{1,2,0},{0,2,0},{0,2,.1f},{3,1,0},{3.1f,1,0},{3,1,.1f}};
        m.triangles={0,1,2,2,3,4,5,6,7};m.mass.assign(8,1);m.pinned={1,0,0,0,0,0,0,0};
        PhysicsSolver reference;reference.build(m);GpuPhysics solver;solver.build(device,queue,cache,shader,m);
        PhysicsConfig c;c.gravity=0;c.dampingPerSecond=0;c.iterations=1;c.enableTethers=true;c.tetherScale=1;
        c.stretchCompliance=c.shearCompliance=c.bendCompliance=1e20f;
        const float path=std::sqrt(2.f)+std::sqrt(1.01f);
        for(bool extended:{false,true}){
            auto points=m.rest;points[4]={0,1,path+(extended?1.f:-.1f)};points[5].y=2;
            const Vec3 shift{.4f,.3f,-.2f};auto pins=m.rest;for(auto& v:pins)v+=shift;for(auto& v:points)v+=shift;
            reference.reset(points);solver.reset(points);reference.step(1.f/240,pins,nullptr,nullptr,c);
            auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,pins,nullptr,nullptr,c);device->flushCommandBuffer(cmd,queue,true);
            auto result=solver.readback();require(extended?"extended geodesic tether CPU/GPU":"slack geodesic tether CPU/GPU",maxDifference(result,reference.positions()),2e-6);
            require(extended?"tether caps extension without ML":"tether preserves folded slack",std::abs(double(length(result[4]-pins[0])-(extended?path:path-.1f))),2e-6);
            require("tether leaves disconnected island free",length(result[5]-points[5]),1e-6);
        }
    }
    for(bool vertex:{false,true}){
        TriangleCollider body;
        if(vertex){body.current={{0,1,0},{-1,1.2f,.2f},{-1,.8f,.2f},{-1,1,-.2f}};body.triangles={0,2,3,0,1,2,0,3,1,1,3,2};}
        else{body.current={{0,1,-.2f},{-1,1.2f,-.2f},{-1,.8f,-.2f},{0,1,.2f},{-1,1.2f,.2f},{-1,.8f,.2f}};
            body.triangles={0,3,2,2,3,5,0,1,3,1,4,3,1,2,4,2,5,4,0,2,1,3,4,5};}
        Vec3 center{};for(auto p:body.current)center+=p;center=center/static_cast<float>(body.current.size());
        for(size_t t=0;t<body.triangles.size();t+=3){auto a=body.current[body.triangles[t]],b=body.current[body.triangles[t+1]],c=body.current[body.triangles[t+2]];
            if(dot(cross(b-a,c-a),(a+b+c)/3.f-center)<0)std::swap(body.triangles[t+1],body.triangles[t+2]);}
        body.previous=body.current;const Vec3 point{.04f,1.1f,vertex?.03f:0};
        Mesh m;m.rest={point,{2,1,0},{2,1,.1f}};m.triangles={0,1,2};m.mass={1,1,1};m.pinned={0,1,1};
        PhysicsSolver reference;reference.build(m);GpuPhysics solver;solver.build(device,queue,cache,shader,m);solver.reset(m.rest);
        PhysicsConfig c;c.iterations=0;c.gravity=0;c.enableCollision=true;reference.step(1.f/240,m.rest,nullptr,&body,c);
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,nullptr,&body,c);device->flushCommandBuffer(cmd,queue,true);
        auto result=solver.readback();require(vertex?"vertex pseudonormal preserves outside point":"edge pseudonormal preserves outside point",length(result[0]-point),1e-6);
        require(vertex?"vertex feature CPU/GPU":"edge feature CPU/GPU",maxDifference(result,reference.positions()),1e-6);
    }
    {
        TriangleCollider body;body.current={{-.3f,.7f,-.02f},{.3f,.7f,-.02f},{.3f,1.3f,-.02f},{-.3f,1.3f,-.02f},
            {-.3f,.7f,.02f},{.3f,.7f,.02f},{.3f,1.3f,.02f},{-.3f,1.3f,.02f}};body.previous=body.current;
        body.triangles={0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5};
        Mesh m;m.rest={{0,1,.08f},{2,1,0},{2,1,.1f}};m.triangles={0,1,2};m.mass={1,1,1};m.pinned={0,1,1};
        PhysicsSolver reference;reference.build(m);GpuPhysics solver;solver.build(device,queue,cache,shader,m);
        std::vector<Vec3> velocity(3);velocity[0].z=-23.76f;solver.reset(m.rest,velocity);reference.reset(m.rest,velocity);
        PhysicsConfig c;c.iterations=0;c.gravity=0;c.dampingPerSecond=0;c.enableCollision=true;reference.step(1.f/240,m.rest,nullptr,&body,c);
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,nullptr,&body,c);device->flushCommandBuffer(cmd,queue,true);
        auto result=solver.readback();require("earliest body impact retains entrance side",std::max(0.,double(.023f-result[0].z)),1e-5);
        require("earliest body impact CPU/GPU",maxDifference(result,reference.positions()),1e-6);
    }
    for(int kind=0;kind<3;++kind){
        const float surface=kind==0?0.f:1.f;const std::string label=kind==0?"ground":kind==1?"STM":"capsule";
        Mesh m;m.rest={{0,surface+.009f,0},{.001f,surface+.009f,0},{0,surface+.009f,.001f}};
        m.triangles={0,2,1};m.mass={.01f,.01f,.01f};m.pinned={0,0,0};
        TriangleCollider body;if(kind==1){body.current={{-1,1,-1},{-1,1,1},{1,1,-1},{1,1,1}};body.previous=body.current;body.triangles={0,1,2,2,1,3};}
        if(kind==2)body.capsules={{{-1,.94f,0},{1,.94f,0},{-1,.94f,0},{1,.94f,0},.06f}};
        GpuPhysics solver;solver.build(device,queue,cache,shader,m);PhysicsSolver reference;reference.build(m);
        PhysicsConfig c;c.gravity=0;c.dampingPerSecond=0;c.enableCollision=true;c.guideCompliance=.001f;
        for(Vec3 shift:{Vec3{0,-.02f,0},Vec3{0,.02f,0},Vec3{.02f,0,0},Vec3{0,.03f,0}}){
            auto guide=m.rest;for(auto& v:guide)v+=shift;if(shift.y==.03f)guide[0].y-=.06f;
            std::vector<Vec3> results[2];
            for(int enabled=0;enabled<2;++enabled){c.contactAwareGuide=enabled!=0;solver.reset(m.rest);reference.reset(m.rest);
                auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,&guide,kind?&body:nullptr,c);device->flushCommandBuffer(cmd,queue,true);
                reference.step(1.f/240,m.rest,&guide,kind?&body:nullptr,c);results[enabled]=solver.readback();
                require((label+" guide "+(shift.y<0?"inward":shift.y>0?"outward":"tangent")+(enabled?" aware":" baseline")+" CPU/GPU").c_str(),maxDifference(results[enabled],reference.positions()),1e-6);}
            if(shift.y<0)require((label+" weakens inward guide before contact").c_str(),std::max(0.,double(results[0][0].y+1e-5f-results[1][0].y)),1e-7);
            else require((label+" preserves outward or tangent guide").c_str(),maxDifference(results[0],results[1]),1e-6);
        }
    }
    {
        Mesh m;m.rest={{-.1f,1,0},{.1f,1,0},{-.1f,1,.2f}};m.triangles={0,2,1};m.mass={1,1,1};m.pinned={0,1,1};
        for(uint32_t i=0;i<300;++i){m.rest.push_back({-.095f+float(i%10)*.0001f,1.0005f,.005f+float(i/10)*.0001f});m.mass.push_back(1);m.pinned.push_back(0);}
        GpuPhysics solver;solver.build(device,queue,cache,shader,m);PhysicsSolver reference;reference.build(m);
        PhysicsConfig c;c.gravity=0;c.dampingPerSecond=0;c.iterations=0;c.enableSelfCollision=true;
        auto advance=[&](){auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,nullptr,nullptr,c);device->flushCommandBuffer(cmd,queue,true);};
        solver.reset(m.rest);advance();reference.step(1.f/240,m.rest,nullptr,nullptr,c);auto expected=solver.readback();
        require("self gather overflow retains all contacts CPU/GPU",maxDifference(expected,reference.positions()),1e-6);
        for(int repeat=0;repeat<3;++repeat){solver.reset(m.rest);advance();require("dense self gather repeats exactly",maxDifference(expected,solver.readback()),0);}
    }
    {
        Mesh m;for(uint32_t i=0;i<513;++i){uint32_t v=static_cast<uint32_t>(m.rest.size());m.rest.insert(m.rest.end(),{{-.5f,1,-.5f},{-.5f,1,.5f},{.5f,1,-.5f}});
            m.triangles.insert(m.triangles.end(),{v,v+1,v+2});m.mass.insert(m.mass.end(),3,1);m.pinned.insert(m.pinned.end(),3,1);}
        m.rest.push_back({-.1f,1.0005f,-.1f});m.mass.push_back(1);m.pinned.push_back(0);
        GpuPhysics solver;solver.build(device,queue,cache,shader,m);solver.reset(m.rest);
        PhysicsConfig c;c.gravity=0;c.dampingPerSecond=0;c.iterations=0;c.enableSelfCollision=true;
        auto cmd=device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,true);solver.recordStep(cmd,1.f/240,m.rest,nullptr,nullptr,c);device->flushCommandBuffer(cmd,queue,true);
        require("large self grid bucket overflow uses complete BVH",std::abs(double(solver.readback().back().y-1.003f)),1e-6);
    }
    report["passed"]=passed;report["limitations"]="Synthetic contract checks only; full-asset contact and long-run qualification are separate.";return report;
}
}
