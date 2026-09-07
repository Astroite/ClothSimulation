#include "demo_session.h"
#include "demo_surface_adapter.h"
#include <chrono>
#include <cmath>
#include <fstream>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace mlcloth::demo {
static Vec3 physicsVector(const glm::vec3& v){return {v.x,v.y,v.z};}
static std::vector<Vec3> physicsVectors(const std::vector<glm::vec3>& input){std::vector<Vec3> out;out.reserve(input.size());for(auto v:input)out.push_back(physicsVector(v));return out;}
InferenceStream::InferenceStream(const DemoAssetManifest& a,const Animation& animation,int threads,bool loop,std::shared_ptr<const Prediction> seed)
    :assets_(a),animation_(animation),threads_(threads),loop_(loop),seed_(std::move(seed)){
    if(seed_){std::promise<std::shared_ptr<const Prediction>> ready;ready.set_value(seed_);
        results_[seed_->tick]=ready.get_future().share();scheduled_=seed_->tick+1;}
    worker_=std::thread(&InferenceStream::work,this);
}
InferenceStream::~InferenceStream(){ {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;}wake_.notify_all();space_.notify_all();if(worker_.joinable())worker_.join();}
void InferenceStream::prefetch(uint64_t tick){
    std::unique_lock<std::mutex> lock(mutex_);
    while(scheduled_<=tick){
        space_.wait(lock,[this]{return stopping_||jobs_.size()<3;});
        if(stopping_)return;
        Job job;job.tick=scheduled_;
        results_[scheduled_]=job.promise.get_future().share();jobs_.push_back(std::move(job));++scheduled_;wake_.notify_one();
    }
    while(results_.size()>5 && results_.begin()->first+3<tick)results_.erase(results_.begin());
}
std::shared_ptr<const Prediction> InferenceStream::at(uint64_t tick){
    prefetch(tick);std::shared_future<std::shared_ptr<const Prediction>> result;
    {std::lock_guard<std::mutex> lock(mutex_);auto found=results_.find(tick);if(found==results_.end())throw std::runtime_error("Inference history expired; seek must replay from reset");result=found->second;}
    return result.get();
}
void InferenceStream::work(){
    std::unique_ptr<AILabRuntime> runtime;std::exception_ptr failure;
    try{runtime=std::make_unique<AILabRuntime>(assets_.runtime,assets_.modelBytes.data()+assets_.modelInfo.payloadOffset,assets_.modelInfo.payloadLen,1969,16394,threads_);}catch(...){failure=std::current_exception();}
    MLClothSequenceState sequence;if(seed_)sequence=seed_->state;Pose pose;
    for(;;){Job job;{std::unique_lock<std::mutex> lock(mutex_);wake_.wait(lock,[this]{return stopping_||!jobs_.empty();});if(stopping_)return;job=std::move(jobs_.front());jobs_.pop_front();space_.notify_all();}
        try{if(failure)std::rethrow_exception(failure);animation_.sample(double(job.tick)/30,loop_,pose);
            auto result=std::make_shared<Prediction>();result->tick=job.tick;
            result->milliseconds=sequence.inferFrame(*runtime,0,pose.localFu,pose.componentFu,pose.positionsCm);
            result->localCm.resize(5294);std::memcpy(result->localCm.data(),sequence.output().data(),5294*3*sizeof(float));
            result->state=sequence;job.promise.set_value(result);
        }catch(...){job.promise.set_exception(std::current_exception());}
    }
}
void DemoSession::load(const std::filesystem::path& path){
    std::ofstream log(path.parent_path()/"startup.log");log<<"Loading assets"<<std::endl;
    assets.load(path);mesh_.rest=physicsVectors(assets.cloth.rest);mesh_.triangles=assets.cloth.triangles;mesh_.mass=assets.cloth.mass;
    log<<"Assets loaded"<<std::endl;
    for(auto p:assets.cloth.pinned)mesh_.pinned.push_back(static_cast<uint8_t>(p));
    settings.physics.iterations=2;settings.physics.stretchCompliance=1e-7f;settings.physics.shearCompliance=1e-6f;
    settings.physics.bendCompliance=1e4f;settings.physics.guideCompliance=1e-3f;settings.physics.enableCollision=true;
    settings.physics.enableSelfCollision=true;settings.physics.dampingPerSecond=.7f;
    for(size_t i=0;i<assets.animations.size();++i)if(assets.animations[i].id=="walk")for(auto& actor:actors)actor.animation=static_cast<int>(i);
    for(auto& actor:actors){actor.solver.build(mesh_);log<<"CPU constraint tables built"<<std::endl;}
    loadSettings();log<<"Initializing actors; renderOnly="<<renderOnly<<std::endl;reset();log<<"Actors initialized"<<std::endl;
}
void DemoSession::initialize(int index){
    auto& actor=actors[index];actor.inference.reset();actor.time=0;actor.steps=0;actor.interpolationReady=false;actor.generation=++generation;
    actor.restoreCheckpoint=0;actor.checkpoints.clear();actor.seekTarget.reset();
    const auto& animation=assets.animations.at(actor.animation);animation.sample(0,settings.loop,actor.pose);
    assets.cloth.pins(actor.pose,actor.pinTargets,&assets.body,renderOnly);updateCollider(actor.pose,actor.collider,actor.colliderPoints,true);
    // Begin at the asset T pose. Warm-up continuously moves the body and attachments
    // into the selected pose, instead of teleporting limbs through the rest garment.
    actor.cloth=assets.cloth.rest;
    actor.solver.reset(physicsVectors(actor.cloth));
    if(index!=0&&!renderOnly&&!gpuMode){
        warmup(index,[&](const PhysicsStep& step){actor.solver.step(step.dt,step.pins,nullptr,&step.collider,step.config);});
        const auto& solved=actor.solver.positions();for(size_t i=0;i<solved.size();++i)actor.cloth[i]=glm::vec3(solved[i].x,solved[i].y,solved[i].z);
    }
    actor.previousCloth=actor.cloth;
    if(index==0||(index==2&&settings.hybridAlgorithm==0)){
        if(index==2&&settings.synchronized&&actors[0].inference)actor.inference=actors[0].inference;
        else actor.inference=std::make_shared<InferenceStream>(assets,animation,settings.threads,settings.loop);
        actor.inference->prefetch(1);
        if(index==0){auto prediction=actor.inference->at(0);for(size_t v=0;v<actor.cloth.size();++v)actor.cloth[v]=glm::vec3(actor.pose.world[animation.root]*glm::vec4(prediction->localCm[v],1));}
    }
    updateDisplay(index);
    checkpoint(index);
}
void DemoSession::reset(int actor){accumulator=0;physicsSteps.clear();if(actor<0||settings.synchronized){for(int i=0;i<3;++i)initialize(i);}else initialize(actor);status="Ready";}
void DemoSession::select(int actor,int animation){
    if(animation<0||animation>=static_cast<int>(assets.animations.size()))return;
    if(settings.synchronized){for(auto& a:actors)a.animation=animation;reset();}
    else{actors.at(actor).animation=animation;reset(actor);}
}
void DemoSession::synchronize(bool enabled){
    if(settings.synchronized==enabled)return;
    if(enabled){double t=actors[0].time;int clip=actors[0].animation;settings.synchronized=true;for(auto& a:actors){a.animation=clip;a.paused=false;}reset();seek(-1,t);}
    else{settings.synchronized=false;if(settings.hybridAlgorithm!=0)return;
        // Preserve physical states and fork the recurrent stream at its current ML tick.
        auto& actor=actors[2];auto seed=actor.inference->at(static_cast<uint64_t>(std::floor(actor.time*30+1e-8)));
        actor.inference=std::make_shared<InferenceStream>(assets,assets.animations[actor.animation],settings.threads,settings.loop,seed);
    }
}
void DemoSession::warmup(int index,const std::function<void(const PhysicsStep&)>& consume)const{
    const auto& actor=actors[index];auto reference=std::find_if(assets.animations.begin(),assets.animations.end(),[](const Animation& clip){return clip.id=="complex";});
    if(reference==assets.animations.end())throw std::runtime_error("Missing reference T-pose animation");
    Pose initial,pose;reference->sample(0,false,initial);std::vector<glm::vec3> pins,body;
    PhysicsStep step;step.actor=index;step.dt=1.f/240;step.config=settings.physics;step.config.guideCompliance=-1;
    updateCollider(initial,step.collider,body,true);
    for(int n=0;n<240;++n){float t=std::clamp(float(n-30)/180,0.f,1.f);t=t*t*(3-2*t);
        blendPoses(initial,actor.pose,reference->parents,t,pose);
        assets.cloth.pins(pose,pins,&assets.body);step.pins=physicsVectors(pins);updateCollider(pose,step.collider,body);consume(step);
    }
}
void DemoSession::seek(int actor,double time){
    if(!std::isfinite(time))throw std::runtime_error("Invalid timeline time");
    const uint64_t count=static_cast<uint64_t>(std::floor(std::max(0.,time)*settings.physicsHz+1e-6));
    accumulator=0;
    for(int i=0;i<3;++i){if(actor>=0&&!settings.synchronized&&i!=actor)continue;
        auto& a=actors[i];auto found=a.checkpoints.upper_bound(count);if(found==a.checkpoints.begin())throw std::runtime_error("Missing initial checkpoint");
        --found;restore(i,found->second);a.seekTarget=count;
    }
    physicsSteps.clear();status="Restoring timeline...";
}
bool DemoSession::seeking()const{for(const auto& actor:actors)if(actor.seekTarget)return true;return false;}
uint64_t DemoSession::checkpoint(int index){
    auto& a=actors[index];const uint64_t interval=uint64_t(settings.physicsHz)*2;
    // Keep the first display cycle: bounded memory, independent of loop count.
    if(a.steps%interval||a.time>assets.animations[a.animation].duration+1e-8)return 0;
    if(a.checkpoints.count(a.steps))return 0;
    StateCheckpoint state;state.id=++checkpointSerial_;state.steps=a.steps;
    if(a.inference)state.prediction=a.inference->at(a.steps*30/uint64_t(settings.physicsHz));
    if(index!=0&&!gpuMode){state.positions=a.solver.positions();state.velocities=a.solver.velocities();}
    const auto id=state.id;a.checkpoints.emplace(a.steps,std::move(state));return id;
}
void DemoSession::restore(int index,const StateCheckpoint& state){
    auto& a=actors[index];a.steps=state.steps;a.time=double(state.steps)/settings.physicsHz;
    a.interpolationReady=false;
    a.generation=++generation;a.restoreCheckpoint=index==0?0:state.id;
    const auto& animation=assets.animations[a.animation];animation.sample(a.time,settings.loop,a.pose);
    assets.cloth.pins(a.pose,a.pinTargets,&assets.body,renderOnly);updateCollider(a.pose,a.collider,a.colliderPoints,true);
    if(index==0||(index==2&&settings.hybridAlgorithm==0)){
        if(index==2&&settings.synchronized)a.inference=actors[0].inference;
        else a.inference=std::make_shared<InferenceStream>(assets,animation,settings.threads,settings.loop,state.prediction);
        a.inference->prefetch(state.prediction->tick+1);
        if(index==0){a.cloth.resize(5294);for(size_t v=0;v<a.cloth.size();++v)
            a.cloth[v]=glm::vec3(a.pose.world[animation.root]*glm::vec4(state.prediction->localCm[v],1));}
    }
    if(index!=0&&!gpuMode){a.solver.reset(state.positions,state.velocities);a.cloth.resize(state.positions.size());
        for(size_t v=0;v<a.cloth.size();++v)a.cloth[v]=glm::vec3(state.positions[v].x,state.positions[v].y,state.positions[v].z);}
    a.previousCloth=a.cloth;updateDisplay(index);
}
void DemoSession::updateCollider(const Pose& pose,TriangleCollider& collider,std::vector<glm::vec3>& display,bool resetHistory)const{
    auto next=assetCollider(assets,pose,settings.collisionMode);
    if(!resetHistory&&collider.current.size()==next.current.size())next.previous=collider.current;
    if(!resetHistory&&collider.capsules.size()==next.capsules.size())for(size_t i=0;i<next.capsules.size();++i){
        next.capsules[i].previousA=collider.capsules[i].currentA;next.capsules[i].previousB=collider.capsules[i].currentB;}
    collider=std::move(next);display.clear();display.reserve(collider.current.size());
    for(auto v:collider.current)display.emplace_back(v.x,v.y,v.z);
}
TriangleCollider DemoSession::diagnosticCollider(int index,bool display)const{
    TriangleCollider result;std::vector<glm::vec3> points;updateCollider(display?presentation(index).pose:actors.at(index).pose,result,points,true);return result;
}
Presentation DemoSession::presentation(int index)const{
    const auto& actor=actors.at(index);Presentation result;
    // Interpolate one fixed step behind the advancing clock. Pauses and explicit
    // restores display the exact current state; no extrapolation through contacts.
    if(actor.interpolationReady&&actor.steps&&!settings.paused&&!actor.paused&&!seeking())result.blend=static_cast<float>(std::clamp(accumulator*settings.physicsHz,0.,1.));
    result.time=std::max(0.,actor.time-(1.-result.blend)/settings.physicsHz);
    if(result.blend==1)result.pose=actor.pose;
    else assets.animations[actor.animation].sample(result.time,settings.loop,result.pose);
    result.displayOffset=instanceOffset(index);
    if(settings.inPlace){result.displayOffset.x-=result.pose.rootWorld.x;result.displayOffset.z-=result.pose.rootWorld.z;}
    return result;
}
glm::vec3 DemoSession::instanceOffset(int index)const{
    if(index<0||index>=3)throw std::runtime_error("Invalid display actor");
    float spacing=settings.spacing;
    if(settings.autoSpacing&&!settings.inPlace&&index!=1){
        const auto& middle=assets.animations[actors[1].animation];const auto& outer=assets.animations[actors[index].animation];
        const float pathSeparation=index==0?middle.rootPathMax.x-outer.rootPathMin.x:outer.rootPathMax.x-middle.rootPathMin.x;
        spacing=std::max(spacing,pathSeparation+1.6f);
    }
    return glm::vec3((1-index)*spacing,0,0);
}
void DemoSession::updateDisplay(int index){
    auto& actor=actors[index];actor.displayOffset=instanceOffset(index);
    if(settings.inPlace){actor.displayOffset.x-=actor.pose.rootWorld.x;actor.displayOffset.z-=actor.pose.rootWorld.z;}
}
void DemoSession::step(int index,float dt){
    auto& actor=actors[index];const auto& animation=assets.animations.at(actor.animation);
    actor.previousCloth=actor.cloth;actor.interpolationReady=true;actor.time=double(actor.steps+1)/settings.physicsHz;
    const bool shared=settings.synchronized&&index>0&&actors[0].steps==actor.steps+1;
    if(shared)actor.pose=actors[0].pose;else animation.sample(actor.time,settings.loop,actor.pose);
    updateDisplay(index);
    if(index==2&&shared&&settings.hybridAlgorithm==0){actor.guide=actors[0].guide;actor.inferenceMs=actors[0].inferenceMs;}
    else if(index==0||(index==2&&settings.hybridAlgorithm==0)){
        double phase=actor.time*30;uint64_t tick=static_cast<uint64_t>(std::floor(phase));float blend=static_cast<float>(phase-tick);
        auto first=actor.inference->at(tick),second=actor.inference->at(tick+1);actor.inference->prefetch(tick+2);
        actor.inferenceMs=second->milliseconds;actor.guide.resize(5294);
        const auto& root=actor.pose.world[animation.root];
        for(size_t i=0;i<5294;++i)actor.guide[i]=glm::vec3(root*glm::vec4(glm::mix(first->localCm[i],second->localCm[i],blend),1));
        if(index==0){actor.cloth=actor.guide;++actor.steps;checkpoint(index);return;}
    }
    const auto begin=std::chrono::steady_clock::now();
    const bool sharedBody=shared&&index==2&&actors[1].steps==actor.steps+1;
    if(sharedBody)actor.pinTargets=actors[1].pinTargets;else assets.cloth.pins(actor.pose,actor.pinTargets,&assets.body,renderOnly);
    if(renderOnly){actor.cloth=index==2?actor.guide:actor.pinTargets;++actor.steps;checkpoint(index);return;}
    if(sharedBody){actor.collider=actors[1].collider;actor.colliderPoints=actors[1].colliderPoints;}
    else updateCollider(actor.pose,actor.collider,actor.colliderPoints);
    const auto pins=physicsVectors(actor.pinTargets);const auto guide=physicsVectors(actor.guide);
    if(gpuMode){
        PhysicsStep step;step.actor=index;step.dt=dt;step.generation=actor.generation;step.pins=pins;step.guide=guide;step.collider=actor.collider;step.config=settings.physics;
        if(index==2&&settings.hybridAlgorithm!=0){
            step.guide.clear();step.config.guideCompliance=-1;step.config.gnnAcceleration=true;
            step.inferGnn=actor.steps%uint64_t(settings.physicsHz/30)==0;
            if(step.inferGnn){Pose current,future;const double start=double(actor.steps)/settings.physicsHz;step.gnnTime=start;
                animation.sample(start,settings.loop,current);animation.sample(start+1./30,settings.loop,future);
                std::vector<glm::vec3> points,targets;updateCollider(current,step.gnnCurrent,points,true);updateCollider(future,step.gnnFuture,points,true);
                assets.cloth.pins(future,targets,&assets.body);step.gnnPins=physicsVectors(targets);
            }
            actor.inferenceMs=0;
        }
        ++actor.steps;step.checkpoint=checkpoint(index);physicsSteps.push_back(std::move(step));return;
    }
    actor.solver.step(dt,pins,index==2?&guide:nullptr,&actor.collider,settings.physics);
    const auto& result=actor.solver.positions();for(size_t i=0;i<result.size();++i)actor.cloth[i]=glm::vec3(result[i].x,result[i].y,result[i].z);
    actor.solveMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();++actor.steps;checkpoint(index);
}
void DemoSession::update(double seconds){
    for(int i=0;i<3;++i)updateDisplay(i);
    for(auto& actor:actors)if(settings.paused||actor.paused)actor.interpolationReady=false;
    if(seeking()){
        const float dt=1.f/settings.physicsHz;
        for(int n=0;n<4;++n)for(int i=0;i<3;++i){auto& a=actors[i];if(!a.seekTarget)continue;
            if(a.steps<*a.seekTarget)step(i,dt);if(a.steps>=*a.seekTarget)a.seekTarget.reset();}
        if(!seeking())status="Timeline restored";return;
    }
    if(settings.paused)return;
    if(!std::isfinite(seconds)||seconds<0)throw std::runtime_error("Invalid frame time");
    accumulator+=seconds*settings.speed;const double dt=1.0/settings.physicsHz;
    int count=0;while(accumulator+1e-10>=dt&&count<8){for(int i=0;i<3;++i)if(!actors[i].paused){
        if(!settings.loop&&actors[i].time+dt>assets.animations[actors[i].animation].duration+1e-8)actors[i].paused=true;
        else step(i,static_cast<float>(dt));}accumulator-=dt;++count;}
    if(accumulator>.5){settings.paused=true;status="Simulation backlog exceeded 0.5 s; paused. Reset or lower playback speed.";}
}
void DemoSession::saveSettings() const {
    const auto& p=settings.physics;
    Json j={{"version",2},{"speed",settings.speed},{"physicsHz",settings.physicsHz},{"threads",settings.threads},
        {"spacing",settings.spacing},{"autoSpacing",settings.autoSpacing},{"inPlace",settings.inPlace},{"material",settings.material},{"view",settings.view},
        {"synchronized",settings.synchronized},{"collisionMode",settings.collisionMode},{"loop",settings.loop},{"focus",settings.focus},{"autoCamera",settings.autoCamera},
        {"hybridAlgorithm",settings.hybridAlgorithm},{"gnnStrength",settings.gnnStrength},{"gnnTrust",settings.gnnTrust},
        {"physics",{{"iterations",p.iterations},{"body",p.enableCollision},{"self",p.enableSelfCollision},{"thickness",p.thickness},
            {"damping",p.dampingPerSecond},{"friction",p.friction},{"guide",p.guideCompliance},{"stretch",p.stretchCompliance},
            {"shear",p.shearCompliance},{"bend",p.bendCompliance},{"tethers",p.enableTethers},{"tetherScale",p.tetherScale},{"contactAwareGuide",p.contactAwareGuide}}}};
    const auto path=assets.directory/"user-settings.json";const auto temporary=assets.directory/"user-settings.json.tmp";
    {std::ofstream stream(temporary);if(!stream)throw std::runtime_error("Cannot save settings");stream<<j.dump(2);if(!stream)throw std::runtime_error("Cannot write settings");}
    if(!MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("Cannot publish settings");
}
void DemoSession::loadSettings(){
    try{std::ifstream stream(assets.directory/"user-settings.json");if(!stream)return;Json j;stream>>j;if(j.at("version")!=1&&j.at("version")!=2)return;
        settings.collisionMode=std::clamp(j.value("collisionMode",0),0,2);
        settings.hybridAlgorithm=gpuMode&&!renderOnly?std::clamp(j.value("hybridAlgorithm",0),0,2):0;
        settings.gnnStrength=std::clamp(j.value("gnnStrength",1.f),0.f,1.f);settings.gnnTrust=std::clamp(j.value("gnnTrust",2.f),.1f,8.f);
        settings.speed=std::clamp(j.value("speed",1.f),.25f,2.f);int hz=j.value("physicsHz",240);if(hz==120||hz==240||hz==480)settings.physicsHz=hz;
        int threads=j.value("threads",1);if(threads==1||threads==2||threads==4)settings.threads=threads;
        settings.spacing=std::clamp(j.value("spacing",2.5f),1.5f,12.f);settings.inPlace=j.value("inPlace",false);
        settings.autoSpacing=j.value("autoSpacing",true);
        settings.material=std::clamp(j.value("material",0),0,2);settings.view=std::clamp(j.value("view",0),0,3);
        settings.synchronized=j.value("synchronized",true);settings.loop=j.value("loop",true);settings.focus=std::clamp(j.value("focus",-1),-1,2);settings.autoCamera=j.value("autoCamera",true);
        if(j.count("physics")){const auto& p=j.at("physics");auto& config=settings.physics;
            config.iterations=std::clamp(p.value("iterations",2),1,8);config.enableCollision=p.value("body",true);config.enableSelfCollision=p.value("self",true);
            config.thickness=std::clamp(p.value("thickness",.003f),.001f,.01f);config.dampingPerSecond=std::clamp(p.value("damping",.7f),0.f,4.f);
            config.friction=std::clamp(p.value("friction",.1f),0.f,1.f);config.guideCompliance=std::clamp(p.value("guide",.001f),.00001f,.1f);
            config.stretchCompliance=std::clamp(p.value("stretch",1e-7f),0.f,.01f);config.shearCompliance=std::clamp(p.value("shear",1e-6f),0.f,.01f);
              config.bendCompliance=std::clamp(p.value("bend",1e4f),0.f,1e6f);
            config.enableTethers=p.value("tethers",false);config.tetherScale=std::clamp(p.value("tetherScale",1.05f),1.f,1.3f);
            config.contactAwareGuide=p.value("contactAwareGuide",true);
        }
    }catch(const std::exception&){status="Invalid saved settings ignored";}
}
}
