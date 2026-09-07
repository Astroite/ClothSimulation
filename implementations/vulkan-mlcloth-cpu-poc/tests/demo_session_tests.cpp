#include "demo_session.h"
#include <iostream>

using namespace mlcloth::demo;
static void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
static double difference(const std::vector<glm::vec3>& a,const std::vector<glm::vec3>& b){
    require(a.size()==b.size(),"vertex count changed");double result=0;
    for(size_t i=0;i<a.size();++i)result=std::max(result,double(glm::length(a[i]-b[i])));return result;
}
static void finishSeek(DemoSession& s){
    size_t frames=0;while(s.seeking()){s.update(0);require(s.physicsSteps.size()<=8,"unbounded replay queue");s.physicsSteps.clear();
        require(++frames<3000,"seek failed to complete");}
}
int main(int argc,char** argv){try{
    require(argc==2,"manifest required");DemoSession s;s.gpuMode=true;s.load(argv[1]);s.settings=DemoSettings{};
    for(size_t i=0;i<s.assets.animations.size();++i)if(s.assets.animations[i].id=="complex")s.select(0,int(i));
    s.settings.physics.enableSelfCollision=false; // only queues work; physics is validated separately
    s.reset();
    for(int step=0;step<600;++step){for(int i=0;i<3;++i)s.step(i,1.f/240);s.physicsSteps.clear();}
    const auto expected=s.actors[0].cloth;
    require(s.actors[0].checkpoints.size()>=2,"missing two-second checkpoint");
    require(!s.actors[1].inference,"pure XPBD acquired an ML instance");
    require(s.actors[0].inference==s.actors[2].inference,"synchronized stream is not shared");
    const auto checkpointId=s.actors[1].checkpoints.at(480).id;
    s.seek(-1,2.5);require(s.actors[1].restoreCheckpoint==checkpointId,"wrong GPU checkpoint selected");
    require(s.actors[0].steps==480,"seek blocked until target instead of scheduling replay");
    s.settings.paused=true;finishSeek(s);
    require(s.settings.paused,"seek lost paused state");
    require(difference(expected,s.actors[0].cloth)<1e-6,"restored ML recurrent state differs from sequential playback");
    s.settings.paused=false;s.seek(-1,2.0);finishSeek(s);
    require(s.presentation(0).blend==1&&s.presentation(1).time==2.0,"exact checkpoint seek interpolated stale history");
    s.settings.paused=true;
    const auto physicalGeneration=s.actors[2].generation;
    s.synchronize(false);require(s.actors[2].generation==physicalGeneration,"desync reset physical state");
    require(s.actors[0].inference!=s.actors[2].inference,"desync did not fork inference");
    s.step(0,1.f/240);s.step(2,1.f/240);s.physicsSteps.clear();
    require(difference(s.actors[0].guide,s.actors[2].guide)<1e-6,"forked stream changed prediction");
    const auto otherTime=s.actors[0].time;s.seek(2,1.0);s.seek(2,.25);finishSeek(s);
    require(s.actors[2].steps==60&&s.actors[0].time==otherTime,"rapid independent seek affected another actor");
    s.settings.inPlace=true;s.update(0);auto before=s.actors[0].displayOffset;
    s.settings.spacing+=1;s.update(0);require(std::abs(s.actors[0].displayOffset.x-before.x-1)<1e-6,"paused display settings did not apply");
    s.synchronize(true);finishSeek(s);require(s.actors[0].steps==s.actors[1].steps&&s.actors[1].steps==s.actors[2].steps,"resync time mismatch");
    require(s.actors[0].inference==s.actors[2].inference,"resync stream mismatch");
    if(!s.assets.capsules.empty()){
        for(int mode=0;mode<3;++mode){s.settings.collisionMode=mode;s.reset();auto previousCollider=s.actors[1].collider;
            for(int actor=0;actor<3;++actor)s.step(actor,1.f/240);s.physicsSteps.clear();const auto& body=s.actors[1].collider;
            require(body.capsules.size()==(mode==1?0:s.assets.capsules.size()),"collision mode did not change capsule data");
            require(body.current.empty()==(mode==2),"collision mode did not change STM data");
            for(size_t i=0;i<body.capsules.size();++i)require(length(body.capsules[i].previousA-previousCollider.capsules[i].currentA)<1e-7f,"capsule sweep history was lost");
            auto mlDiagnostic=s.diagnosticCollider(0);require(mlDiagnostic.current.size()==body.current.size(),"ML reference collider uses different topology");
            for(size_t i=0;i<body.current.size();++i)require(length(mlDiagnostic.current[i]-body.current[i])<1e-6f,"ML diagnostic collider stayed at an old pose");
            require(!s.actors[1].inference,"collision switch created an ML instance for XPBD");
        }
    }
    s.settings.paused=false;s.settings.inPlace=true;s.accumulator=.25/240;
    const auto simulationTime=s.actors[0].time;const auto simulationSteps=s.actors[0].steps;
    const auto display=s.presentation(0);
    require(std::abs(display.blend-.25f)<1e-7f,"presentation did not use the fractional fixed-step clock");
    require(std::abs(display.time-(simulationTime-.75/240))<1e-9,"body display time differs from cloth endpoint interpolation");
    require(s.actors[0].time==simulationTime&&s.actors[0].steps==simulationSteps,"presentation advanced simulation");
    require(std::abs(display.displayOffset.x+display.pose.rootWorld.x-s.settings.spacing)<1e-6,"in-place offset used simulation root instead of displayed root");
    for(int actor=1;actor<3;++actor)require(std::abs(s.presentation(actor).time-display.time)<1e-9,"synchronized actors have different display times");
    s.actors[1].paused=true;require(s.presentation(1).blend==1&&s.presentation(1).time==s.actors[1].time,"independently paused actor kept interpolating");
    s.settings.paused=true;require(s.presentation(0).blend==1&&s.presentation(0).time==simulationTime,"pause did not expose exact current state");
    s.update(0);s.settings.paused=false;require(s.presentation(0).blend==1,"resume moved backward before the next physics step");
    s.seek(-1,0);finishSeek(s);require(s.presentation(0).time==0&&s.presentation(0).blend==1,"reset retained a stale interpolation interval");
    s.settings.inPlace=false;s.settings.autoSpacing=true;
    const auto& left=s.assets.animations[s.actors[0].animation];const auto& middle=s.assets.animations[s.actors[1].animation];const auto& right=s.assets.animations[s.actors[2].animation];
    require(left.rootPathMin.x+s.instanceOffset(0).x>=middle.rootPathMax.x+1.5999f,"left first-cycle trajectory overlaps middle display margin");
    require(right.rootPathMax.x+s.instanceOffset(2).x<=middle.rootPathMin.x-1.5999f,"right first-cycle trajectory overlaps middle display margin");
    s.settings.autoSpacing=false;require(s.instanceOffset(0).x==s.settings.spacing&&s.instanceOffset(2).x==-s.settings.spacing,"manual spacing toggle did not apply");
    s.settings.hybridAlgorithm=1;s.settings.collisionMode=0;
    for(int hz:{120,240,480}){s.settings.physicsHz=hz;s.reset();int inferences=0;
        require(!s.actors[2].inference,"GNN hybrid created an MLCloth stream");
        for(int i=0;i<hz/10;++i){s.step(2,1.f/hz);const auto& step=s.physicsSteps.back();
            require(step.config.gnnAcceleration&&step.guide.empty(),"GNN was treated as an ML shape guide");
            if(step.inferGnn){++inferences;require(step.gnnPins.size()==s.physicsMesh().rest.size(),"GNN future pins missing");
                require(step.gnnCurrent.current.size()==step.gnnFuture.current.size(),"GNN lookahead body topology changed");}
            s.physicsSteps.clear();}
        require(inferences==3,"GNN frequency depends on physics rate");
        s.seek(2,0);finishSeek(s);s.step(2,1.f/hz);require(s.physicsSteps.back().inferGnn,"reset retained old GNN phase");s.physicsSteps.clear();
    }
    std::cout<<"PASS: checkpoint replay, recurrent state, bounded seek, cancellation, desync/resync, pure XPBD independence, GNN clock and targets\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
