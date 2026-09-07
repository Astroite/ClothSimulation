// Pure CPU XPBD reference and 240 Hz exogenous inputs. Never loads a neural runtime.
#include "demo_surface_adapter.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
using namespace mlcloth::demo;
namespace fs=std::filesystem;
static std::vector<Vec3> vectors(const std::vector<glm::vec3>& source){
    std::vector<Vec3> out;for(auto v:source)out.push_back({v.x,v.y,v.z});return out;
}
struct Writer {
    std::ofstream stream;uint64_t offset{};
    explicit Writer(const fs::path& path):stream(path,std::ios::binary){if(!stream)throw std::runtime_error("Cannot create reference data");}
    template<class T> Json put(const std::vector<T>& data,const char* dtype,std::vector<size_t> shape){
        const auto bytes=data.size()*sizeof(T);const auto start=offset;
        stream.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(bytes));
        if(!stream)throw std::runtime_error("Reference data write failed");offset+=bytes;
        return {{"offset",start},{"bytes",bytes},{"dtype",dtype},{"shape",shape}};
    }
    Json xyz(const std::vector<Vec3>& data){static_assert(sizeof(Vec3)==12);return put(data,"<f4",{data.size(),3});}
};
static void previousFrom(TriangleCollider& next,const TriangleCollider& previous){
    next.previous=previous.current;
    for(size_t i=0;i<next.capsules.size();++i){next.capsules[i].previousA=previous.capsules[i].currentA;next.capsules[i].previousB=previous.capsules[i].currentB;}
}
static Json colliderRow(Writer& writer,const TriangleCollider& body){
    std::vector<float> caps;for(auto c:body.capsules)for(float v:{c.currentA.x,c.currentA.y,c.currentA.z,c.currentB.x,c.currentB.y,c.currentB.z,c.radius})caps.push_back(v);
    return {{"positions",writer.xyz(body.current)},{"capsules",writer.put(caps,"<f4",{body.capsules.size(),7})}};
}
int main(int argc,char** argv){try{
    fs::path manifest,output;std::string clip="walk";int hz=480,iterations=8;double seconds=.3,speed=1;bool inPlace=false;
    for(int i=1;i<argc;++i){const std::string arg=argv[i];
        if(i+1>=argc)throw std::runtime_error("Missing argument value");const std::string value=argv[++i];
        if(arg=="--manifest")manifest=value;else if(arg=="--output")output=value;else if(arg=="--clip")clip=value;
        else if(arg=="--hz")hz=std::stoi(value);else if(arg=="--iterations")iterations=std::stoi(value);
        else if(arg=="--seconds")seconds=std::stod(value);else if(arg=="--speed")speed=std::stod(value);else if(arg=="--in-place")inPlace=std::stoi(value)!=0;
        else throw std::runtime_error("Unknown argument: "+arg);
    }
    if(manifest.empty()||output.empty()||(hz!=480&&hz!=960)||iterations!=8||!std::isfinite(seconds)||seconds<=0||!std::isfinite(speed)||speed<=0)
        throw std::runtime_error("Required: --manifest FILE --output NEWDIR [--clip walk --hz 480|960 --iterations 8 --seconds .3 --speed 1 --in-place 0]");
    if(fs::exists(output/"metadata.json"))throw std::runtime_error("Reference already exists; choose a new output directory");
    DemoAssetManifest assets;assets.load(manifest);
    const auto selected=std::find_if(assets.animations.begin(),assets.animations.end(),[&](const Animation& a){return a.id==clip;});
    const auto initialClip=std::find_if(assets.animations.begin(),assets.animations.end(),[](const Animation& a){return a.id=="complex";});
    if(selected==assets.animations.end()||initialClip==assets.animations.end())throw std::runtime_error("Missing selected/T-pose clip");
    Mesh mesh;mesh.rest=vectors(assets.cloth.rest);mesh.triangles=assets.cloth.triangles;mesh.mass=assets.cloth.mass;
    for(auto p:assets.cloth.pinned)mesh.pinned.push_back(static_cast<uint8_t>(p));
    PhysicsConfig config;config.iterations=iterations;config.stretchCompliance=1e-7f;config.shearCompliance=1e-6f;
    config.bendCompliance=1e4f;config.guideCompliance=-1;config.dampingPerSecond=.7f;config.enableCollision=true;config.enableSelfCollision=true;
    PhysicsSolver solver;solver.build(mesh);solver.reset(mesh.rest);
    Pose initial,target,pose;initialClip->sample(0,false,initial);selected->sample(0,true,target);
    auto body=assetCollider(assets,initial);std::vector<glm::vec3> pinGlm;
    // Identical Demo 240 Hz warm-up for 480/960: same initial x and v.
    for(int i=0;i<240;++i){float t=std::clamp(float(i-30)/180,0.f,1.f);t=t*t*(3-2*t);
        blendPoses(initial,target,initialClip->parents,t,pose);assets.cloth.pins(pose,pinGlm,&assets.body);
        auto next=assetCollider(assets,pose);previousFrom(next,body);body=std::move(next);
        solver.step(1.f/240,vectors(pinGlm),nullptr,&body,config);
    }
    fs::create_directories(output);Writer writer(output/"frames.bin");
    Json meta={{"version",2},{"clip",clip},{"hz",hz},{"iterations",iterations},{"speed",speed},{"in_place",inPlace},
        {"root_mode_semantics","display offset only; world physical trajectory unchanged"},
        {"speed_semantics","physical animation retiming; separate from Demo wall-clock playback speed"},
        {"manifest",fs::absolute(manifest).generic_string()},{"warmup","Demo 240 Hz x 240, 8 iterations; identical for both reference rates"},
        {"continuous",true},{"inference_hz",30},{"student_hz",240},{"frames_file","frames.bin"},
        {"physics_config",{{"iterations",iterations},{"stretchCompliance",config.stretchCompliance},{"shearCompliance",config.shearCompliance},
            {"bendCompliance",config.bendCompliance},{"guideCompliance",-1},{"dampingPerSecond",config.dampingPerSecond},
            {"gravity",config.gravity},{"enableCollision",true},{"enableSelfCollision",true},{"thickness",config.thickness},{"friction",config.friction}}}};
    meta["arrays"]={{"rest",writer.xyz(mesh.rest)},{"triangles",writer.put(mesh.triangles,"<u4",{mesh.triangles.size()/3,3})},
        {"mass",writer.put(mesh.mass,"<f4",{mesh.mass.size()})},{"pinned",writer.put(mesh.pinned,"|u1",{mesh.pinned.size()})},
        {"collider_triangles",writer.put(body.triangles,"<u4",{body.triangles.size()/3,3})}};
    auto state=[&](double time){
        auto surface=sampleBodySurface(body,body,time);
        return Json{{"time",time},{"x",writer.xyz(solver.positions())},{"v",writer.xyz(solver.velocities())},
            {"body_positions",writer.xyz(surface.positions)},{"body_normals",writer.xyz(surface.normals)},
            {"surface_ids",writer.put(surface.ids,"<u8",{surface.ids.size()})}};
    };
    meta["frames"]=Json::array();meta["intervals"]=Json::array();meta["frames"].push_back(state(0));
    const int frames=static_cast<int>(std::ceil(seconds*30-1e-9)),stride=hz/240,substeps=hz/30;
    for(int frame=0;frame<frames;++frame){
        Json interval={{"colliders",Json::array()},{"pins",Json::array()}};interval["colliders"].push_back(colliderRow(writer,body));
        for(int sub=0;sub<substeps;++sub){
            const double time=double(frame*substeps+sub+1)/hz;
            selected->sample(time*speed,true,pose);assets.cloth.pins(pose,pinGlm,&assets.body);auto pins=vectors(pinGlm);
            auto next=assetCollider(assets,pose);previousFrom(next,body);body=std::move(next);
            solver.step(1.f/hz,pins,nullptr,&body,config);
            if((sub+1)%stride==0){interval["pins"].push_back(writer.xyz(pins));interval["colliders"].push_back(colliderRow(writer,body));}
        }
        meta["intervals"].push_back(std::move(interval));meta["frames"].push_back(state(double(frame+1)/30));
        std::cout<<clip<<" "<<hz<<" Hz: "<<frame+1<<"/"<<frames<<" inference intervals\n"<<std::flush;
    }
    writer.stream.close();meta["byte_count"]=writer.offset;meta["frame_count"]=meta["frames"].size();
    std::ofstream out(output/"metadata.json");out<<meta.dump(2);if(!out)throw std::runtime_error("Metadata write failed");
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
