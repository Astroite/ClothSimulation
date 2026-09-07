#include "demo_physics.h"
#include <json.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace mlcloth::demo;
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("Expected explicit self-contact fixture");
    nlohmann::json j;std::ifstream input(argv[1]);if(!input)throw std::runtime_error("Missing fixture");input>>j;
    auto vectors=[&](const char* name){std::vector<Vec3> out;for(const auto& v:j.at(name))out.push_back({v[0].get<float>(),v[1].get<float>(),v[2].get<float>()});return out;};
    Mesh mesh;mesh.rest=vectors("rest");mesh.triangles=j["triangles"].get<std::vector<uint32_t>>();mesh.mass=j["mass"].get<std::vector<float>>();mesh.pinned=j["pinned"].get<std::vector<uint8_t>>();
    auto initial=vectors("positions"),pins=vectors("pins"),gpu=vectors("gpu");PhysicsSolver solver;solver.build(mesh);solver.reset(initial);
    PhysicsConfig config;config.gravity=0;config.iterations=0;config.enableSelfCollision=true;config.thickness=j.at("thickness");
    solver.step(1.f/240,pins,nullptr,nullptr,config);double maximum=0,error=0;uint32_t worst=0;
    for(uint32_t i=0;i<initial.size();++i){auto delta=length(solver.positions()[i]-initial[i]);if(!std::isfinite(delta))throw std::runtime_error("Non-finite self correction");
        if(delta>maximum){maximum=delta;worst=i;}error=std::max(error,double(length(solver.positions()[i]-gpu[i])));}
    std::cout<<"Maximum correction "<<maximum<<" m at "<<worst<<"; CPU/GPU max difference "<<error<<" m\n";
    if(maximum>.03)throw std::runtime_error("Static self-contact generated an unbounded correction");
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
