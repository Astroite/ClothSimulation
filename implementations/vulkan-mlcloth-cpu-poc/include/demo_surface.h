#pragma once
#include "demo_physics.h"
#include <stdexcept>
#include <algorithm>

namespace mlcloth::demo {
struct BodySurfaceFrame {
    double time{};
    std::vector<uint64_t> ids;
    std::vector<Vec3> positions,normals,targets;
    void validate()const {
        if(!std::isfinite(time)||ids.size()!=positions.size()||normals.size()!=ids.size()||(!targets.empty()&&targets.size()!=ids.size()))
            throw std::runtime_error("Invalid body surface frame shape/time");
        auto sorted=ids;std::sort(sorted.begin(),sorted.end());
        if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())throw std::runtime_error("Duplicate surface sample ID");
        for(const auto* values:{&positions,&normals,&targets})for(auto v:*values)
            if(!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.z))throw std::runtime_error("Nonfinite surface frame");
    }
};

// Fixed triangle-vertex barycentrics preserve the existing STM sampling density.
// No skeleton name, bone number or cloth template enters this interface.
inline BodySurfaceFrame sampleBodySurface(const TriangleCollider& body,const TriangleCollider& future,double time){
    if(body.triangles.size()%3||body.triangles!=future.triangles||body.current.size()!=future.current.size()||body.capsules.size()!=future.capsules.size())
        throw std::runtime_error("Surface topology changed between animation samples");
    BodySurfaceFrame out;out.time=time;out.positions=body.current;out.targets=future.current;out.normals.resize(body.current.size());
    for(size_t i=0;i<body.current.size();++i)out.ids.push_back(i);
    for(size_t t=0;t<body.triangles.size();t+=3){auto a=body.triangles[t],b=body.triangles[t+1],c=body.triangles[t+2];
        if(std::max({a,b,c})>=body.current.size())throw std::runtime_error("Surface triangle out of range");
        auto n=cross(body.current[b]-body.current[a],body.current[c]-body.current[a]);out.normals[a]+=n;out.normals[b]+=n;out.normals[c]+=n;}
    for(auto& n:out.normals)n=normalize(n);
    for(size_t i=0;i<body.capsules.size();++i){const auto& a=body.capsules[i];const auto& b=future.capsules[i];
        if(!a.hasSurfaceFrame||!b.hasSurfaceFrame)throw std::runtime_error("Capsule surface requires a stable material frame");
        for(int ring=0;ring<5;++ring)for(int k=0;k<8;++k){const float angle=float(k)*6.28318530718f/8,t=float(ring)/4;
            const auto n=a.surfaceU*std::cos(angle)+a.surfaceV*std::sin(angle),nn=b.surfaceU*std::cos(angle)+b.surfaceV*std::sin(angle);
            out.ids.push_back((uint64_t(1)<<63)|(uint64_t(i)*40+ring*8+k));
            out.positions.push_back(a.currentA*(1-t)+a.currentB*t+n*a.radius);out.targets.push_back(b.currentA*(1-t)+b.currentB*t+nn*b.radius);out.normals.push_back(n);}
    }
    out.validate();return out;
}
}
