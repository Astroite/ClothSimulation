#pragma once
#include "demo_assets.h"
#include "demo_surface.h"

namespace mlcloth::demo {
// Asset-specific skinning stays outside the network contract.
inline TriangleCollider assetCollider(const DemoAssetManifest& assets,const Pose& pose,int mode=0){
    TriangleCollider result;
    if(mode!=2){std::vector<glm::vec3> vertices;assets.collision.skin(pose,vertices);
        for(auto v:vertices)result.current.push_back({v.x,v.y,v.z});result.triangles=assets.collision.triangles;}
    result.previous=result.current;
    if(mode!=1)for(const auto& binding:assets.capsules){
        auto a=glm::vec3(pose.world.at(binding.bone)*glm::vec4(binding.a,1));
        auto b=glm::vec3(pose.world.at(binding.bone)*glm::vec4(binding.b,1));
        auto axis=binding.b-binding.a;
        if(glm::length(axis)<1e-12f)throw std::runtime_error("Degenerate capsule binding axis");
        axis=glm::normalize(axis);const auto seed=std::abs(axis.x)<.8f?glm::vec3(1,0,0):glm::vec3(0,0,1);
        const auto u=glm::normalize(glm::cross(axis,seed)),v=glm::cross(axis,u);
        const auto worldU=glm::normalize(glm::vec3(pose.world[binding.bone]*glm::vec4(u,0)));
        const auto worldV=glm::normalize(glm::vec3(pose.world[binding.bone]*glm::vec4(v,0)));
        MovingCapsule cap;cap.currentA=cap.previousA={a.x,a.y,a.z};cap.currentB=cap.previousB={b.x,b.y,b.z};cap.radius=binding.radius;
        cap.surfaceU={worldU.x,worldU.y,worldU.z};cap.surfaceV={worldV.x,worldV.y,worldV.z};cap.hasSurfaceFrame=true;
        result.capsules.push_back(cap);
    }
    return result;
}
}
