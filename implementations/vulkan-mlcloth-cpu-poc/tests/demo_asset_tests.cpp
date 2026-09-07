#include "demo_assets.h"
#include <iostream>
#include <fstream>
#include <algorithm>

int main(int argc,char** argv){
    try{
        if(argc!=2)throw std::runtime_error("Expected demo manifest path");
        mlcloth::demo::DemoAssetManifest assets;assets.load(argv[1]);
        if(assets.animations.size()!=7)throw std::runtime_error("Expected exactly seven demo animations");
        auto body=mlcloth::demo::AssetPack::read(assets.directory/"character.dmp");
        auto reference=body.array<glm::vec3>("reference_skin","<f4");
        double skinError=0,attachmentError=0;mlcloth::demo::Json clips=mlcloth::demo::Json::array();
        for(const auto& clip:assets.animations){
            mlcloth::demo::Pose pose;
            if(clip.bones.size()!=assets.body.inverseBind.size())throw std::runtime_error("Incompatible body skeleton");
            for(double fraction:{0.,.137,.513,.997,1.}){
                clip.sample(clip.duration*fraction,false,pose);
                for(int axis=0;axis<3;++axis)if(pose.rootWorld[axis]<clip.rootPathMin[axis]-.0001f||pose.rootWorld[axis]>clip.rootPathMax[axis]+.0001f)
                    throw std::runtime_error("Root path bounds disagree with full hierarchy sampling: "+clip.id);
            }
            auto data=mlcloth::demo::AssetPack::read(assets.directory/(clip.id+".dma"));
            double error=clip.validateReferences(data);
            auto blender=mlcloth::demo::AssetPack::read(assets.directory/(clip.id+".reference.dmp"));
            auto sampled=blender.array<uint32_t>("frames","<u4");auto direct=blender.array<glm::vec3>("skin","<f4");
            auto bones=blender.array<float>("bones","<f4");
            auto validatedBones=blender.array<uint32_t>("validated_bone_indices","<u4");
            if(validatedBones.size()!=1015)throw std::runtime_error("Expected all 1015 original Blender bones");
            if(direct.size()!=sampled.size()*assets.body.positions.size()||bones.size()!=sampled.size()*clip.bones.size()*16)
                throw std::runtime_error("Invalid independent Blender reference shape");
            double bonePosition=0,boneAngle=0,clipSkin=0;
            for(size_t sample=0;sample<sampled.size();++sample){clip.sample(double(sampled[sample])/60,false,pose);
                std::vector<glm::vec3> skinned;assets.body.skin(pose,skinned);
                for(size_t v=0;v<skinned.size();++v)clipSkin=std::max(clipSkin,double(glm::length(skinned[v]-direct[sample*skinned.size()+v])));
                if(!assets.cloth.attachmentVertices.empty()){
                    std::vector<glm::vec3> pins;assets.cloth.pins(pose,pins,&assets.body);
                    for(size_t v=0;v<pins.size();++v)if(assets.cloth.pinned[v]){
                        auto ids=assets.cloth.attachmentVertices[v];auto offsetIndex=sample*skinned.size();
                        auto a=direct[offsetIndex+ids.x],b=direct[offsetIndex+ids.y],c=direct[offsetIndex+ids.z];
                        auto normal=glm::normalize(glm::cross(b-a,c-a)),tangent=glm::normalize(b-a),bitangent=glm::cross(normal,tangent);
                        auto bc=assets.cloth.attachmentBary[v],offset=assets.cloth.attachmentOffset[v];
                        auto expected=a*bc.x+b*bc.y+c*bc.z+tangent*offset.x+bitangent*offset.y+normal*offset.z;
                        attachmentError=std::max(attachmentError,double(glm::length(pins[v]-expected)));
                    }
                }
                for(auto j:validatedBones){if(j>=clip.bones.size())throw std::runtime_error("Blender bone index out of range");glm::dmat4 expected(1);
                    for(int r=0;r<4;++r)for(int c=0;c<4;++c)expected[c][r]=bones[(sample*clip.bones.size()+j)*16+r*4+c];
                    glm::dmat4 actual(pose.componentCm[j]);bonePosition=std::max(bonePosition,glm::length(glm::dvec3(actual[3]-expected[3]))*.01);
                    auto qa=glm::normalize(glm::quat_cast(glm::dmat3(actual))),qb=glm::normalize(glm::quat_cast(glm::dmat3(expected)));
                    boneAngle=std::max(boneAngle,glm::degrees(2*std::acos(std::clamp(std::abs(glm::dot(qa,qb)),0.,1.))));
                }
            }
            if(bonePosition>.001||boneAngle>.1||clipSkin>.001)throw std::runtime_error("Animated Blender reference tolerance exceeded: "+clip.id);
            skinError=std::max(skinError,clipSkin);
            for(double t:{0.0,clip.duration*.25,clip.duration*.5,clip.duration,clip.duration*20.0+.1}){
                clip.sample(t,true,pose);
                for(const auto& m:pose.world)for(int c=0;c<4;++c)for(int r=0;r<4;++r)if(!std::isfinite(m[c][r]))throw std::runtime_error("Nonfinite pose");
            }
            clip.sample(0,false,pose);
            if(clip.id=="complex"){
                std::vector<glm::vec3> skinned;assets.body.skin(pose,skinned);
                for(size_t i=0;i<skinned.size();++i)skinError=std::max(skinError,double(glm::length(skinned[i]-reference[i])));
            }
            clips.push_back({{"clip",clip.id},{"frames",clip.frameCount},{"duration",clip.duration},{"reference_error",error},
                {"blender_bone_position_m",bonePosition},{"blender_bone_angle_degrees",boneAngle},{"blender_body_skin_m",clipSkin}});
        }
        if(skinError>.001||attachmentError>.001)throw std::runtime_error("Skinning or attachment error exceeds 1 mm");
        mlcloth::demo::Json report={{"passed",true},{"full_skeleton_bones",assets.body.inverseBind.size()},{"body_skin_max_error_m",skinError},
            {"surface_attachments",!assets.cloth.attachmentVertices.empty()},{"attachment_max_error_m",attachmentError},{"clips",clips}};
        std::ofstream(assets.directory/"asset-validation.json")<<report.dump(2);std::cout<<report.dump(2)<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
