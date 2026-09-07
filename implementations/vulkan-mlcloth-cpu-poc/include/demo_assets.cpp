#include "demo_assets.h"
#include <glm/gtc/matrix_transform.hpp>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>

namespace mlcloth::demo {
std::vector<uint8_t> readBytes(const std::filesystem::path& path) {
    std::ifstream stream(path,std::ios::binary|std::ios::ate);
    if(!stream) throw std::runtime_error("Cannot open "+path.string());
    const auto end=stream.tellg();
    if(end<0 || uint64_t(end)>uint64_t(2)*1024*1024*1024) throw std::runtime_error("Invalid asset size");
    std::vector<uint8_t> result(static_cast<size_t>(end)); stream.seekg(0);
    if(!stream.read(reinterpret_cast<char*>(result.data()),static_cast<std::streamsize>(result.size())))
        throw std::runtime_error("Truncated asset "+path.string());
    return result;
}
AssetPack AssetPack::read(const std::filesystem::path& path) {
    auto data=readBytes(path); uint64_t length{};
    if(data.size()<16 || std::memcmp(data.data(),"DMPACK01",8)) throw std::runtime_error("Invalid DMPACK01 asset");
    std::memcpy(&length,data.data()+8,8);
    if(length>data.size()-16 || length>16*1024*1024) throw std::runtime_error("Invalid pack header length");
    AssetPack result;
    result.meta=Json::parse(data.begin()+16,data.begin()+16+static_cast<ptrdiff_t>(length));
    if(result.meta.at("version")!=1) throw std::runtime_error("Unsupported demo asset version");
    result.payload.assign(data.begin()+16+static_cast<ptrdiff_t>(length),data.end());
    if(sha256_hex(sha256(result.payload))!=result.meta.at("sha256").get<std::string>())
        throw std::runtime_error("Demo asset checksum failed: "+path.string());
    return result;
}
glm::mat4 componentToWorld() {
    glm::mat4 m(1); m[0]=glm::vec4(.01f,0,0,0);m[1]=glm::vec4(0,0,-.01f,0);m[2]=glm::vec4(0,.01f,0,0);return m;
}
void blendPoses(const Pose& a,const Pose& b,const std::vector<int>& parents,float alpha,Pose& out){
    if(a.componentCm.size()!=parents.size()||b.componentCm.size()!=parents.size())throw std::runtime_error("Pose blend skeleton mismatch");
    out.componentCm.resize(parents.size());out.world.resize(parents.size());
    for(size_t i=0;i<parents.size();++i){auto localA=a.componentCm[i],localB=b.componentCm[i];
        if(parents[i]>=0){localA=glm::inverse(a.componentCm[parents[i]])*localA;localB=glm::inverse(b.componentCm[parents[i]])*localB;}
        auto rotation=glm::normalize(glm::slerp(glm::quat_cast(localA),glm::quat_cast(localB),alpha));
        auto local=glm::mat4_cast(rotation);local[3]=glm::mix(localA[3],localB[3],alpha);
        out.componentCm[i]=parents[i]>=0?out.componentCm[parents[i]]*local:local;out.world[i]=componentToWorld()*out.componentCm[i];
    }
}
void Animation::load(const std::filesystem::path& file,const std::vector<std::string>& driverNames) {
    const auto p=AssetPack::read(file); id=p.meta.at("name").get<std::string>();
    bones=p.meta.at("bones").get<std::vector<std::string>>();parents=p.meta.at("parents").get<std::vector<int>>();
    if(bones.empty()||bones.size()>4096||parents.size()!=bones.size()) throw std::runtime_error("Invalid skeleton");
    for(size_t i=0;i<parents.size();++i) if(parents[i]>=static_cast<int>(i)||parents[i]<-1) throw std::runtime_error("Skeleton must be parent ordered");
    root=p.meta.at("root"); if(root<0||root>=static_cast<int>(bones.size())) throw std::runtime_error("Invalid root");
    local=p.array<float>("local","<f4");
    if(local.size()%(bones.size()*7)) throw std::runtime_error("Invalid pose array size");
    frameCount=static_cast<int>(local.size()/(bones.size()*7));
    if(frameCount<2 || p.meta.at("fps")!=60) throw std::runtime_error("Expected 60 Hz poses");
    duration=double(frameCount-1)/60;
    for(int axis=0;axis<3;++axis) rootDeltaCm[axis]=p.meta.at("root_delta_cm").at(axis).get<float>();
    for(const auto& driver:driverNames) {
        auto it=std::find(bones.begin(),bones.end(),driver);
        if(it==bones.end()) throw std::runtime_error("Missing ML driver "+driver);
        drivers.push_back(static_cast<int>(it-bones.begin()));
    }
    for(float v:local) if(!std::isfinite(v)) throw std::runtime_error("Non-finite animation");
    for(size_t i=0;i<local.size();i+=7) {
        float norm=0; for(int j=3;j<7;++j) norm+=local[i+j]*local[i+j];
        if(std::abs(norm-1)>1e-3f) throw std::runtime_error("Invalid pose quaternion");
    }
    if(validateReferences(p)>.001) throw std::runtime_error("Animation disagrees with Blender reference");
    std::vector<int> chain;for(int bone=root;bone>=0;bone=parents[bone])chain.push_back(bone);
    std::reverse(chain.begin(),chain.end());rootPathMin=glm::vec3(1e30f);rootPathMax=glm::vec3(-1e30f);
    // Only the root's ancestor chain is needed for layout. Include quarter-frame
    // samples so parent rotation between 60 Hz keys contributes to the path bounds.
    for(int tick=0;tick<=(frameCount-1)*4;++tick){const int a=tick/4,b=std::min(a+1,frameCount-1);const float u=float(tick%4)*.25f;
        glm::mat4 component(1);
        for(auto bone:chain){const auto* x=local.data()+(size_t(a)*bones.size()+bone)*7;const auto* y=local.data()+(size_t(b)*bones.size()+bone)*7;
            auto matrix=glm::mat4_cast(glm::normalize(glm::slerp(glm::quat(x[6],x[3],x[4],x[5]),glm::quat(y[6],y[3],y[4],y[5]),u)));
            matrix[3]=glm::vec4(glm::mix(glm::vec3(x[0],x[1],x[2]),glm::vec3(y[0],y[1],y[2]),u),1);component*=matrix;}
        const auto point=glm::vec3(componentToWorld()*component[3]);rootPathMin=glm::min(rootPathMin,point);rootPathMax=glm::max(rootPathMax,point);
    }
}
void Animation::sample(double seconds,bool loop,Pose& out) const {
    if(!std::isfinite(seconds)||seconds<0) throw std::runtime_error("Invalid animation time");
    double cycle=loop?std::floor(seconds/duration):0;
    double t=loop?seconds-cycle*duration:std::min(seconds,duration);
    const double frame=t*60; const int a=std::min(static_cast<int>(frame),frameCount-1),b=std::min(a+1,frameCount-1);
    const float u=static_cast<float>(frame-a); const size_t count=bones.size();
    out.componentCm.resize(count);out.world.resize(count);
    std::vector<glm::quat> rotations(count);
    for(size_t i=0;i<count;++i) {
        const float* x=local.data()+(size_t(a)*count+i)*7;const float* y=local.data()+(size_t(b)*count+i)*7;
        const glm::vec3 translation=glm::mix(glm::vec3(x[0],x[1],x[2]),glm::vec3(y[0],y[1],y[2]),u);
        rotations[i]=glm::normalize(glm::slerp(glm::quat(x[6],x[3],x[4],x[5]),glm::quat(y[6],y[3],y[4],y[5]),u));
        glm::mat4 m=glm::mat4_cast(rotations[i]);m[3]=glm::vec4(translation,1);
        if(parents[i]>=0) m=out.componentCm[parents[i]]*m;
        out.componentCm[i]=m;
    }
    const glm::vec3 shift=rootDeltaCm*static_cast<float>(cycle);
    for(size_t i=0;i<count;++i) {out.componentCm[i][3]+=glm::vec4(shift,0);out.world[i]=componentToWorld()*out.componentCm[i];}
    // ML features retain UE local axes and centimetres, separately from render space.
    out.localFu.resize(drivers.size()*6);out.componentFu.resize(drivers.size()*6);out.positionsCm.resize(drivers.size()*3);
    for(size_t i=0;i<drivers.size();++i) {
        const int j=drivers[i];const auto r=glm::mat3_cast(rotations[j]);const auto c=glm::mat3(out.componentCm[j]);
        for(int k=0;k<3;++k) {out.localFu[i*6+k]=r[2][k];out.localFu[i*6+3+k]=r[1][k];
            out.componentFu[i*6+k]=c[2][k];out.componentFu[i*6+3+k]=c[1][k];out.positionsCm[i*3+k]=out.componentCm[j][3][k];}
    }
    out.rootWorld=glm::vec3(out.world[root][3]);
}
double Animation::validateReferences(const AssetPack& p) const {
    auto indices=p.array<uint32_t>("reference_frames","<u4");auto matrices=p.array<float>("reference_bones","<f4");
    if(matrices.size()!=indices.size()*bones.size()*16) throw std::runtime_error("Invalid reference matrices");
    double maximum=0;Pose pose;
    for(size_t s=0;s<indices.size();++s) {
        sample(double(indices[s])/60,false,pose);
        for(size_t j=0;j<bones.size();++j) for(int r=0;r<3;++r) for(int c=0;c<4;++c) {
            const double error=std::abs(pose.componentCm[j][c][r]-matrices[(s*bones.size()+j)*16+r*4+c]);
            maximum=std::max(maximum,c==3?error*.01:error); // metres for positions; matrix-entry rotation error
        }
    }
    return maximum;
}
void CharacterAsset::load(const AssetPack& p,const std::vector<glm::mat4>* sharedBind) {
    positions=p.array<glm::vec3>("positions","<f4");normals=p.array<glm::vec3>("normals","<f4");uv=p.array<glm::vec2>("uv","<f4");
    triangles=p.array<uint32_t>("triangles","<u4");material=p.array<uint32_t>("material","<u4");
    boneIds=p.array<uint32_t>("bone_ids","<u4");weights=p.array<float>("weights","<f4");influences=p.meta.at("influences");
    if(sharedBind) inverseBind=*sharedBind;
    else {
        auto raw=p.array<float>("inverse_bind","<f4"); if(raw.size()%12) throw std::runtime_error("Invalid inverse bind array");
        inverseBind.assign(raw.size()/12,glm::mat4(1));
        for(size_t i=0;i<inverseBind.size();++i) for(int r=0;r<3;++r) for(int c=0;c<4;++c) inverseBind[i][c][r]=raw[i*12+r*4+c];
    }
    if(positions.empty()||normals.size()!=positions.size()||uv.size()!=positions.size()||influences==0||influences>32||
       boneIds.size()!=positions.size()*influences||weights.size()!=boneIds.size()||triangles.size()%3||material.size()!=triangles.size()/3)
        throw std::runtime_error("Invalid character arrays");
    for(auto i:triangles) if(i>=positions.size()) throw std::runtime_error("Body triangle out of range");
    for(size_t i=0;i<positions.size();++i) {double sum=0;for(size_t k=0;k<influences;++k){const size_t j=i*influences+k;
        if(boneIds[j]>=inverseBind.size()||weights[j]<0||!std::isfinite(weights[j])) throw std::runtime_error("Invalid body weights");sum+=weights[j];}
        if(std::abs(sum-1)>1e-4) throw std::runtime_error("Body weights do not sum to one");}
}
void CharacterAsset::skin(const Pose& pose,std::vector<glm::vec3>& out) const {
    std::vector<glm::mat4> matrices(inverseBind.size());
    for(size_t i=0;i<matrices.size();++i)matrices[i]=pose.world[i]*inverseBind[i];
    out.assign(positions.size(),glm::vec3(0));
    for(size_t i=0;i<positions.size();++i)for(size_t k=0;k<influences;++k){size_t j=i*influences+k;
        if(weights[j]>0)out[i]+=glm::vec3(matrices[boneIds[j]]*glm::vec4(positions[i],1))*weights[j];}
}
void ClothAsset::load(const AssetPack& p) {
    rest=p.array<glm::vec3>("positions","<f4");localCm=p.array<glm::vec3>("local_cm","<f4");triangles=p.array<uint32_t>("triangles","<u4");
    pinned=p.array<uint32_t>("pinned","<u4");mass=p.array<float>("mass","<f4");boneIds=p.array<uint32_t>("bone_ids","<u4");
    weights=p.array<float>("weights","<f4");bindLocal=p.array<glm::vec3>("bind_local","<f4");
    if(rest.size()!=5294||mass.size()!=rest.size()||pinned.size()!=rest.size()||localCm.size()!=rest.size()||triangles.size()%3||
       boneIds.size()!=rest.size()*influences||weights.size()!=boneIds.size()||bindLocal.size()!=boneIds.size()) throw std::runtime_error("Invalid cloth arrays");
    for(auto index:triangles)if(index>=rest.size())throw std::runtime_error("Cloth triangle out of range");
    if(p.meta.at("arrays").count("attachment_vertices")){
        attachmentVertices=p.array<glm::uvec3>("attachment_vertices","<u4");
        attachmentBary=p.array<glm::vec3>("attachment_bary","<f4");attachmentOffset=p.array<glm::vec3>("attachment_offset","<f4");
        if(attachmentVertices.size()!=rest.size()||attachmentBary.size()!=rest.size()||attachmentOffset.size()!=rest.size())throw std::runtime_error("Invalid surface attachment arrays");
    }
}
void ClothAsset::pins(const Pose& pose,std::vector<glm::vec3>& out,const CharacterAsset* body,bool skinAll) const {
    out=rest;
    const bool surface=body&&!attachmentVertices.empty();
    for(size_t i=0;i<rest.size();++i)if((pinned[i]&&!surface)||skinAll){out[i]=glm::vec3(0);
        for(size_t k=0;k<influences;++k){size_t j=i*influences+k;if(boneIds[j]>=pose.world.size())throw std::runtime_error("Cloth bone out of range");
            if(weights[j]>0)out[i]+=glm::vec3(pose.world[boneIds[j]]*glm::vec4(bindLocal[j],1))*weights[j];}
    }
    if(!surface)return;
    std::vector<glm::mat4> palette(body->inverseBind.size());
    for(size_t i=0;i<palette.size();++i)palette[i]=pose.world.at(i)*body->inverseBind[i];
    std::vector<glm::vec3> vertices(body->positions.size());std::vector<uint8_t> ready(vertices.size());
    auto skinVertex=[&](uint32_t v){if(v>=vertices.size())throw std::runtime_error("Attachment body vertex out of range");
        if(!ready[v]){vertices[v]=glm::vec3(0);for(uint32_t k=0;k<body->influences;++k){auto slot=v*body->influences+k;
                if(body->weights[slot]>0)vertices[v]+=glm::vec3(palette[body->boneIds[slot]]*glm::vec4(body->positions[v],1))*body->weights[slot];}ready[v]=1;}
        return vertices[v];};
    for(size_t i=0;i<rest.size();++i)if(pinned[i]){auto ids=attachmentVertices[i];auto a=skinVertex(ids.x),b=skinVertex(ids.y),c=skinVertex(ids.z);
        auto normal=glm::cross(b-a,c-a);if(glm::length(normal)<1e-10f)throw std::runtime_error("Degenerate attachment surface");normal=glm::normalize(normal);
        auto tangent=glm::normalize(b-a),bitangent=glm::cross(normal,tangent),bc=attachmentBary[i],offset=attachmentOffset[i];
        out[i]=a*bc.x+b*bc.y+c*bc.z+tangent*offset.x+bitangent*offset.y+normal*offset.z;
    }
}
void DemoAssetManifest::load(const std::filesystem::path& manifest) {
    directory=std::filesystem::absolute(manifest).parent_path(); std::ifstream stream(manifest);
    if(!stream)throw std::runtime_error("Demo assets are missing. Run prepare_demo.ps1 first.");
    stream>>provenance;if(provenance.at("version")!=1)throw std::runtime_error("Unsupported demo manifest");
    model=(directory/provenance.at("model").get<std::string>()).lexically_normal();runtime=model.parent_path();
    modelBytes=readBytes(model);std::string error;
    if(!parse_model(modelBytes.data(),modelBytes.size(),modelInfo,error))throw std::runtime_error(error);
    const auto hash=sha256_hex(sha256(modelBytes));if(hash!=provenance.at("model_sha256"))throw std::runtime_error("Demo model hash mismatch");
    body.load(AssetPack::read(directory/provenance.at("body").get<std::string>()));
    collision.load(AssetPack::read(directory/provenance.at("collision").get<std::string>()),&body.inverseBind);
    capsules.clear();
    if(provenance.count("capsules")){
        auto pack=AssetPack::read(directory/provenance.at("capsules").get<std::string>());
        auto bones=pack.array<uint32_t>("bone","<u4");auto a=pack.array<glm::vec3>("a","<f4"),b=pack.array<glm::vec3>("b","<f4");auto radii=pack.array<float>("radius","<f4");
        if(bones.size()>64||a.size()!=bones.size()||b.size()!=bones.size()||radii.size()!=bones.size())throw std::runtime_error("Invalid capsule arrays");
        for(size_t i=0;i<bones.size();++i){bool finite=true;for(int axis=0;axis<3;++axis)finite&=std::isfinite(a[i][axis])&&std::isfinite(b[i][axis]);
            if(!finite||bones[i]>=body.inverseBind.size()||!std::isfinite(radii[i])||radii[i]<=0)throw std::runtime_error("Invalid capsule binding");
            capsules.push_back({bones[i],a[i],b[i],radii[i]});}
    }
    const auto mesh=AssetPack::read(directory/provenance.at("cloth").get<std::string>());
    if(mesh.meta.at("model_sha256")!=hash)throw std::runtime_error("Cloth/model mismatch");cloth.load(mesh);
    for(const auto& entry:provenance.at("clips")) {
        Animation clip;clip.load(directory/entry.at("animation").get<std::string>(),modelInfo.driverNames);
        clip.name=entry.at("name");clip.warmup=entry.value("warmup",0.0);animations.push_back(std::move(clip));
    }
    if(animations.empty())throw std::runtime_error("Empty demo animation set");
    for(const auto& p:provenance.at("textures"))textures.push_back(directory/p.get<std::string>());
}
}
