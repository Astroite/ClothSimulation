#pragma once
#include "mlcloth_formats.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <json.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace mlcloth::demo {
using Json = nlohmann::json;

struct AssetPack {
    Json meta;
    std::vector<uint8_t> payload;
    static AssetPack read(const std::filesystem::path& path);
    template<class T> std::vector<T> array(const char* key, const char* dtype) const {
        const auto& item=meta.at("arrays").at(key);
        const size_t offset=item.at("offset").get<size_t>(), bytes=item.at("bytes").get<size_t>();
        if(item.at("dtype")!=dtype || bytes%sizeof(T) || offset>payload.size() || bytes>payload.size()-offset)
            throw std::runtime_error(std::string("Invalid asset array: ")+key);
        std::vector<T> result(bytes/sizeof(T));
        std::memcpy(result.data(),payload.data()+offset,bytes); return result;
    }
};

struct Pose {
    std::vector<glm::mat4> componentCm;
    std::vector<glm::mat4> world;
    std::vector<float> localFu,componentFu,positionsCm;
    glm::vec3 rootWorld{};
};

struct Animation {
    std::string id,name;
    std::vector<std::string> bones;
    std::vector<int> parents,drivers;
    std::vector<float> local;
    int frameCount{},root{};
    double duration{},warmup{};
    glm::vec3 rootDeltaCm{};
    glm::vec3 rootPathMin{},rootPathMax{}; // sampled first-cycle world-space trajectory
    void load(const std::filesystem::path& file,const std::vector<std::string>& driverNames);
    void sample(double seconds,bool loop,Pose& out) const;
    double validateReferences(const AssetPack& pack) const;
};

struct CharacterAsset {
    std::vector<glm::vec3> positions,normals;
    std::vector<glm::vec2> uv;
    std::vector<uint32_t> triangles,material,boneIds;
    std::vector<float> weights;
    std::vector<glm::mat4> inverseBind;
    uint32_t influences{};
    void load(const AssetPack& data,const std::vector<glm::mat4>* sharedBind=nullptr);
    void skin(const Pose& pose,std::vector<glm::vec3>& out) const;
};

struct ClothAsset {
    std::vector<glm::vec3> rest,localCm,bindLocal;
    std::vector<uint32_t> triangles,pinned,boneIds;
    std::vector<float> mass,weights;
    std::vector<glm::uvec3> attachmentVertices;
    std::vector<glm::vec3> attachmentBary,attachmentOffset;
    uint32_t influences{8};
    void load(const AssetPack& data);
    void pins(const Pose& pose,std::vector<glm::vec3>& out,const CharacterAsset* body=nullptr,bool skinAll=false) const;
};

struct DemoAssetManifest {
    struct CapsuleBinding { uint32_t bone{};glm::vec3 a{},b{};float radius{}; };
    std::filesystem::path directory,model,runtime;
    std::vector<std::filesystem::path> textures;
    std::vector<Animation> animations;
    CharacterAsset body,collision;
    ClothAsset cloth;
    std::vector<CapsuleBinding> capsules;
    std::vector<uint8_t> modelBytes;
    mlcloth::ModelInfo modelInfo;
    Json provenance;
    void load(const std::filesystem::path& manifest);
};

glm::mat4 componentToWorld();
void blendPoses(const Pose&,const Pose&,const std::vector<int>& parents,float alpha,Pose&);
std::vector<uint8_t> readBytes(const std::filesystem::path& path);
}
