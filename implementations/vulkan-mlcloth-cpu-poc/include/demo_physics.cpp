#include "demo_physics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mlcloth::demo {
void BodySurfaceTopology::build(uint32_t count,const std::vector<uint32_t>& indices){
    if(indices.size()%3)throw std::runtime_error("Invalid body surface triangle count");
    vertices=count;triangles=indices;offsets.assign(count+1,0);incident.clear();edgeNeighbors.resize(indices.size());
    std::map<std::pair<uint32_t,uint32_t>,uint32_t> edges;
    for(uint32_t slot=0;slot<indices.size();++slot){
        if(indices[slot]>=count)throw std::runtime_error("Body surface vertex out of range");
        ++offsets[indices[slot]+1];edgeNeighbors[slot]=slot/3;
        const uint32_t next=(slot/3)*3+(slot+1)%3;auto key=std::minmax(indices[slot],indices[next]);
        auto found=edges.find(key);if(found==edges.end())edges[key]=slot;
        else{edgeNeighbors[slot]=found->second/3;edgeNeighbors[found->second]=slot/3;}
    }
    for(uint32_t v=0;v<count;++v)offsets[v+1]+=offsets[v];
    incident.resize(indices.size());auto cursor=offsets;
    for(uint32_t slot=0;slot<indices.size();++slot)incident[cursor[indices[slot]]++]=slot/3;
}
std::vector<uint32_t> BodySurfaceTopology::packed()const{
    std::vector<uint32_t> data{vertices,static_cast<uint32_t>(triangles.size()/3),vertices+5,vertices+5+static_cast<uint32_t>(incident.size())};
    data.insert(data.end(),offsets.begin(),offsets.end());data.insert(data.end(),incident.begin(),incident.end());data.insert(data.end(),edgeNeighbors.begin(),edgeNeighbors.end());return data;
}
std::vector<Vec3> bodySurfaceNormals(const TriangleCollider& body,const BodySurfaceTopology& topology){
    const auto n=topology.vertices;const auto count=static_cast<uint32_t>(topology.triangles.size()/3);std::vector<Vec3> result(n+4*count);
    for(uint32_t t=0;t<count;++t){auto a=body.current[body.triangles[3*t]],b=body.current[body.triangles[3*t+1]],c=body.current[body.triangles[3*t+2]];
        result[n+4*t]=normalize(cross(b-a,c-a));}
    for(uint32_t v=0;v<n;++v){Vec3 sum{};
        for(uint32_t i=topology.offsets[v];i<topology.offsets[v+1];++i){auto t=topology.incident[i];uint32_t corner=0;while(corner<2&&body.triangles[3*t+corner]!=v)++corner;
            auto e1=body.current[body.triangles[3*t+(corner+1)%3]]-body.current[v],e2=body.current[body.triangles[3*t+(corner+2)%3]]-body.current[v];
            const float angle=std::atan2(length(cross(e1,e2)),dot(e1,e2));sum+=result[n+4*t]*angle;}
        result[v]=normalize(sum);
    }
    for(uint32_t t=0;t<count;++t)for(uint32_t k=0;k<3;++k)result[n+4*t+1+k]=normalize(result[n+4*t]+result[n+4*topology.edgeNeighbors[3*t+k]]);
    return result;
}
Vec3 bodyFeatureNormal(const BodySurfaceTopology& topology,const std::vector<Vec3>& normals,uint32_t triangle,Vec3 bary){
    const float weights[]={bary.x,bary.y,bary.z};const uint32_t base=topology.vertices+4*triangle;Vec3 normal=normals[base];
    for(uint32_t k=0;k<3;++k)if(weights[k]>=1-1e-6f){normal=normals[topology.triangles[3*triangle+k]];return length(normal)>.5f?normal:normals[base];}
    for(uint32_t k=0;k<3;++k)if(weights[(k+2)%3]<=1e-6f){normal=normals[base+1+k];break;}
    return length(normal)>.5f?normal:normals[base];
}
namespace {

Vec3 distanceNormal(Vec3 delta,Vec3 pseudo,float distance){
    return distance>1e-8f?delta*((dot(delta,pseudo)<0?-1.f:1.f)/distance):pseudo;
}
Vec3 bodyFeatureNormalAt(const TriangleCollider& body,const BodySurfaceTopology& topology,uint32_t tri,Vec3 bary,float time){
    auto point=[&](uint32_t v){return body.previous[v]+(body.current[v]-body.previous[v])*time;};
    auto face=[&](uint32_t t){auto a=point(body.triangles[3*t]),b=point(body.triangles[3*t+1]),c=point(body.triangles[3*t+2]);return normalize(cross(b-a,c-a));};
    const float weights[]={bary.x,bary.y,bary.z};auto normal=face(tri);
    for(uint32_t k=0;k<3;++k)if(weights[k]>=1-1e-6f){
        const auto v=body.triangles[3*tri+k];Vec3 sum{};
        for(auto i=topology.offsets[v];i<topology.offsets[v+1];++i){const auto t=topology.incident[i];uint32_t corner=0;
            while(corner<2&&body.triangles[3*t+corner]!=v)++corner;
            auto e1=point(body.triangles[3*t+(corner+1)%3])-point(v),e2=point(body.triangles[3*t+(corner+2)%3])-point(v);
            if(dot(e1,e1)*dot(e2,e2)>1e-20f)sum+=face(t)*std::atan2(length(cross(e1,e2)),dot(e1,e2));}
        auto result=normalize(sum);return length(result)>.5f?result:normal;
    }
    for(uint32_t k=0;k<3;++k)if(weights[(k+2)%3]<=1e-6f){auto result=normalize(normal+face(topology.edgeNeighbors[3*tri+k]));return length(result)>.5f?result:normal;}
    return normal;
}

constexpr float kEpsilon = 1e-12f;
constexpr float kDenominatorFloor = 1e-20f;

inline float alphaTilde(float compliance, float dt) {
    return compliance / std::max(dt * dt, kEpsilon);
}

inline bool isFinite(Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// AABB utilities
AABB mergeAABB(const AABB& a, const AABB& b) {
    return {
        { std::min(a.min.x, b.min.x), std::min(a.min.y, b.min.y), std::min(a.min.z, b.min.z) },
        { std::max(a.max.x, b.max.x), std::max(a.max.y, b.max.y), std::max(a.max.z, b.max.z) }
    };
}

AABB pointBounds(Vec3 p, float r) {
    return { { p.x - r, p.y - r, p.z - r }, { p.x + r, p.y + r, p.z + r } };
}

bool intersects(const AABB& a, const AABB& b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x &&
           a.min.y <= b.max.y && a.max.y >= b.min.y &&
           a.min.z <= b.max.z && a.max.z >= b.min.z;
}

// Signed distance from point to plane defined by triangle (a, b, c).
// Positive = same side as normal (right-hand rule a->b->c).

// Point-triangle closest point with barycentric coordinates
float pointTriangleDistSqBary(Vec3 p, Vec3 a, Vec3 b, Vec3 c,
                               Vec3& closest, float& bu, float& bv, float* b0=nullptr) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 area=cross(ab,ac);const float edge2=std::max({dot(ab,ab),dot(ac,ac),dot(c-b,c-b)});
    if(dot(area,area)<=1e-12f*edge2*edge2){
        float best=std::numeric_limits<float>::infinity();Vec3 bary{};
        auto segment=[&](Vec3 x,Vec3 y,Vec3 bx,Vec3 by){auto edge=y-x;float norm=dot(edge,edge);
            float t=norm>1e-20f?std::clamp(dot(p-x,edge)/norm,0.f,1.f):0.f;auto point=x+edge*t;
            float distance=dot(p-point,p-point);if(distance<best){best=distance;closest=point;bary=bx+(by-bx)*t;}};
        segment(a,b,{1,0,0},{0,1,0});segment(b,c,{0,1,0},{0,0,1});segment(c,a,{0,0,1},{1,0,0});
        bu=bary.y;bv=bary.z;if(b0)*b0=bary.x;return best;
    }
    const Vec3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { closest = a; bu = 0; bv = 0; if(b0)*b0=1; return dot(ap, ap); }

    const Vec3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { closest = b; bu = 1; bv = 0; if(b0)*b0=0; return dot(bp, bp); }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        closest = a + ab * v;
        bu = v; bv = 0; if(b0)*b0=1-v;
        const Vec3 d = p - closest;
        return dot(d, d);
    }

    const Vec3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { closest = c; bu = 0; bv = 1; if(b0)*b0=0; return dot(cp, cp); }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        closest = a + ac * w;
        bu = 0; bv = w; if(b0)*b0=1-w;
        const Vec3 d = p - closest;
        return dot(d, d);
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest = b + (c - b) * w;
        bu = 1.0f - w; bv = w; if(b0)*b0=0;
        const Vec3 d = p - closest;
        return dot(d, d);
    }

    if(std::abs(va+vb+vc)<1e-20f){closest=a;bu=0;bv=0;if(b0)*b0=1;const auto delta=p-a;return dot(delta,delta);}
    const float denom = 1.0f / (va + vb + vc);
    const float v = std::clamp(vb * denom,0.f,1.f);
    const float w = std::clamp(vc * denom,0.f,1.f-v);
    closest = a + ab * v + ac * w;
    bu = v; bv = w; if(b0)*b0=std::max(0.f,1-v-w);
    const Vec3 d = p - closest;
    return dot(d, d);
}

// Edge-edge distance (squared)
float edgeEdgeDistSq(Vec3 a0, Vec3 a1, Vec3 b0, Vec3 b1, Vec3& pa, Vec3& pb, float& s, float& t) {
    const Vec3 d1 = a1 - a0;
    const Vec3 d2 = b1 - b0;
    const Vec3 r = a0 - b0;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);
    s=0; t=0;

    if (a <= 1e-16f && e <= 1e-16f) {
        pa = a0; pb = b0;
        const Vec3 d = pa - pb;
        return dot(d, d);
    }
    if (a <= 1e-16f) {
        s = 0.0f;
        t = std::clamp(f / e, 0.0f, 1.0f);
    } else {
        const float c = dot(d1, r);
        if (e <= 1e-16f) {
            t = 0.0f;
            s = std::clamp(-c / a, 0.0f, 1.0f);
        } else {
            const float b_val = dot(d1, d2);
            const float denom = a * e - b_val * b_val;
            s = std::abs(denom)>1e-16f ? std::clamp((b_val * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;
            t = (b_val * s + f) / e;
            if (t < 0.0f) { t = 0.0f; s = std::clamp(-c / a, 0.0f, 1.0f); }
            else if (t > 1.0f) { t = 1.0f; s = std::clamp((b_val - c) / a, 0.0f, 1.0f); }
        }
    }
    pa = a0 + d1 * s;
    pb = b0 + d2 * t;
    const Vec3 d = pa - pb;
    return dot(d, d);
}

// ---------------------------------------------------------------------------
// BVH for body collision — stores ALL triangles in leaves
// ---------------------------------------------------------------------------

void computeLeafBounds(BVHNode& node,
                       const std::vector<uint32_t>& triIndices,
                       const std::vector<Vec3>& positions,
                       const std::vector<uint32_t>& triangles) {
    node.bounds = { {1e18f, 1e18f, 1e18f}, {-1e18f, -1e18f, -1e18f} };
    for (uint32_t i = node.triBegin; i < node.triEnd; ++i) {
        const uint32_t tri = triIndices[i];
        for (int k = 0; k < 3; ++k) {
            const Vec3& p = positions[triangles[3 * tri + k]];
            node.bounds.min = { std::min(node.bounds.min.x, p.x), std::min(node.bounds.min.y, p.y), std::min(node.bounds.min.z, p.z) };
            node.bounds.max = { std::max(node.bounds.max.x, p.x), std::max(node.bounds.max.y, p.y), std::max(node.bounds.max.z, p.z) };
        }
    }
}

uint32_t buildBodyBVHRecursive(std::vector<BVHNode>& nodes,
                                std::vector<uint32_t>& indices,
                                const std::vector<Vec3>& positions,
                                const std::vector<uint32_t>& triangles,
                                uint32_t begin, uint32_t end) {
    const uint32_t nodeIndex = static_cast<uint32_t>(nodes.size());
    nodes.push_back(BVHNode{});
    nodes[nodeIndex].triBegin = begin;
    nodes[nodeIndex].triEnd = end;

    // Compute bounds for all triangles in this node
    computeLeafBounds(nodes[nodeIndex], indices, positions, triangles);

    constexpr uint32_t kMaxLeafSize = 8;
    if (end - begin <= kMaxLeafSize) {
        nodes[nodeIndex].isLeaf = true;
        return nodeIndex;
    }

    // Split along longest axis at median
    const Vec3 extent = nodes[nodeIndex].bounds.max - nodes[nodeIndex].bounds.min;
    int axis = 0;
    if (extent.y > extent.x && extent.y > extent.z) axis = 1;
    else if (extent.z > extent.x && extent.z > extent.y) axis = 2;

    const uint32_t mid = (begin + end) / 2;
    std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end,
        [&](uint32_t a, uint32_t b) {
            Vec3 ca{}, cb{};
            for (int k = 0; k < 3; ++k) {
                ca = ca + positions[triangles[3 * a + k]];
                cb = cb + positions[triangles[3 * b + k]];
            }
            ca = ca / 3.0f;
            cb = cb / 3.0f;
            return axis == 0 ? ca.x < cb.x : (axis == 1 ? ca.y < cb.y : ca.z < cb.z);
        });

    nodes[nodeIndex].left = buildBodyBVHRecursive(nodes, indices, positions, triangles, begin, mid);
    nodes[nodeIndex].right = buildBodyBVHRecursive(nodes, indices, positions, triangles, mid, end);
    nodes[nodeIndex].isLeaf = false;
    return nodeIndex;
}

void refitBodyBVH(std::vector<BVHNode>& nodes,
                  const std::vector<uint32_t>& triIndices,
                  const std::vector<Vec3>& positions,
                  const std::vector<uint32_t>& triangles) {
    // Refit bottom-up (nodes are stored in DFS order, so reverse traversal works)
    for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
        if (nodes[i].isLeaf) {
            computeLeafBounds(nodes[i], triIndices, positions, triangles);
        } else {
            nodes[i].bounds = mergeAABB(nodes[nodes[i].left].bounds, nodes[nodes[i].right].bounds);
        }
    }
}

// Traverse BVH and call visitor(nodeIndex) for each leaf whose AABB intersects the query AABB.
template<typename F>
void traverseBVH(const std::vector<BVHNode>& nodes, uint32_t root,
                 const AABB& query, F&& visitor) {
    if (root == 0xFFFFFFFFu) return;
    std::vector<uint32_t> stack;
    stack.push_back(root);
    while (!stack.empty()) {
        const uint32_t ni = stack.back();
        stack.pop_back();
        if (!intersects(nodes[ni].bounds, query)) continue;
        if (nodes[ni].isLeaf) {
            visitor(ni);
        } else {
            stack.push_back(nodes[ni].left);
            stack.push_back(nodes[ni].right);
        }
    }
}

// Build adjacency from edge pairs
std::vector<std::vector<uint32_t>> buildAdjacency(uint32_t vertexCount,
                                                    const std::vector<uint32_t>& pairs) {
    std::vector<std::vector<uint32_t>> adj(vertexCount);
    const uint32_t pairCount = static_cast<uint32_t>(pairs.size() / 2);
    for (uint32_t i = 0; i < pairCount; ++i) {
        const uint32_t a = pairs[2 * i];
        const uint32_t b = pairs[2 * i + 1];
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    return adj;
}

}  // namespace

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

void PhysicsSolver::build(const Mesh& mesh) {
    if (mesh.rest.empty()) throw std::runtime_error("mesh has no vertices");
    if (mesh.triangles.empty()) throw std::runtime_error("mesh has no triangles");
    if (mesh.rest.size() != mesh.mass.size())
        throw std::runtime_error("mass count != vertex count");
    if (mesh.rest.size() != mesh.pinned.size())
        throw std::runtime_error("pinned count != vertex count");
    if (mesh.triangles.size() % 3 != 0)
        throw std::runtime_error("triangle indices not a multiple of 3");

    vertexCount_ = static_cast<uint32_t>(mesh.rest.size());
    triangles_ = mesh.triangles;

    inverseMass_.resize(vertexCount_);
    pinned_ = mesh.pinned;
    for (uint32_t i = 0; i < vertexCount_; ++i) {
        if (mesh.mass[i] <= 0.0f || !std::isfinite(mesh.mass[i]))
            throw std::runtime_error("invalid mass at vertex " + std::to_string(i));
        if (!isFinite(mesh.rest[i]))
            throw std::runtime_error("non-finite rest position at vertex " + std::to_string(i));
        inverseMass_[i] = pinned_[i] ? 0.0f : 1.0f / mesh.mass[i];
    }

    // Validate triangle indices
    const uint32_t triCount = static_cast<uint32_t>(mesh.triangles.size() / 3);
    for (uint32_t t = 0; t < triCount; ++t) {
        for (int k = 0; k < 3; ++k) {
            if (mesh.triangles[3 * t + k] >= vertexCount_)
                throw std::runtime_error("triangle index out of range at triangle " + std::to_string(t));
        }
    }

    positions_ = mesh.rest;
    velocities_.assign(vertexCount_, Vec3{});

    buildConstraints(mesh);
    buildCoarsePatches(mesh);
    colorConstraints();
    buildSelfExclusions(mesh);
    buildTethers(mesh);
}
void PhysicsSolver::buildTethers(const Mesh& mesh){
    const auto adjacent=buildAdjacency(vertexCount_,stretchPairs_);
    tethers_.assign(vertexCount_,{});
    std::vector<uint8_t> visited(vertexCount_);
    for(uint32_t source=0;source<vertexCount_;++source){
        if(!pinned_[source]||visited[source])continue;
        // Nearby pins in one seam form one source region; four slots should not all
        // select adjacent neck samples while missing a separate cuff attachment.
        std::vector<uint32_t> region{source};visited[source]=1;
        for(size_t i=0;i<region.size();++i)for(auto next:adjacent[region[i]])
            if(pinned_[next]&&!visited[next]){visited[next]=1;region.push_back(next);}
        using Entry=std::pair<float,uint32_t>;
        std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> queue;
        std::vector<float> distances(vertexCount_,std::numeric_limits<float>::infinity());
        std::vector<uint32_t> anchors(vertexCount_,~0u);
        for(auto v:region){distances[v]=0;anchors[v]=v;queue.push({0.f,v});}
        while(!queue.empty()){
            auto [distance,v]=queue.top();queue.pop();if(distance>distances[v])continue;
            for(auto next:adjacent[v]){
                const float candidate=distance+length(mesh.rest[next]-mesh.rest[v]);
                if(candidate<distances[next]||(candidate==distances[next]&&anchors[v]<anchors[next])){
                    distances[next]=candidate;anchors[next]=anchors[v];queue.push({candidate,next});}
            }
        }
        for(uint32_t v=0;v<vertexCount_;++v){
            if(pinned_[v]||!std::isfinite(distances[v])||distances[v]<=0)continue;
            Tether candidate{anchors[v],distances[v]};
            for(auto& slot:tethers_[v]){
                if(slot.length==0){slot=candidate;break;}
                if(candidate.length<slot.length||(candidate.length==slot.length&&candidate.anchor<slot.anchor))std::swap(candidate,slot);
            }
        }
    }
}
void PhysicsSolver::applyTethers(const std::vector<Vec3>& pins,float scale){
    // One writer per free particle. Read kinematic targets, never another mutable
    // free position. Average unilateral corrections; compressed cloth stays free.
    for(uint32_t v=0;v<vertexCount_;++v){
        if(pinned_[v])continue;
        Vec3 correction{};uint32_t count=0;
        for(auto tether:tethers_[v])if(tether.length>0){
            ++count;const auto delta=positions_[v]-pins[tether.anchor];const float distance=length(delta);
            const float limit=tether.length*scale;
            if(distance>limit)correction-=delta*((distance-limit)/distance);
        }
        if(count)positions_[v]+=correction/static_cast<float>(count);
    }
}
void PhysicsSolver::buildSelfExclusions(const Mesh& mesh){
    // A 3 mm contact shell must not separate samples only 0.4 mm apart on the same
    // cloth sheet. Use rest geodesic distance, so nearby folds on remote regions still collide.
    const auto adjacent=buildAdjacency(vertexCount_,stretchPairs_);
    selfOffsets_={0};selfVertices_.clear();selfDistances_.clear();
    for(uint32_t source=0;source<vertexCount_;++source){
        std::map<uint32_t,float> distances;distances[source]=0;
        using Entry=std::pair<float,uint32_t>;
        std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> queue;queue.push({0.f,source});
        while(!queue.empty()){auto [distance,v]=queue.top();queue.pop();if(distance>distances[v])continue;
            for(auto next:adjacent[v]){float candidate=distance+length(mesh.rest[next]-mesh.rest[v]);if(candidate>.020001f)continue;
                auto found=distances.find(next);if(found==distances.end()||candidate<found->second){distances[next]=candidate;queue.push({candidate,next});}}
        }
        // Direct topological neighbors are always excluded, regardless of edge length.
        for(auto v:adjacent[source])distances[v]=0;
        for(auto entry:distances){selfVertices_.push_back(entry.first);selfDistances_.push_back(entry.second);}
        selfOffsets_.push_back(static_cast<uint32_t>(selfVertices_.size()));
    }
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void PhysicsSolver::reset(const std::vector<Vec3>& positions) {
    if (positions.size() != vertexCount_)
        throw std::runtime_error("position count mismatch");
    positions_ = positions;
    velocities_.assign(vertexCount_, Vec3{});
}

void PhysicsSolver::reset(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities) {
    if (positions.size() != vertexCount_)
        throw std::runtime_error("position count mismatch");
    if (velocities.size() != vertexCount_)
        throw std::runtime_error("velocity count mismatch");
    positions_ = positions;
    velocities_ = velocities;
}

// ---------------------------------------------------------------------------
// Constraint building
// ---------------------------------------------------------------------------

void PhysicsSolver::buildConstraints(const Mesh& mesh) {
    const uint32_t triCount = static_cast<uint32_t>(mesh.triangles.size() / 3);

    // Build edge set from triangles (bidirectional stretch)
    std::set<std::pair<uint32_t, uint32_t>> edgeSet;
    for (uint32_t t = 0; t < triCount; ++t) {
        for (int k = 0; k < 3; ++k) {
            uint32_t a = mesh.triangles[3 * t + k];
            uint32_t b = mesh.triangles[3 * t + (k + 1) % 3];
            edgeSet.insert({ std::min(a, b), std::max(a, b) });
        }
    }

    // Stretch constraints (edge pairs, bidirectional)
    stretchPairs_.clear();
    stretchRestLen_.clear();
    for (const auto& [a, b] : edgeSet) {
        stretchPairs_.push_back(a);
        stretchPairs_.push_back(b);
        stretchRestLen_.push_back(length(mesh.rest[a] - mesh.rest[b]));
    }

    // Shear constraints: triangle dot-product/strain metric.
    // For each triangle (a, b, c), the shear constraint is:
    //   C = dot(b-a, c-a) - restDot
    // where restDot is computed at rest pose.
    // Gradient: dC/da = -(e1+e2), dC/db = e2, dC/dc = e1
    //   where e1 = b-a, e2 = c-a.
    shearConstraints_.clear();
    shearPairs_.clear();
    shearRestLen_.clear();
    for (uint32_t t = 0; t < triCount; ++t) {
        const uint32_t i0 = mesh.triangles[3 * t];
        const uint32_t i1 = mesh.triangles[3 * t + 1];
        const uint32_t i2 = mesh.triangles[3 * t + 2];
        const Vec3 e1 = mesh.rest[i1] - mesh.rest[i0];
        const Vec3 e2 = mesh.rest[i2] - mesh.rest[i0];
        ShearConstraint sc;
        sc.a = i0; sc.b = i1; sc.c = i2;
        sc.restDot = dot(e1, e2);
        shearConstraints_.push_back(sc);
        // Backward-compatible flat pair form (a,b), (b,c), (c,a) with rest lengths
        shearPairs_.push_back(i0); shearPairs_.push_back(i1);
        shearRestLen_.push_back(length(mesh.rest[i0] - mesh.rest[i1]));
        shearPairs_.push_back(i1); shearPairs_.push_back(i2);
        shearRestLen_.push_back(length(mesh.rest[i1] - mesh.rest[i2]));
        shearPairs_.push_back(i2); shearPairs_.push_back(i0);
        shearRestLen_.push_back(length(mesh.rest[i2] - mesh.rest[i0]));
    }

    // Bend constraints: four-point signed dihedral angle.
    // For each pair of triangles sharing edge (a, b) with opposite vertices c, d:
    //   angle = atan2(dot(cross(n1hat, n2hat), ehat), dot(n1hat, n2hat))
    //   where n1 = cross(b-a, c-a), n2 = cross(b-a, d-a), ehat = normalize(b-a).
    //   C = angle - restAngle
    // Gradient:
    //   hc = |n1|/|e|, hd = |n2|/|e|
    //   tc = dot(c-a, e)/|e|^2, td = dot(d-a, e)/|e|^2
    //   dC/dc = -n1hat/hc, dC/dd = +n2hat/hd
    //   dC/da = -(1-tc)*dC/dc - (1-td)*dC/dd
    //   dC/db = -tc*dC/dc - td*dC/dd
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edgeOwners;
    for (uint32_t t = 0; t < triCount; ++t) {
        for (int k = 0; k < 3; ++k) {
            uint32_t a = mesh.triangles[3 * t + k];
            uint32_t b = mesh.triangles[3 * t + (k + 1) % 3];
            edgeOwners[{ std::min(a, b), std::max(a, b) }].push_back(t);
        }
    }

    bendConstraints_.clear();
    bendPairs_.clear();
    bendRestLen_.clear();
    std::set<std::pair<uint32_t, uint32_t>> uniqueBend;
    for (const auto& [edge, tris] : edgeOwners) {
        if (tris.size() != 2) continue;
        uint32_t opposite[2] = { vertexCount_, vertexCount_ };
        for (int side = 0; side < 2; ++side) {
            for (int k = 0; k < 3; ++k) {
                const uint32_t corner = mesh.triangles[3 * tris[side] + k];
                if (corner != edge.first && corner != edge.second) opposite[side] = corner;
            }
        }
        if (opposite[0] >= vertexCount_ || opposite[1] >= vertexCount_) continue;
        if (opposite[0] == opposite[1]) continue;

        const uint32_t a = edge.first;
        const uint32_t b = edge.second;
        const uint32_t c = opposite[0];
        const uint32_t d = opposite[1];

        // Compute rest dihedral angle
        const Vec3 e = mesh.rest[b] - mesh.rest[a];
        const Vec3 n1 = cross(e, mesh.rest[c] - mesh.rest[a]);
        const Vec3 n2 = cross(e, mesh.rest[d] - mesh.rest[a]);
        const float n1Len = length(n1);
        const float n2Len = length(n2);
        float restAngle = 0.0f;
        if (n1Len > kEpsilon && n2Len > kEpsilon) {
            const Vec3 n1hat = n1 / n1Len;
            const Vec3 n2hat = n2 / n2Len;
            const Vec3 ehat = e / length(e);
            const float cosA = std::clamp(dot(n1hat, n2hat), -1.0f, 1.0f);
            const float sinA = dot(cross(n1hat, n2hat), ehat);
            restAngle = std::atan2(sinA, cosA);
        }

        BendConstraint bc;
        bc.a = a; bc.b = b; bc.c = c; bc.d = d;
        bc.restAngle = restAngle;
        bendConstraints_.push_back(bc);

        // Backward-compatible flat pair form (opposite corners as distance pairs)
        auto pair = std::make_pair(std::min(c, d), std::max(c, d));
        if (uniqueBend.insert(pair).second) {
            bendPairs_.push_back(pair.first);
            bendPairs_.push_back(pair.second);
            bendRestLen_.push_back(length(mesh.rest[pair.first] - mesh.rest[pair.second]));
        }
    }
}

// ---------------------------------------------------------------------------
// Coarse patches for guide projection
// ---------------------------------------------------------------------------

void PhysicsSolver::buildCoarsePatches(const Mesh& /*mesh*/) {
    auto adjacency = buildAdjacency(vertexCount_, stretchPairs_);

    // BFS to assign patches, bounded to ~32 vertices each.
    // No crossing disconnected components.
    patchAssignment_.assign(vertexCount_, 0xFFFFFFFFu);
    patchOffsets_.clear();
    patchVertices_.clear();
    patchCount_ = 0;

    constexpr uint32_t kMaxPatchSize = 32;

    for (uint32_t seed = 0; seed < vertexCount_; ++seed) {
        if (patchAssignment_[seed] != 0xFFFFFFFFu) continue;
        if (pinned_[seed]) continue;

        // BFS from seed, max kMaxPatchSize vertices per patch
        std::queue<uint32_t> queue;
        queue.push(seed);
        patchAssignment_[seed] = patchCount_;
        patchOffsets_.push_back(static_cast<uint32_t>(patchVertices_.size()));

        uint32_t count = 0;
        while (!queue.empty() && count < kMaxPatchSize) {
            const uint32_t v = queue.front();
            queue.pop();
            if (patchAssignment_[v] != patchCount_) continue;
            patchVertices_.push_back(v);
            ++count;

            for (uint32_t neighbor : adjacency[v]) {
                if (patchAssignment_[neighbor] == 0xFFFFFFFFu && !pinned_[neighbor]) {
                    patchAssignment_[neighbor] = patchCount_;
                    queue.push(neighbor);
                }
            }
        }

        // Un-assign remaining frontier vertices so they become seeds for new patches
        while (!queue.empty()) {
            patchAssignment_[queue.front()] = 0xFFFFFFFFu;
            queue.pop();
        }

        if (count > 0) ++patchCount_;
    }

    // Assign pinned vertices to their own single-vertex patches
    for (uint32_t v = 0; v < vertexCount_; ++v) {
        if (pinned_[v]) {
            patchAssignment_[v] = patchCount_++;
            patchOffsets_.push_back(static_cast<uint32_t>(patchVertices_.size()));
            patchVertices_.push_back(v);
        }
    }

    // Final sentinel so patchOffsets_.size() == patchCount_ + 1
    patchOffsets_.push_back(static_cast<uint32_t>(patchVertices_.size()));
}

// ---------------------------------------------------------------------------
// Constraint coloring (for GPU parallelism)
// ---------------------------------------------------------------------------

void PhysicsSolver::colorConstraints() {
    auto color = [&](const std::vector<std::vector<uint32_t>>& records) {
        ConstraintGroups out;
        std::vector<std::vector<uint8_t>> occupied;
        std::vector<std::vector<uint32_t>> colors;
        for(uint32_t i=0;i<records.size();++i) {
            size_t g=0;
            for(;g<colors.size();++g) {
                bool conflict=false;
                for(auto v:records[i]) conflict|=occupied[g][v]!=0;
                if(!conflict)break;
            }
            if(g==colors.size()){colors.emplace_back();occupied.emplace_back(vertexCount_,uint8_t{0});}
            colors[g].push_back(i);for(auto v:records[i])occupied[g][v]=1;
        }
        for(const auto& group:colors) {
            out.groupStart.push_back(static_cast<uint32_t>(out.constraintIndices.size()));
            out.groupCount.push_back(static_cast<uint32_t>(group.size()));
            out.constraintIndices.insert(out.constraintIndices.end(),group.begin(),group.end());
        }
        out.groupCountN=static_cast<uint32_t>(colors.size());return out;
    };
    std::vector<std::vector<uint32_t>> records;
    for(size_t i=0;i<stretchPairs_.size();i+=2)records.push_back({stretchPairs_[i],stretchPairs_[i+1]});
    stretchGroups_=color(records);records.clear();
    for(auto c:shearConstraints_)records.push_back({c.a,c.b,c.c});
    shearGroups_=color(records);records.clear();
    for(auto c:bendConstraints_)records.push_back({c.a,c.b,c.c,c.d});
    bendGroups_=color(records);
}

// ---------------------------------------------------------------------------
// Step
// ---------------------------------------------------------------------------

void PhysicsSolver::step(float dt,
                         const std::vector<Vec3>& pinTargets,
                         const std::vector<Vec3>* guide,
                         const TriangleCollider* body,
                         const PhysicsConfig& config) {
    // Validate inputs BEFORE any indexed accesses
    if(config.gnnAcceleration)throw std::runtime_error("GNN acceleration requires the Vulkan backend");
    if (dt <= 0.0f || !std::isfinite(dt))
        throw std::runtime_error("invalid dt");
    if (pinTargets.size() != vertexCount_)
        throw std::runtime_error("pinTargets count mismatch");
    if (config.dampingPerSecond < 0.0f || !std::isfinite(config.dampingPerSecond))
        throw std::runtime_error("dampingPerSecond must be >= 0 and finite");
    if (!std::isfinite(config.gravity))
        throw std::runtime_error("gravity must be finite");
    if(!std::isfinite(config.tetherScale)||config.tetherScale<1)
        throw std::runtime_error("tetherScale must be finite and >= 1");
    if (guide) {
        if (guide->size() != vertexCount_)
            throw std::runtime_error("guide size mismatch");
    }
    if (body) {
        if (body->previous.size() != body->current.size())
            throw std::runtime_error("body previous/current size mismatch");
        const uint32_t bodyTriCount = static_cast<uint32_t>(body->triangles.size() / 3);
        for (uint32_t t = 0; t < bodyTriCount; ++t) {
            for (int k = 0; k < 3; ++k) {
                if (body->triangles[3 * t + k] >= static_cast<uint32_t>(body->current.size()))
                    throw std::runtime_error("body triangle index out of range");
            }
        }
    }

    // Validate all positions and velocities are finite
    for (uint32_t i = 0; i < vertexCount_; ++i) {
        if (!isFinite(positions_[i]))
            throw std::runtime_error("non-finite position at vertex " + std::to_string(i));
        if (!isFinite(velocities_[i]))
            throw std::runtime_error("non-finite velocity at vertex " + std::to_string(i));
    }

    stats_ = PhysicsStats{};
    previous_ = positions_;
    const auto& prevPositions = previous_;

    // 1. Pin assignment BEFORE structural passes and contact tests
    for (uint32_t i = 0; i < vertexCount_; ++i) {
        if (pinned_[i]) {
            positions_[i] = pinTargets[i];
            velocities_[i] = {};
        }
    }

    // Store previous positions for velocity reconstruction

    // Damping: exp(-dampingPerSecond * dt). Zero damping preserves velocity.
    const float damping = std::exp(-config.dampingPerSecond * dt);
    for (uint32_t i = 0; i < vertexCount_; ++i) {
        if (!pinned_[i]) {
            velocities_[i] = velocities_[i] * damping;
        }
    }

    // Gravity
    for (uint32_t i = 0; i < vertexCount_; ++i) {
        if (!pinned_[i]) {
            velocities_[i].y += config.gravity * dt;
        }
    }

    // Semi-implicit Euler
    for (uint32_t i = 0; i < vertexCount_; ++i) {
        if (!pinned_[i]) {
            positions_[i] = positions_[i] + velocities_[i] * dt;
        }
    }

    // Resize scratch
    accumulator_.resize(vertexCount_, Vec3{});
    stretchMultiplier_.resize(stretchPairs_.size() / 2, 0.0f);
    shearMultiplier_.resize(shearConstraints_.size(), 0.0f);
    bendMultiplier_.resize(bendConstraints_.size(), 0.0f);
    guideLambda_.resize(patchCount_, Vec3{});

    // Reset multipliers per step
    std::fill(stretchMultiplier_.begin(), stretchMultiplier_.end(), 0.0f);
    std::fill(shearMultiplier_.begin(), shearMultiplier_.end(), 0.0f);
    std::fill(bendMultiplier_.begin(), bendMultiplier_.end(), 0.0f);
    std::fill(guideLambda_.begin(), guideLambda_.end(), Vec3{});
    if(body&&config.enableCollision)prepareBodyCollision(*body,config.thickness);
    guideContacts_.clear();
    if(guide&&config.guideCompliance>=0&&config.enableCollision&&config.contactAwareGuide)
        prepareGuideContacts(body,config.thickness);

    // XPBD iterations — apply each constraint type group-by-group in-place
    for (int iter = 0; iter < config.iterations; ++iter) {
        applyStretch(positions_, config.stretchCompliance, dt);
        applyShear(positions_, config.shearCompliance, dt);
        applyBend(positions_, config.bendCompliance, dt);

        if (guide && config.guideCompliance >= 0.0f) {
            applyGuide(*guide, positions_, config.guideCompliance, dt);
        }
        if(config.enableTethers)applyTethers(pinTargets,config.tetherScale);
    }

    if(config.enableSelfCollision)applySelfCollision(positions_,config.thickness);

    // Body collision (position correction + contact recording)
    bodyContacts_.clear();
    if (body && config.enableCollision) {
        applyBodyCollision(positions_, velocities_, *body, config.thickness, config.friction, dt);
        applyCapsuleCollision(positions_,body->capsules,config.thickness,dt);
        recoverBodyContacts(positions_,*body,config.thickness,dt);
    }
    std::vector<float> groundImpulse(vertexCount_);
    if(config.enableCollision)for(uint32_t v=0;v<vertexCount_;++v)if(!pinned_[v]&&positions_[v].y<config.thickness){
        groundImpulse[v]=(config.thickness-positions_[v].y)/dt;positions_[v].y=config.thickness;}

    // Velocity reconstruction from position change
    for (uint32_t i = 0; i < vertexCount_; ++i) {
        if (!pinned_[i]) {
            velocities_[i] = (positions_[i] - prevPositions[i]) / dt;
        }
    }

    // Body-relative velocity friction (applied AFTER velocity reconstruction)
    if (body && config.enableCollision) {
        for (const auto& contact : bodyContacts_) {
            const uint32_t v = contact.vertex;
            if (pinned_[v]||groundImpulse[v]>0) continue;
            const Vec3 bodyVelAtContact=contact.bodyVelocity;

            const Vec3 relVel = velocities_[v] - bodyVelAtContact;
            const float vn = dot(relVel, contact.normal);
            const Vec3 tangentialVel = relVel - contact.normal * vn;
            const float tangLen = length(tangentialVel);
            if (tangLen > kEpsilon) {
                const float maxFriction = contact.normalImpulse * config.friction;
                const float reduction = std::min(tangLen, maxFriction);
                velocities_[v] -= normalize(tangentialVel) * reduction;
            }
            if(vn<0||contact.recovery)velocities_[v]-=contact.normal*vn;
        }
    }

    // Hard pin overwrite at end
    for(uint32_t v=0;v<vertexCount_;++v)if(groundImpulse[v]>0){
        Vec3 tangent{velocities_[v].x,0,velocities_[v].z};float speed=length(tangent);
        if(speed>1e-9f)velocities_[v]-=tangent*std::min(1.f,config.friction*groundImpulse[v]/speed);
        velocities_[v].y=std::max(0.f,velocities_[v].y);}
    for (uint32_t i = 0; i < vertexCount_; ++i) {
        if (pinned_[i]) {
            positions_[i] = pinTargets[i];
            velocities_[i] = {};
        }
    }
}

// ---------------------------------------------------------------------------
// Constraint solvers
// ---------------------------------------------------------------------------

void PhysicsSolver::applyStretch(std::vector<Vec3>& position,
                                 float compliance, float dt) {
    const uint32_t count = static_cast<uint32_t>(stretchPairs_.size() / 2);
    if (count == 0) return;

    const float alpha = alphaTilde(compliance, dt);

    // Apply each group in-place before the next group (correct group-by-group application).
    for (uint32_t g = 0; g < stretchGroups_.groupCountN; ++g) {
        std::fill(accumulator_.begin(), accumulator_.end(), Vec3{});

        for (uint32_t idx = 0; idx < stretchGroups_.groupCount[g]; ++idx) {
            const uint32_t ci = stretchGroups_.constraintIndices[stretchGroups_.groupStart[g] + idx];
            const uint32_t a = stretchPairs_[2 * ci];
            const uint32_t b = stretchPairs_[2 * ci + 1];

            const Vec3 delta = position[a] - position[b];
            const float dist = length(delta);
            if (dist < kEpsilon) continue;

            // Bidirectional: correct both stretch (residual > 0) and compression (residual < 0)
            const float residual = dist - stretchRestLen_[ci];

            const float wSum = inverseMass_[a] + inverseMass_[b];
            const float denom = std::max(wSum + alpha, kDenominatorFloor);
            const float step = (-residual - alpha * stretchMultiplier_[ci]) / denom;
            stretchMultiplier_[ci] += step;

            const Vec3 grad = delta / dist;
            accumulator_[a] += grad * step;
            accumulator_[b] -= grad * step;
        }

        // Apply this group's corrections in-place
        for (uint32_t i = 0; i < vertexCount_; ++i) {
            if (pinned_[i]) continue;
            const Vec3 corr = accumulator_[i] * inverseMass_[i];
            position[i] += corr;
            if (length(corr) > stats_.maxCorrection)
                stats_.maxCorrection = length(corr);
        }
        stats_.stretchActive += stretchGroups_.groupCount[g];
    }
}

void PhysicsSolver::applyShear(std::vector<Vec3>& position,float compliance,float dt) {
    const float alpha=alphaTilde(compliance,dt);
    for(auto ci:shearGroups_.constraintIndices) {
        const auto& sc=shearConstraints_[ci];
        const Vec3 e1=position[sc.b]-position[sc.a],e2=position[sc.c]-position[sc.a];
        const Vec3 ga=-e1-e2,gb=e2,gc=e1;
        const float denominator=inverseMass_[sc.a]*dot(ga,ga)+inverseMass_[sc.b]*dot(gb,gb)+inverseMass_[sc.c]*dot(gc,gc)+alpha;
        if(denominator<1e-12f)continue;
        const float dl=(-(dot(e1,e2)-sc.restDot)-alpha*shearMultiplier_[ci])/denominator;
        shearMultiplier_[ci]+=dl;
        position[sc.a]+=ga*(inverseMass_[sc.a]*dl);
        position[sc.b]+=gb*(inverseMass_[sc.b]*dl);
        position[sc.c]+=gc*(inverseMass_[sc.c]*dl);
        ++stats_.shearActive;
    }
}

void PhysicsSolver::applyBend(std::vector<Vec3>& position,
                              float compliance, float dt) {
    if (bendConstraints_.empty()) return;

    const float alpha = alphaTilde(compliance, dt);

    // Bend constraint: four-point signed dihedral angle.
    // For edge (a, b) shared by triangles (a,b,c) and (a,b,d):
    //   n1 = cross(b-a, c-a), n2 = cross(b-a, d-a)
    //   angle = atan2(dot(cross(n1hat,n2hat), ehat), dot(n1hat,n2hat))
    //   C = angle - restAngle
    // Gradients:
    //   hc = |n1|/|e|, hd = |n2|/|e|
    //   tc = dot(c-a, e)/|e|^2, td = dot(d-a, e)/|e|^2
    //   dC/dc = -n1hat/hc, dC/dd = +n2hat/hd
    //   dC/da = -(1-tc)*dC/dc - (1-td)*dC/dd
    //   dC/db = -tc*dC/dc - td*dC/dd

    const uint32_t bendTriCount = static_cast<uint32_t>(bendConstraints_.size());

    // All four vertices participate in coloring; each record is visited exactly once.
    for (uint32_t g = 0; g < bendGroups_.groupCountN; ++g) {
        std::fill(accumulator_.begin(), accumulator_.end(), Vec3{});

        std::set<uint32_t> processedBends;

        for (uint32_t idx = 0; idx < bendGroups_.groupCount[g]; ++idx) {
            const uint32_t pi = bendGroups_.constraintIndices[bendGroups_.groupStart[g] + idx];
            // Group entries index the dihedral records directly.
            if (pi >= bendTriCount) continue;
            if (processedBends.count(pi)) continue;
            processedBends.insert(pi);

            const auto& bc = bendConstraints_[pi];
            const uint32_t a = bc.a, b = bc.b, c = bc.c, d = bc.d;

            const Vec3 e = position[b] - position[a];
            const float eLen = length(e);
            if (eLen < kEpsilon) continue;
            const Vec3 ehat = e / eLen;

            const Vec3 n1 = cross(e, position[c] - position[a]);
            const Vec3 n2 = cross(e, position[d] - position[a]);
            const float n1Len = length(n1);
            const float n2Len = length(n2);
            if (n1Len < kEpsilon || n2Len < kEpsilon) continue;

            const Vec3 n1hat = n1 / n1Len;
            const Vec3 n2hat = n2 / n2Len;

            const float cosA = std::clamp(dot(n1hat, n2hat), -1.0f, 1.0f);
            const float sinA = dot(cross(n1hat, n2hat), ehat);
            const float angle = std::atan2(sinA, cosA);

            const float C = std::atan2(std::sin(angle-bc.restAngle),std::cos(angle-bc.restAngle));

            // Heights
            const float hc = n1Len / eLen;
            const float hd = n2Len / eLen;

            // Projection parameters
            const float tc = dot(position[c] - position[a], e) / dot(e, e);
            const float td = dot(position[d] - position[a], e) / dot(e, e);

            // Gradients of C w.r.t. each vertex
            const Vec3 grad_c = -n1hat / hc;
            const Vec3 grad_d = n2hat / hd;
            const Vec3 grad_a = grad_c * (-(1.0f - tc)) + grad_d * (-(1.0f - td));
            const Vec3 grad_b = grad_c * (-tc) + grad_d * (-td);

            const float wSum = inverseMass_[a] * dot(grad_a, grad_a) +
                               inverseMass_[b] * dot(grad_b, grad_b) +
                               inverseMass_[c] * dot(grad_c, grad_c) +
                               inverseMass_[d] * dot(grad_d, grad_d);
            const float denom = std::max(wSum + alpha, kDenominatorFloor);
            const float dLambda = (-C - alpha * bendMultiplier_[pi]) / denom;
            bendMultiplier_[pi] += dLambda;

            accumulator_[a] += grad_a * dLambda;
            accumulator_[b] += grad_b * dLambda;
            accumulator_[c] += grad_c * dLambda;
            accumulator_[d] += grad_d * dLambda;
        }

        for (uint32_t i = 0; i < vertexCount_; ++i) {
            if (pinned_[i]) continue;
            const Vec3 corr = accumulator_[i] * inverseMass_[i];
            position[i] += corr;
        }
        stats_.bendActive += static_cast<uint32_t>(processedBends.size());
    }
}

void PhysicsSolver::applyGuide(const std::vector<Vec3>& guide, std::vector<Vec3>& position,
                               float compliance, float dt) {
    const float alpha = alphaTilde(compliance, dt);

    // Vector XPBD lambda (Vec3 per patch, not scalar)
    for (uint32_t p = 0; p < patchCount_; ++p) {
        const uint32_t begin = patchOffsets_[p];
        const uint32_t end = patchOffsets_[p + 1];
        if (begin == end) continue;

        // Compute patch centroid (mass-weighted)
        Vec3 centroid{};
        float totalMass = 0.0f;
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t v = patchVertices_[i];
            const float w = inverseMass_[v] > 0.0f ? 1.0f / inverseMass_[v] : 0.0f;
            centroid += position[v] * w;
            totalMass += w;
        }
        if (totalMass < kEpsilon) continue;
        centroid = centroid / totalMass;

        // Guide centroid
        Vec3 guideCentroid{};
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t v = patchVertices_[i];
            const float w = inverseMass_[v] > 0.0f ? 1.0f / inverseMass_[v] : 0.0f;
            guideCentroid += guide[v] * w;
        }
        guideCentroid = guideCentroid / totalMass;

        // Vector constraint: C = centroid - guideCentroid (XYZ)
        const Vec3 C = centroid - guideCentroid;
        const float dist = length(C);
        if (dist < kEpsilon) continue;
        float confidence=1;
        if(!guideContacts_.empty())for(uint32_t i=begin;i<end;++i)if(!pinned_[patchVertices_[i]])
            for(const auto& contact:guideContacts_[patchVertices_[i]])
                confidence=std::min(confidence,1-contact.proximity*std::clamp(dot(C,contact.normal)/guideContactBand_,0.f,1.f));
        if(confidence<=1e-6f){guideLambda_[p]={};continue;}

        // Vector XPBD update (component-wise)
        // For centroid constraint: ∂C/∂p_i = mass_i/totalMass
        // ∑ w_i |∂C/∂p_i|² = 1/totalMass
        const float wSum = 1.0f / totalMass;
        const float adjustedAlpha=alpha/(std::max(.01f,std::clamp(1-dist/.5f,0.f,1.f))*confidence);
        const float denom = std::max(wSum + adjustedAlpha, kDenominatorFloor);
        const Vec3 dLambda = (C * (-1.0f) - guideLambda_[p] * adjustedAlpha) / denom * (alpha==0?confidence:1.f);
        guideLambda_[p] += dLambda;

        // Correction: Δp_i = w_i * Δλ * ∂C/∂p_i = (1/mass_i) * Δλ * (mass_i/totalMass) = Δλ/totalMass
        const Vec3 correction = dLambda / totalMass;
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t v = patchVertices_[i];
            if (pinned_[v]) continue;
            position[v] += correction;
        }
    }
}

void PhysicsSolver::prepareBodyCollision(const TriangleCollider& body,float thickness) {
    const uint32_t bodyTriCount = static_cast<uint32_t>(body.triangles.size() / 3);
    if (bodyTriCount == 0) return;
    if(bodySurface_.vertices!=body.current.size()||bodySurface_.triangles!=body.triangles)bodySurface_.build(static_cast<uint32_t>(body.current.size()),body.triangles);
    bodyNormals_=bodySurfaceNormals(body,bodySurface_);

    // Build or rebuild BVH for body triangles using swept AABB (previous + current)
    bodyBvhTriIndices_.resize(bodyTriCount);
    for (uint32_t i = 0; i < bodyTriCount; ++i) bodyBvhTriIndices_[i] = i;

    bodyBvhNodes_.clear();
    // Use swept positions for BVH build
    std::vector<Vec3> sweptPositions(body.previous.size());
    for (size_t i = 0; i < body.previous.size(); ++i) {
        sweptPositions[i] = body.current[i]; // use current for build, refit with swept below
    }
    bodyBvhRoot_ = buildBodyBVHRecursive(bodyBvhNodes_, bodyBvhTriIndices_,
                                          sweptPositions, body.triangles, 0, bodyTriCount);

    // Refit BVH to swept AABB (union of previous and current positions)
    // We need to compute swept bounds for each leaf
    for (auto& node : bodyBvhNodes_) {
        if (node.isLeaf) {
            AABB swept = { {1e18f, 1e18f, 1e18f}, {-1e18f, -1e18f, -1e18f} };
            for (uint32_t i = node.triBegin; i < node.triEnd; ++i) {
                const uint32_t tri = bodyBvhTriIndices_[i];
                for (int k = 0; k < 3; ++k) {
                    const Vec3& pp = body.previous[body.triangles[3 * tri + k]];
                    const Vec3& pc = body.current[body.triangles[3 * tri + k]];
                    swept.min = { std::min({swept.min.x, pp.x, pc.x}),
                                  std::min({swept.min.y, pp.y, pc.y}),
                                  std::min({swept.min.z, pp.z, pc.z}) };
                    swept.max = { std::max({swept.max.x, pp.x, pc.x}),
                                  std::max({swept.max.y, pp.y, pc.y}),
                                  std::max({swept.max.z, pp.z, pc.z}) };
                }
            }
            // Expand by collision thickness
            swept.min.x -= thickness; swept.min.y -= thickness; swept.min.z -= thickness;
            swept.max.x += thickness; swept.max.y += thickness; swept.max.z += thickness;
            node.bounds = swept;
        }
    }
    // Refit internal nodes bottom-up
    for (int i = static_cast<int>(bodyBvhNodes_.size()) - 1; i >= 0; --i) {
        if (!bodyBvhNodes_[i].isLeaf) {
            bodyBvhNodes_[i].bounds = mergeAABB(bodyBvhNodes_[bodyBvhNodes_[i].left].bounds,
                                                 bodyBvhNodes_[bodyBvhNodes_[i].right].bounds);
        }
    }

}

void PhysicsSolver::prepareGuideContacts(const TriangleCollider* body,float thickness){
    const float band=std::max(4*thickness,.012f),reach=thickness+band;guideContactBand_=band;
    guideContacts_.resize(vertexCount_);
    for(uint32_t v=0;v<vertexCount_;++v){
        if(pinned_[v])continue;
        const Vec3 point=positions_[v];
        auto contact=[&](uint32_t slot,Vec3 normal,float distance){
            const float proximity=std::clamp((reach-distance)/band,0.f,1.f);
            if(proximity>guideContacts_[v][slot].proximity)guideContacts_[v][slot]={normal,proximity};
        };
        contact(0,{0,1,0},point.y);
        if(!body)continue;
        for(const auto& capsule:body->capsules){auto axis=capsule.currentB-capsule.currentA;
            const float t=std::clamp(dot(point-capsule.currentA,axis)/std::max(dot(axis,axis),1e-20f),0.f,1.f);
            const auto delta=point-(capsule.currentA+axis*t);const float distance=length(delta);
            if(distance>1e-8f)contact(1,delta/distance,distance-capsule.radius);
        }
        if(body->triangles.empty())continue;
        float nearest=reach;uint32_t selected=~0u;Vec3 normal{},closest{};
        std::vector<uint32_t> stack{bodyBvhRoot_};
        while(!stack.empty()){auto ni=stack.back();stack.pop_back();const auto& node=bodyBvhNodes_[ni];
            Vec3 delta{std::max({node.bounds.min.x-point.x,point.x-node.bounds.max.x,0.f}),std::max({node.bounds.min.y-point.y,point.y-node.bounds.max.y,0.f}),std::max({node.bounds.min.z-point.z,point.z-node.bounds.max.z,0.f})};
            if(dot(delta,delta)>(nearest+1e-6f)*(nearest+1e-6f))continue;
            if(!node.isLeaf){stack.push_back(node.right);stack.push_back(node.left);continue;}
            for(uint32_t item=node.triBegin;item<node.triEnd;++item){auto tri=bodyBvhTriIndices_[item];
                auto a=body->current[body->triangles[3*tri]],b=body->current[body->triangles[3*tri+1]],c=body->current[body->triangles[3*tri+2]];
                if(length(cross(b-a,c-a))<1e-12f)continue;
                Vec3 q;float u{},w{};const float distance=std::sqrt(pointTriangleDistSqBary(point,a,b,c,q,u,w));
                if(distance<nearest-1e-6f||(std::abs(distance-nearest)<=1e-6f&&tri<selected)){
                    nearest=distance;selected=tri;closest=q;normal=distanceNormal(point-q,bodyFeatureNormal(bodySurface_,bodyNormals_,tri,{std::max(0.f,1-u-w),u,w}),distance);}
            }
        }
        if(selected!=~0u)contact(2,normal,dot(point-closest,normal));
    }
}

void PhysicsSolver::applyBodyCollision(std::vector<Vec3>& position, std::vector<Vec3>& /*velocity*/,
                                       const TriangleCollider& body, float thickness,
                                       float /*friction*/, float dt) {
    if(body.triangles.empty())return;
    for(uint32_t v=0;v<vertexCount_;++v) {
        if(pinned_[v])continue;
        const Vec3 predicted=position[v],start=previous_[v];
        AABB bounds=mergeAABB(pointBounds(start,thickness),pointBounds(predicted,thickness));
        float best=1e30f,bestTime=2,nearest=1e30f,nearestSigned=0;BodyContact selected{},embedded{};bool found=false;
        auto visit=[&](uint32_t ni) {
            const auto& node=bodyBvhNodes_[ni];
            for(uint32_t ti=node.triBegin;ti<node.triEnd;++ti) {
                uint32_t t=bodyBvhTriIndices_[ti];
                auto ia=body.triangles[3*t],ib=body.triangles[3*t+1],ic=body.triangles[3*t+2];
                Vec3 a=body.current[ia],b=body.current[ib],c=body.current[ic];
                Vec3 oa=body.previous[ia],ob=body.previous[ib],oc=body.previous[ic];
                Vec3 normal=normalize(cross(b-a,c-a));if(length(normal)<.5f)continue;
                Vec3 q;float u{},w{};
                float distance=std::sqrt(pointTriangleDistSqBary(predicted,a,b,c,q,u,w));
                normal=distanceNormal(predicted-q,bodyFeatureNormal(bodySurface_,bodyNormals_,t,{std::max(0.f,1-u-w),u,w}),distance);
                if(distance<nearest-1e-6f||(std::abs(distance-nearest)<=1e-6f&&t<embedded.bodyTri)){nearest=distance;nearestSigned=dot(predicted-q,normal);embedded={v,normal,q,t,u,w,(thickness-nearestSigned)/dt,true};}
                bool contact=false;
                // Hausdorff motion bound in the particle's translating frame:
                // common root movement cancels, while every relative trajectory remains covered.
                const auto motion=predicted-start;
                const float bound=std::max({length(a-oa-motion),length(b-ob-motion),length(c-oc-motion)});
                float time=0;
                for(int iteration=0;iteration<32&&!contact&&bound>1e-9f&&distance<=bound+thickness+1e-5f;++iteration) {
                    Vec3 at=oa+(a-oa)*time,bt=ob+(b-ob)*time,ct=oc+(c-oc)*time,pt=start+(predicted-start)*time;
                    Vec3 closestAtTime;float bu{},bv{};
                    float gap=std::sqrt(pointTriangleDistSqBary(pt,at,bt,ct,closestAtTime,bu,bv));
                    if(gap<=thickness+1e-5f){contact=true;u=bu;w=bv;q=a+(b-a)*u+(c-a)*w;
                        normal=distanceNormal(pt-closestAtTime,bodyFeatureNormalAt(body,bodySurface_,t,{std::max(0.f,1-u-w),u,w},time),gap);break;}
                    time+=(gap-thickness)/bound;if(time>1)break;
                }
                if(!contact&&distance<thickness){contact=true;time=bound<=1e-9f?0.f:1.f;}
                const float signedDistance=dot(predicted-q,normal);
                const bool earlier=time<bestTime-1e-6f;
                const bool sameTime=std::abs(time-bestTime)<=1e-6f;
                if(contact&&signedDistance<thickness&&(earlier||(sameTime&&(distance<best-1e-6f||(std::abs(distance-best)<=1e-6f&&t<selected.bodyTri))))) {
                    best=distance;bestTime=time;found=true;
                    position[v]=predicted+normal*(thickness-signedDistance);
                    selected={v,normal,q,t,u,w,(thickness-signedDistance)/dt};
                }
            }
        };
        std::vector<uint32_t> stack{bodyBvhRoot_};while(!stack.empty()){
            auto ni=stack.back();stack.pop_back();const auto& node=bodyBvhNodes_[ni];
            Vec3 delta{std::max({node.bounds.min.x-predicted.x,predicted.x-node.bounds.max.x,0.f}),
                std::max({node.bounds.min.y-predicted.y,predicted.y-node.bounds.max.y,0.f}),std::max({node.bounds.min.z-predicted.z,predicted.z-node.bounds.max.z,0.f})};
            if(!intersects(bounds,node.bounds)&&dot(delta,delta)>(nearest+1e-6f)*(nearest+1e-6f))continue;
            if(node.isLeaf)visit(ni);else{stack.push_back(node.right);stack.push_back(node.left);}
        }
        if(!found&&nearestSigned<0){selected=embedded;found=true;position[v]=predicted+embedded.normal*(thickness-nearestSigned);}
        if(found){auto t=selected.bodyTri;auto a=body.triangles[3*t],b=body.triangles[3*t+1],c=body.triangles[3*t+2];
            selected.bodyVelocity=((body.current[a]-body.previous[a])*(1-selected.baryU-selected.baryV)+
                (body.current[b]-body.previous[b])*selected.baryU+(body.current[c]-body.previous[c])*selected.baryV)/dt;
            bodyContacts_.push_back(selected);++stats_.contactsResolved;}
    }
}

void PhysicsSolver::recoverBodyContacts(std::vector<Vec3>& position,const TriangleCollider& body,float thickness,float dt){
    if(body.triangles.empty()||bodyContacts_.empty())return;
    for(int pass=0;pass<2;++pass){std::vector<uint8_t> changed(vertexCount_);bool any=false;
        for(auto& contact:bodyContacts_){const auto vertex=contact.vertex;if(pinned_[vertex])continue;
            const auto point=position[vertex];float nearest=1e30f,signedDistance=0;BodyContact selected{};selected.bodyTri=~0u;
            std::vector<uint32_t> stack{bodyBvhRoot_};
            while(!stack.empty()){auto index=stack.back();stack.pop_back();const auto& node=bodyBvhNodes_[index];
                Vec3 delta{std::max({node.bounds.min.x-point.x,point.x-node.bounds.max.x,0.f}),std::max({node.bounds.min.y-point.y,point.y-node.bounds.max.y,0.f}),std::max({node.bounds.min.z-point.z,point.z-node.bounds.max.z,0.f})};
                if(dot(delta,delta)>(nearest+1e-6f)*(nearest+1e-6f))continue;
                if(!node.isLeaf){stack.push_back(node.right);stack.push_back(node.left);continue;}
                for(uint32_t item=node.triBegin;item<node.triEnd;++item){auto tri=bodyBvhTriIndices_[item];auto ia=body.triangles[3*tri],ib=body.triangles[3*tri+1],ic=body.triangles[3*tri+2];
                    auto a=body.current[ia],b=body.current[ib],c=body.current[ic],normal=normalize(cross(b-a,c-a));if(length(normal)<.5f)continue;
                    Vec3 q;float u{},w{};float distance=std::sqrt(pointTriangleDistSqBary(point,a,b,c,q,u,w));
                    normal=distanceNormal(point-q,bodyFeatureNormal(bodySurface_,bodyNormals_,tri,{std::max(0.f,1-u-w),u,w}),distance);
                    if(distance<nearest-1e-6f||(std::abs(distance-nearest)<=1e-6f&&tri<selected.bodyTri)){nearest=distance;signedDistance=dot(point-q,normal);
                        selected={vertex,normal,q,tri,u,w,(thickness-signedDistance)/dt,true};
                        selected.bodyVelocity=((a-body.previous[ia])*(1-u-w)+(b-body.previous[ib])*u+(c-body.previous[ic])*w)/dt;}}
            }
            if(signedDistance>=-1e-6f)continue;
            position[vertex]=point+selected.normal*(thickness-signedDistance);contact=selected;changed[vertex]=1;any=true;++stats_.contactsResolved;
        }
        if(!any)break;
        applyCapsuleCollision(position,body.capsules,thickness,dt,false,&changed);
    }
}

void PhysicsSolver::applyCapsuleCollision(std::vector<Vec3>& position,const std::vector<MovingCapsule>& capsules,float thickness,float dt,bool swept,const std::vector<uint8_t>* filter){
    if(capsules.empty())return;
    for(const auto& c:capsules)if(!isFinite(c.previousA)||!isFinite(c.previousB)||!isFinite(c.currentA)||!isFinite(c.currentB)||!std::isfinite(c.radius)||c.radius<=0)
        throw std::runtime_error("Invalid moving capsule");
    auto closest=[](Vec3 p,Vec3 a,Vec3 b,float& t){auto axis=b-a;float norm=dot(axis,axis);t=norm>1e-16f?std::clamp(dot(p-a,axis)/norm,0.f,1.f):0.f;return a+axis*t;};
    auto lerp=[](Vec3 a,Vec3 b,float t){return a+(b-a)*t;};
    std::vector<int> contactIndex(vertexCount_,-1);for(size_t i=0;i<bodyContacts_.size();++i)contactIndex[bodyContacts_[i].vertex]=static_cast<int>(i);
    for(uint32_t v=0;v<vertexCount_;++v)if(!pinned_[v]&&(!filter||(*filter)[v])){
        for(int pass=0;pass<2;++pass)for(const auto& capsule:capsules){
            const auto end=position[v],start=previous_[v];const float radius=capsule.radius+thickness;
            float t=0,oldT=0;auto q=closest(end,capsule.currentA,capsule.currentB,t);
            auto oldQ=closest(start,capsule.previousA,capsule.previousB,oldT);
            float distance=length(end-q),oldGap=length(start-oldQ)-radius;
            Vec3 normal=end-q;bool hit=distance<radius,recovery=oldGap<0;
            const auto motion=end-start;
            const float bound=std::max(length(capsule.currentA-capsule.previousA-motion),length(capsule.currentB-capsule.previousB-motion));
            if(swept&&pass==0&&oldGap>0&&bound>1e-9f&&distance<=bound+radius){
                float time=0;for(int k=0;k<32;++k){auto a=lerp(capsule.previousA,capsule.currentA,time),b=lerp(capsule.previousB,capsule.currentB,time),point=lerp(start,end,time);
                    float along=0;auto feature=closest(point,a,b,along);float gap=length(point-feature)-radius;
                    if(gap<=1e-5f){normal=point-feature;t=along;q=lerp(capsule.currentA,capsule.currentB,t);hit=true;break;}
                    time+=gap/bound;if(time>1)break;
                }
            }
            if(!hit)continue;
            if(length(normal)<1e-8f)normal=start-oldQ;
            if(length(normal)<1e-8f){auto axis=capsule.currentB-capsule.currentA;normal=cross(axis,{1,0,0});if(length(normal)<1e-8f)normal={0,0,1};}
            normal=normalize(normal);const float residual=dot(end-q,normal)-radius;if(residual>=0)continue;
            position[v]=end-normal*residual;
            BodyContact contact{};contact.vertex=v;contact.bodyTri=~0u;contact.normal=normal;contact.normalImpulse=-residual/dt;contact.recovery=recovery;
            contact.bodyVelocity=((capsule.currentA-capsule.previousA)*(1-t)+(capsule.currentB-capsule.previousB)*t)/dt;
            if(contactIndex[v]<0){contactIndex[v]=static_cast<int>(bodyContacts_.size());bodyContacts_.push_back(contact);}else bodyContacts_[contactIndex[v]]=contact;
            ++stats_.contactsResolved;
        }
    }
}

void PhysicsSolver::applySelfCollision(std::vector<Vec3>& position,float thickness) {
    // Scalar oracle: immutable candidates, mass-weighted contact records, then gather.
    const auto snapshot=position;
    contactRecords.clear();
    auto neighbor=[&](uint32_t a,uint32_t b){auto first=selfVertices_.begin()+selfOffsets_[a],last=selfVertices_.begin()+selfOffsets_[a+1];
        auto found=std::lower_bound(first,last,b);return found!=last&&*found==b&&selfDistances_[size_t(found-selfVertices_.begin())]<=2*thickness;};
    auto lerp=[](Vec3 a,Vec3 b,float t){return a+(b-a)*t;};
    std::vector<Vec3> deltas(vertexCount_);std::vector<uint32_t> counts(vertexCount_);
    auto add=[&](uint32_t v,Vec3 delta){if(inverseMass_[v]>0){deltas[v]+=delta;++counts[v];}};
    struct Contact {uint32_t ids[4]{};float weights[4]{};Vec3 delta{};float distance{1e30f};bool found{};};
    auto emit=[&](const Contact& contact){if(!contact.found)return;
        if(captureContacts){SelfContactRecord record;std::copy_n(contact.ids,4,record.ids.begin());
            std::copy_n(contact.weights,4,record.weights.begin());record.delta=contact.delta;record.distance=contact.distance;contactRecords.push_back(record);}
        for(int k=0;k<4;++k)if(std::abs(contact.weights[k])>1e-6f)add(contact.ids[k],contact.delta*(contact.weights[k]*inverseMass_[contact.ids[k]]));
        ++stats_.selfContactsResolved;
    };
    for(int kind=0;kind<2;++kind){
        // Reuse a triangle BVH for edges by duplicating their first endpoint.
        std::vector<uint32_t> primitives;
        if(kind==0)primitives=triangles_;else for(size_t i=0;i<stretchPairs_.size();i+=2)
            primitives.insert(primitives.end(),{stretchPairs_[i],stretchPairs_[i],stretchPairs_[i+1]});
        const uint32_t count=static_cast<uint32_t>(primitives.size()/3);if(!count)continue;
        std::vector<uint32_t> order(count);for(uint32_t i=0;i<count;++i)order[i]=i;
        std::vector<BVHNode> nodes;auto root=buildBodyBVHRecursive(nodes,order,snapshot,primitives,0,count);
        for(size_t i=nodes.size();i-->0;){auto& n=nodes[i];if(n.isLeaf){
            AABB bounds{{1e30f,1e30f,1e30f},{-1e30f,-1e30f,-1e30f}};
            for(uint32_t j=n.triBegin;j<n.triEnd;++j)for(uint32_t k=0;k<3;++k){auto v=primitives[3*order[j]+k];
                bounds=mergeAABB(bounds,mergeAABB(pointBounds(snapshot[v],0),pointBounds(previous_[v],0)));}n.bounds=bounds;
        }else n.bounds=mergeAABB(nodes[n.left].bounds,nodes[n.right].bounds);}
        const uint32_t queries=kind?count:vertexCount_;
        for(uint32_t id=0;id<queries;++id){
            const uint32_t a=kind?stretchPairs_[2*id]:id,b=kind?stretchPairs_[2*id+1]:id;
            const Vec3 aa=snapshot[a],bb=snapshot[b],oa=previous_[a],ob=previous_[b];
            const AABB bounds=mergeAABB(mergeAABB(pointBounds(aa,thickness),pointBounds(bb,thickness)),
                mergeAABB(pointBounds(oa,thickness),pointBounds(ob,thickness)));
            Contact selected;uint32_t selectedKey=~0u,selectedPrimitive=~0u;
            traverseBVH(nodes,root,bounds,[&](uint32_t nodeId){const auto& node=nodes[nodeId];
                for(uint32_t j=node.triBegin;j<node.triEnd;++j){const uint32_t primitive=order[j];
                    if(kind==0){
                        const auto ia=primitives[3*primitive],ib=primitives[3*primitive+1],ic=primitives[3*primitive+2];
                        if(neighbor(id,ia)||neighbor(id,ib)||neighbor(id,ic))continue;
                        Vec3 pa=snapshot[ia],pb=snapshot[ib],pc=snapshot[ic],opa=previous_[ia],opb=previous_[ib],opc=previous_[ic];
                        Vec3 q;float u{},v{},w0{};const float distance=std::sqrt(pointTriangleDistSqBary(aa,pa,pb,pc,q,u,v,&w0));
                        Vec3 normal=cross(pb-pa,pc-pa);if(length(normal)<1e-12f)continue;normal=normalize(normal);
                        if(dot(oa-opa,cross(opb-opa,opc-opa))<0)normal=-normal;
                        bool contact=distance<thickness;
                        const auto motion=aa-oa;
                        const float bound=std::max({length(pa-opa-motion),length(pb-opb-motion),length(pc-opc-motion)});float time=0;
                        if(distance>bound+thickness)continue;
                        for(int k=0;k<24&&!contact&&bound>1e-9f;++k){Vec3 nearest;float bu{},bv{},bw0{};
                            const float gap=std::sqrt(pointTriangleDistSqBary(lerp(oa,aa,time),lerp(opa,pa,time),lerp(opb,pb,time),lerp(opc,pc,time),nearest,bu,bv,&bw0));
                            if(gap<thickness+1e-5f){u=bu;v=bv;w0=bw0;q=pa+(pb-pa)*u+(pc-pa)*v;contact=true;break;}
                            time+=(gap-thickness)/bound;if(time>1)break;}
                        // At an edge/vertex, use the closest feature's separation normal.
                        // Adjacent faces must not push the same edge contact in different directions.
                        if(std::min({w0,u,v})<1e-6f){auto separation=oa-(opa+(opb-opa)*u+(opc-opa)*v);
                            if(length(separation)>1e-6f)normal=normalize(separation);}
                        const float residual=dot(aa-q,normal)-thickness;if(!contact||residual>=0)continue;
                        const float denom=inverseMass_[id]+inverseMass_[ia]*w0*w0+inverseMass_[ib]*u*u+inverseMass_[ic]*v*v;
                        const uint32_t key=static_cast<uint32_t>(distance*1e6f+.5f);
                        if(denom<1e-12f||key>selectedKey||(key==selectedKey&&primitive>=selectedPrimitive))continue;
                        selectedKey=key;selectedPrimitive=primitive;
                        const float mobile=std::max({inverseMass_[id],inverseMass_[ia]*w0,inverseMass_[ib]*u,inverseMass_[ic]*v});
                        selected={{id,ia,ib,ic},{1,-w0,-u,-v},normal*(-residual/std::max(denom,mobile*.25f)),distance,true};
                    }else{
                        if(primitive<=id)continue;
                        const auto c=stretchPairs_[2*primitive],d=stretchPairs_[2*primitive+1];
                        if(neighbor(a,c)||neighbor(a,d)||neighbor(b,c)||neighbor(b,d))continue;
                        const Vec3 cc=snapshot[c],dd=snapshot[d],oc=previous_[c],od=previous_[d];
                        Vec3 qp,qq;float st{},tt{};
                        const float distance=std::sqrt(edgeEdgeDistSq(aa,bb,cc,dd,qp,qq,st,tt));
                        bool contact=distance<thickness;float time=0;
                        const auto common=((aa-oa)+(bb-ob)+(cc-oc)+(dd-od))*.25f;
                        const float bound=std::max(length(aa-oa-common),length(bb-ob-common))+std::max(length(cc-oc-common),length(dd-od-common));
                        if(distance>bound+thickness)continue;
                        for(int k=0;k<24&&!contact&&bound>1e-9f;++k){Vec3 qa,qb;float ss{},ts{};
                            const float gap=std::sqrt(edgeEdgeDistSq(lerp(oa,aa,time),lerp(ob,bb,time),lerp(oc,cc,time),lerp(od,dd,time),qa,qb,ss,ts));
                            if(gap<thickness+1e-5f){st=ss;tt=ts;contact=true;break;}
                            time+=(gap-thickness)/bound;if(time>1)break;}
                        if(!contact)continue;
                        Vec3 normal=lerp(oa,ob,st)-lerp(oc,od,tt);if(length(normal)<=1e-6f)normal=cross(bb-aa,dd-cc);
                        if(length(normal)<1e-12f)continue;normal=normalize(normal);
                        const float residual=dot(lerp(aa,bb,st)-lerp(cc,dd,tt),normal)-thickness;if(residual>=0)continue;
                        const float w0=1-st,w1=st,w2=1-tt,w3=tt;
                        const float denom=inverseMass_[a]*w0*w0+inverseMass_[b]*w1*w1+inverseMass_[c]*w2*w2+inverseMass_[d]*w3*w3;
                        const uint32_t key=static_cast<uint32_t>(distance*1e6f+.5f);
                        if(denom<1e-12f||key>selectedKey||(key==selectedKey&&primitive>=selectedPrimitive))continue;
                        selectedKey=key;selectedPrimitive=primitive;
                        const float mobile=std::max({inverseMass_[a]*w0,inverseMass_[b]*w1,inverseMass_[c]*w2,inverseMass_[d]*w3});
                        selected={{a,b,c,d},{w0,w1,-w2,-w3},normal*(-residual/std::max(denom,mobile*.25f)),distance,true};
                    }
                }
            });emit(selected);
        }
    }
    for(uint32_t v=0;v<vertexCount_;++v)if(counts[v])position[v]+=deltas[v]/float(counts[v]);
}

}  // namespace mlcloth::demo
