// Shared declarations for the gnncloth compute shaders.
//
// Particle was previously copied into five shaders and UBO into four, one of them
// a truncated version stopping at maxAcceleration. The truncation happened to be
// harmless, and the trailing float3 happened to land at offset 100 where it does
// not straddle a 16-byte row, so the layout matched the 112-byte host struct by
// luck rather than by construction. Only the host had a static_assert. One
// definition removes both hazards.
#ifndef GNNCLOTH_COMMON_HLSLI
#define GNNCLOTH_COMMON_HLSLI

#include "vgnn_weights.hlsli"

struct Particle {
    float4 pos;
    float4 vel;
    float4 uv;
    // xyz = position before this step's integration, used by XPBD to reconstruct
    // velocity from the corrected position. Not a surface normal: the fragment
    // shader derives shading normals from screen-space derivatives instead.
    float4 previousPosition;
    float4 rest; // xyz = rest position, w = pinned flag
};

// Mirrors Compute::UniformData in gnncloth.cpp, which static_asserts its size.
struct UBO {
    float deltaT;
    float particleMass;
    float springStiffness;
    float damping;
    float restDistH;
    float restDistV;
    float restDistD;
    float sphereRadius;
    float4 spherePos;
    float4 gravity;
    int2 particleCount;
    uint vertexCount;
    uint edgeCount;
    float maxSpeed;
    float maxAcceleration;
    float stretchComplianceMicro;
    float bendComplianceMicro;
    float xpbdVelocityDamping;
    float3 sphereVelocity;
};

[[vk::binding(0)]] StructuredBuffer<Particle> particleIn;
[[vk::binding(1)]] RWStructuredBuffer<Particle> particleOut;
cbuffer ubo : register(b2) { UBO params; };
[[vk::binding(3)]] StructuredBuffer<uint> vertexOffsets;
[[vk::binding(4)]] StructuredBuffer<uint> neighborIndices;
[[vk::binding(5)]] StructuredBuffer<float> weights;
[[vk::binding(6)]] RWStructuredBuffer<float4> hiddenState;
[[vk::binding(7)]] RWStructuredBuffer<float4> accelerationOut;
[[vk::binding(8)]] StructuredBuffer<uint4> constraintEdges;
[[vk::binding(9)]] RWStructuredBuffer<float> constraintLambdas;
[[vk::binding(10)]] RWStructuredBuffer<float4> verificationPositions;

float3 clampMagnitude(float3 value, float maximum)
{
    const float magnitude = length(value);
    return (magnitude > maximum && magnitude > 0.0) ? value * (maximum / magnitude) : value;
}

#endif // GNNCLOTH_COMMON_HLSLI
