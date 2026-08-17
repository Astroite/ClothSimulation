struct VSOutput { float4 position : SV_POSITION; [[vk::location(0)]] float3 normal : NORMAL0; };
struct UBO { float4x4 projection; float4x4 modelview; float4 lightPos; float4 spherePosRadius; };
cbuffer ubo : register(b0) { UBO params; };

float3 spherePoint(uint x, uint y)
{
    const float u = float(x) / 32.0;
    const float v = float(y) / 16.0;
    const float phi = u * 6.28318530718;
    const float theta = v * 3.14159265359;
    return float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

VSOutput main(uint vertexID : SV_VertexID)
{
    const uint triangleVertex = vertexID % 6;
    const uint cell = vertexID / 6;
    const uint x = cell % 32;
    const uint y = cell / 32;
    const uint2 corner = triangleVertex == 0 ? uint2(0, 0) :
                         triangleVertex == 1 ? uint2(1, 0) :
                         triangleVertex == 2 ? uint2(0, 1) :
                         triangleVertex == 3 ? uint2(0, 1) :
                         triangleVertex == 4 ? uint2(1, 0) : uint2(1, 1);
    VSOutput output;
    output.normal = spherePoint(x + corner.x, y + corner.y);
    const float3 world = params.spherePosRadius.xyz + output.normal * params.spherePosRadius.w;
    output.position = mul(params.projection, mul(params.modelview, float4(world, 1.0)));
    return output;
}
