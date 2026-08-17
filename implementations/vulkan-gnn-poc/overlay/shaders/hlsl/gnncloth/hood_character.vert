struct VSInput {
    [[vk::location(0)]] float3 position : POSITION0;
    [[vk::location(1)]] float3 normal : NORMAL0;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
};
struct VSOutput {
    float4 position : SV_POSITION;
    [[vk::location(0)]] float3 normal : TEXCOORD0;
    [[vk::location(1)]] float3 worldPosition : TEXCOORD1;
};
struct UBO { float4x4 projection; float4x4 modelview; float4 lightPos; float4 rootPosition; };
cbuffer ubo : register(b0) { UBO params; };
VSOutput main(VSInput input) {
    VSOutput output;
    output.normal = input.normal;
    output.worldPosition = input.position;
    output.position = mul(params.projection, mul(params.modelview, float4(input.position, 1.0)));
    return output;
}
