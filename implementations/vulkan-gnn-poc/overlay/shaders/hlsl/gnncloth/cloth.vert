struct VSInput {
    [[vk::location(0)]] float3 position : POSITION0;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
};

struct VSOutput {
    float4 position : SV_POSITION;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
    [[vk::location(1)]] float3 worldPosition : TEXCOORD1;
};

struct UBO {
    float4x4 projection;
    float4x4 modelview;
    float4 lightPos;
    float4 spherePosRadius;
};
cbuffer ubo : register(b0) { UBO params; };

VSOutput main(VSInput input)
{
    VSOutput output;
    output.uv = input.uv;
    output.worldPosition = input.position;
    output.position = mul(params.projection, mul(params.modelview, float4(input.position, 1.0)));
    return output;
}
