struct VSInput { [[vk::location(0)]] float3 position : POSITION0; };
struct VSOutput { float4 position : SV_POSITION; [[vk::location(0)]] float3 worldPosition : TEXCOORD0; };
struct UBO { float4x4 projection; float4x4 modelview; float4 lightPos; float4 rootPosition; };
cbuffer ubo : register(b0) { UBO params; };
VSOutput main(VSInput input) { VSOutput output; output.worldPosition = input.position; output.position = mul(params.projection, mul(params.modelview, float4(input.position, 1.0))); return output; }
