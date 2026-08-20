struct VSInput { [[vk::location(0)]] float3 position : POSITION0; };
struct VSOutput { float4 position : SV_POSITION; [[vk::location(0)]] float3 worldPosition : TEXCOORD0; };
struct UBO { float4x4 projection; float4x4 modelview; float4 lightPos; float4 rootPosition; };
cbuffer ubo : register(b0) { UBO params; };
// One draw per comparison branch, shifted along x so the three solvers stand side by side. The
// shift is applied after the world position the fragment shader differentiates for its normal, so
// a branch's shading is identical wherever it stands.
struct Instance { float4 offset; float4 tint; };
[[vk::push_constant]] Instance instance;
VSOutput main(VSInput input) { VSOutput output; output.worldPosition = input.position; output.position = mul(params.projection, mul(params.modelview, float4(input.position + instance.offset.xyz, 1.0))); return output; }
