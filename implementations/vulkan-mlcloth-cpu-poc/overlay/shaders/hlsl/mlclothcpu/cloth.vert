// Triangle pass for the predicted cloth and for the character body. Positions come
// from point_transform.comp or body_skin.comp and normals from cloth_normals.comp or
// the same skinning pass, both as per-vertex buffers indexed by the baked triangle
// list.
//
// `instance.offset` is a rigid world translation, used by the side-by-side view for the
// body only. The cloth's branch offset is applied on the host in reference-bone-local
// coordinates before the transform, because that is the space its constraints live in;
// the body is skinned straight into component space and has no such space to work in, so
// one skinned copy is translated per draw instead of being skinned three times.
//
// The world position is passed on rather than the height it used to carry: the shading is a
// light rig now, so the view direction is what the fragment stage needs, and the vertical ramp
// that stood in for shading before is gone.

#include "scene_camera.hlsli"

struct VSInput {
    [[vk::location(0)]] float4 position : POSITION0;
    [[vk::location(1)]] float4 normal : NORMAL0;
};

struct VSOutput {
    float4 position : SV_POSITION;
    [[vk::location(0)]] float3 normal : NORMAL0;
    [[vk::location(1)]] float3 worldM : TEXCOORD0;
};

struct Instance { float4 offset; float4 tint; };
[[vk::push_constant]] Instance instance;

VSOutput main(VSInput input)
{
    VSOutput output;
    const float3 world = input.position.xyz + instance.offset.xyz;
    output.position = mul(camera.projection, mul(camera.view, float4(world, 1.0)));
    output.normal = input.normal.xyz;
    output.worldM = world;
    return output;
}
