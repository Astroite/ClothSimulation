// The point cloud is unlit -- a point has no normal to light -- so its colour is treated as
// emitted radiance and pushed through the same tone curve as everything else. It has to be, now
// that there is a sky behind it: a value written raw to the UNORM swapchain would land on a
// different curve than the backdrop and the cloud would read as too dark against it. The gain is
// what lifts single pixels clear of the horizon, which is the brightest thing they sit against.

#include "scene_lighting.hlsli"

static const float kPointRadiance = 2.2;

struct PSInput {
    [[vk::location(0)]] float3 color : COLOR0;
};

float4 main(PSInput input) : SV_TARGET
{
    return float4(tonemapAndEncode(input.color * kPointRadiance), 1.0);
}
