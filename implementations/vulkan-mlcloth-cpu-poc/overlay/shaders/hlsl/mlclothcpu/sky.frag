// The backdrop: a gradient sky above a gridded floor, both evaluated along the view ray.
//
// Why a floor at all. The scene is a garment on a body in mid-air with a black background, and
// with nothing else in frame there is no cue for how high a hem sits, how far a limb has swung,
// or -- in the side-by-side view -- whether the three branches are at the same height. The grid
// is on the plane the animation was authored against, so it answers all three by eye.
//
// The pass writes no depth and tests none, and is drawn before everything else. That is
// deliberate rather than incidental: a hem or a foot that has gone *below* the floor stays
// visible instead of being swallowed by it. The floor is a reading aid, and a reading aid that
// can hide the thing being judged is worse than none.

#include "scene_camera.hlsli"
#include "scene_lighting.hlsli"

struct PSInput {
    [[vk::location(0)]] float2 ndc : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    // The pixel, back through the same transform the geometry went forward through. Taking the
    // inverse of the host's own view-projection is what keeps the horizon fixed to the geometry
    // under every camera the orbit control can produce.
    const float4 farPoint = mul(camera.inverseViewProjection, float4(input.ndc, 1.0, 1.0));
    const float3 origin = camera.cameraPositionM.xyz;
    const float3 direction = normalize(farPoint.xyz / farPoint.w - origin);

    const float toPlane = (kGroundHeightM - origin.y) / direction.y;
    // The bounds are not decoration: a ray parallel to the floor sends `toPlane` to infinity and
    // the grid's `frac` to NaN, which a distance fade of zero would not clean up.
    const bool hitsGround = abs(direction.y) > 1.0e-4 && toPlane > 0.0 && toPlane < 1.0e4;

    float3 radiance;
    if (hitsGround) {
        radiance = groundRadiance(origin + direction * toPlane, toPlane, origin.y > kGroundHeightM);
    } else {
        radiance = skyRadiance(direction);
    }
    return float4(tonemapAndEncode(radiance), 1.0);
}
