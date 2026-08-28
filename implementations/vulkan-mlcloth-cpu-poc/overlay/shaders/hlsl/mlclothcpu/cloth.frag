// Two-sided shading for the predicted cloth and for the character body.
//
// The light rig is in scene_lighting.hlsli, shared with the sky pass so a surface's ambient is
// the light arriving from the sky that is actually drawn behind it. That sharing is the reason
// the earlier hand-tuned constants are gone: they were a lambert term plus a vertical ramp that
// stood in for ambient, which reads as flat under a bright backdrop and made the scene hard to
// see at all once a body was in frame.
//
// Two things here are diagnostics rather than looks, and survive the rewrite.
//
// The back face gets a distinctly darker albedo rather than the same lambert with a flipped
// normal. Which side of a garment patch is "front" is arbitrary, so the colour is not describing
// the material -- it is making *inconsistent* winding visible. A patch that reads as a single
// shade is coherently wound; a patch that speckles is inside out in places, which is exactly the
// `flipped_fraction` the structure metrics count, seen directly instead of as a scalar. Culling
// stays off so both sides are drawn.
//
// `tint.rgb` replaces the base albedo rather than multiplying it: the comparison branches have to
// be told apart by hue (blue A / orange B / green C), and no multiplier turns the original blue
// orange. The back face is that same hue darkened, so the winding signal survives without
// borrowing another branch's colour. The single-branch path is pushed A's blue, so its picture
// changes only by the lighting. This is also why the specular is one weak white lobe: a strong
// highlight washes the hue out, and the hue is carrying information.

#include "scene_camera.hlsli"
#include "scene_lighting.hlsli"

static const float kBackFaceAlbedo = 0.42;

struct PSInput {
    [[vk::location(0)]] float3 normal : NORMAL0;
    [[vk::location(1)]] float3 worldM : TEXCOORD0;
    bool frontFace : SV_IsFrontFace;
};

struct Instance { float4 offset; float4 tint; };
[[vk::push_constant]] Instance instance;

float4 main(PSInput input) : SV_TARGET
{
    float3 normal = normalize(input.normal);
    if (!input.frontFace) normal = -normal;

    const float3 albedo = instance.tint.rgb * (input.frontFace ? 1.0 : kBackFaceAlbedo);
    const float3 view = normalize(camera.cameraPositionM.xyz - input.worldM);

    float3 irradiance = hemisphereIrradiance(normal);
    irradiance += kSunColour * saturate(dot(normal, kSunDirection));
    irradiance += kFillColour * saturate(dot(normal, kFillDirection));

    // Key light only, broad and weak. It is here so the crest of a fold has a highlight to
    // separate it from the trough, and it is suppressed on back faces because the inside of a
    // garment glinting would fight the winding signal.
    const float3 halfway = normalize(kSunDirection + view);
    const float lobe = pow(saturate(dot(normal, halfway)), 22.0) * (input.frontFace ? 0.11 : 0.0);

    return float4(tonemapAndEncode(albedo * irradiance + kSunColour * lobe), 1.0);
}
