// Two-sided shading for the predicted cloth.
//
// The back face gets a distinctly different albedo rather than the same lambert
// with a flipped normal. Which side of a garment patch is "front" is arbitrary,
// so the colour is not describing the material -- it is making *inconsistent*
// winding visible. A patch that reads as a single colour is coherently wound; a
// patch that speckles is inside out in places, which is exactly the
// `flipped_fraction` the structure metrics count, seen directly instead of as a
// scalar. Culling stays off so both sides are drawn.

struct PSInput {
    [[vk::location(0)]] float3 normal : NORMAL0;
    [[vk::location(1)]] float height : TEXCOORD0;
    bool frontFace : SV_IsFrontFace;
};

float4 main(PSInput input) : SV_TARGET
{
    float3 normal = normalize(input.normal);
    if (!input.frontFace) normal = -normal;

    const float3 lightDirection = normalize(float3(0.35, 0.85, 0.40));
    const float lambert = saturate(dot(normal, lightDirection));
    // A little rim term so the silhouette of a fold reads even where lambert is flat.
    const float wrap = saturate(0.5 + 0.5 * dot(normal, float3(0.0, 1.0, 0.0)));

    const float3 front = lerp(float3(0.10, 0.34, 0.68), float3(0.42, 0.68, 0.98), saturate(input.height * 0.2 + 0.5));
    const float3 back = float3(0.86, 0.42, 0.22);
    const float3 albedo = input.frontFace ? front : back;
    const float3 colour = albedo * (0.22 + 0.18 * wrap) + albedo * lambert * 0.80;
    return float4(colour, 1.0);
}
