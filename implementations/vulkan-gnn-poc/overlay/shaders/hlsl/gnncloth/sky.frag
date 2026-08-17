struct PSInput {
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    const float2 skyUv = float2(input.uv.x, 1.0 - input.uv.y);
    const float vertical = saturate(skyUv.y);
    const float horizonBlend = smoothstep(0.0, 0.78, vertical);
    const float3 horizon = float3(0.78, 0.89, 1.00);
    const float3 zenith = float3(0.16, 0.43, 0.78);
    float3 color = lerp(horizon, zenith, horizonBlend);

    // Warm, soft sun and horizon haze keep the procedural sky readable without
    // introducing texture assets into the self-contained PoC.
    const float2 sunDelta = skyUv - float2(0.58, 0.78);
    const float sunDistance = length(sunDelta * float2(1.0, 1.35));
    const float sunDisc = 1.0 - smoothstep(0.015, 0.055, sunDistance);
    const float sunGlow = exp(-sunDistance * 10.0);
    color += float3(1.00, 0.72, 0.34) * (sunDisc * 0.85 + sunGlow * 0.32);
    color += float3(0.20, 0.18, 0.14) * exp(-vertical * 8.0);
    return float4(saturate(color), 1.0);
}
