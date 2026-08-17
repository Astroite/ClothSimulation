struct PSInput { [[vk::location(0)]] float3 normal : NORMAL0; };
float4 main(PSInput input) : SV_TARGET
{
    const float3 normal = normalize(input.normal);
    const float3 lightDirection = normalize(float3(-0.35, 0.82, -0.46));
    const float diffuse = saturate(dot(normal, lightDirection));
    const float fill = saturate(dot(normal, -lightDirection));
    const float rim = pow(1.0 - abs(normal.z), 3.0);
    const float3 baseColor = float3(0.92, 0.36, 0.08);
    const float3 color = baseColor * (0.52 + 0.70 * diffuse + 0.18 * fill) + float3(1.0, 0.62, 0.24) * rim * 0.16;
    return float4(saturate(color), 1.0);
}
