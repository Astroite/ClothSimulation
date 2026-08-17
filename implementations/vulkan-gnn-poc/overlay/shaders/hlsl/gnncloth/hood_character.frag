struct PSInput { [[vk::location(0)]] float3 normal : TEXCOORD0; [[vk::location(1)]] float3 worldPosition : TEXCOORD1; };
float4 main(PSInput input) : SV_TARGET {
    const float3 normal = normalize(input.normal);
    const float3 light = normalize(float3(-0.32, 0.82, -0.46));
    const float lighting = 0.55 + 0.72 * saturate(dot(normal, light)) + 0.14 * saturate(dot(normal, -light));
    return float4(saturate(float3(0.90, 0.70, 0.58) * lighting), 1.0);
}
