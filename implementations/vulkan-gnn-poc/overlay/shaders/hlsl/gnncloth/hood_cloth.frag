struct PSInput { [[vk::location(0)]] float3 worldPosition : TEXCOORD0; bool frontFace : SV_IsFrontFace; };
float4 main(PSInput input) : SV_TARGET {
    float3 normal = normalize(cross(ddx(input.worldPosition), ddy(input.worldPosition)));
    if (!input.frontFace) normal = -normal;
    const float3 light = normalize(float3(-0.32, 0.82, -0.46));
    const float lighting = 0.48 + 0.78 * saturate(dot(normal, light)) + 0.15 * saturate(dot(normal, -light));
    return float4(saturate(float3(0.12, 0.42, 0.88) * lighting + 0.05), 1.0);
}
