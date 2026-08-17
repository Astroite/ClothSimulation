struct PSInput {
    [[vk::location(0)]] float2 uv : TEXCOORD0;
    [[vk::location(1)]] float3 worldPosition : TEXCOORD1;
    bool frontFace : SV_IsFrontFace;
};

float4 main(PSInput input) : SV_TARGET
{
    float3 normal = normalize(cross(ddx(input.worldPosition), ddy(input.worldPosition)));
    if (!input.frontFace) normal = -normal;
    const float checker = fmod(floor(input.uv.x * 16.0) + floor(input.uv.y * 16.0), 2.0);
    const float3 baseColor = lerp(float3(0.10, 0.38, 0.82), float3(0.25, 0.76, 1.00), checker);
    const float3 lightDirection = normalize(float3(-0.35, 0.82, -0.46));
    const float diffuse = saturate(dot(normal, lightDirection));
    const float fill = saturate(dot(normal, -lightDirection));
    const float hemisphere = 0.5 + 0.5 * normal.y;
    const float lighting = 0.48 + 0.72 * diffuse + 0.16 * fill + 0.12 * hemisphere;
    const float edgeHighlight = pow(1.0 - abs(normal.z), 3.0) * 0.10;
    return float4(saturate(baseColor * lighting + edgeHighlight), 1.0);
}
