struct PSInput { [[vk::location(0)]] float3 worldPosition : TEXCOORD0; bool frontFace : SV_IsFrontFace; };
// `tint.rgb` replaces the base albedo rather than multiplying it: the comparison branches have to be
// told apart by hue (blue A / orange B / green C), and no multiplier turns the original blue orange.
// The single-branch path passes that original blue, so its output is unchanged.
struct Instance { float4 offset; float4 tint; };
[[vk::push_constant]] Instance instance;
float4 main(PSInput input) : SV_TARGET {
    float3 normal = normalize(cross(ddx(input.worldPosition), ddy(input.worldPosition)));
    if (!input.frontFace) normal = -normal;
    const float3 light = normalize(float3(-0.32, 0.82, -0.46));
    const float lighting = 0.48 + 0.78 * saturate(dot(normal, light)) + 0.15 * saturate(dot(normal, -light));
    return float4(saturate(instance.tint.rgb * lighting + 0.05), 1.0);
}
