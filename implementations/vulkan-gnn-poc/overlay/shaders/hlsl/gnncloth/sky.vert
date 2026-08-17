struct VSOutput {
    float4 position : SV_POSITION;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

VSOutput main(uint vertex : SV_VertexID)
{
    const float2 clip = vertex == 0 ? float2(-1.0, -1.0)
        : (vertex == 1 ? float2(3.0, -1.0) : float2(-1.0, 3.0));
    VSOutput output;
    output.position = float4(clip, 1.0, 1.0);
    output.uv = clip * 0.5 + 0.5;
    return output;
}
