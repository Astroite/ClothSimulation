struct VSInput {
    [[vk::location(0)]] float4 position : POSITION0;
};

struct VSOutput {
    float4 position : SV_POSITION;
    [[vk::builtin("PointSize")]] float pointSize : PSIZE;
    [[vk::location(0)]] float3 color : COLOR0;
};

struct CameraParams {
    float4x4 projection;
    float4x4 view;
};
cbuffer cameraParams : register(b0) { CameraParams camera; };

VSOutput main(VSInput input)
{
    VSOutput output;
    output.position = mul(camera.projection, mul(camera.view, float4(input.position.xyz, 1.0)));
    output.pointSize = 1.0;
    const float heightTint = saturate(input.position.y * 0.2 + 0.5);
    output.color = lerp(float3(0.14, 0.58, 1.0), float3(0.84, 0.94, 1.0), heightTint);
    return output;
}
