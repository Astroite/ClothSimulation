// Triangle pass for the predicted cloth. Positions come from point_transform.comp
// and normals from cloth_normals.comp, both as per-vertex buffers indexed by the
// baked triangle list.

struct VSInput {
    [[vk::location(0)]] float4 position : POSITION0;
    [[vk::location(1)]] float4 normal : NORMAL0;
};

struct VSOutput {
    float4 position : SV_POSITION;
    [[vk::location(0)]] float3 normal : NORMAL0;
    [[vk::location(1)]] float height : TEXCOORD0;
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
    output.normal = input.normal.xyz;
    output.height = input.position.y;
    return output;
}
