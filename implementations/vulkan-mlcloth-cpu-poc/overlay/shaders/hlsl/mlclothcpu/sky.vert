// The backdrop, as one triangle covering the viewport.
//
// No vertex buffer and no transform: the three corners are generated from the vertex index and
// written straight to clip space, so the pass costs one draw and no allocation. The clip w is 1
// at all three corners, which makes perspective-correct interpolation of the NDC coordinate
// identical to linear -- the fragment stage can therefore treat the interpolant as the exact
// NDC of its own pixel and unproject it.

struct VSOutput {
    float4 position : SV_POSITION;
    [[vk::location(0)]] float2 ndc : TEXCOORD0;
};

VSOutput main(uint vertexId : SV_VertexID)
{
    // (-1,-1), (-1,3), (3,-1): one oversized triangle, cheaper than a quad and with no seam
    // down the diagonal.
    const float2 ndc = float2(vertexId == 2 ? 3.0 : -1.0, vertexId == 1 ? 3.0 : -1.0);
    VSOutput output;
    // z = 1 is the far plane (the base defines GLM_FORCE_DEPTH_ZERO_TO_ONE). The pass neither
    // tests nor writes depth, so this is only a statement of where the backdrop sits.
    output.position = float4(ndc, 1.0, 1.0);
    output.ndc = ndc;
    return output;
}
