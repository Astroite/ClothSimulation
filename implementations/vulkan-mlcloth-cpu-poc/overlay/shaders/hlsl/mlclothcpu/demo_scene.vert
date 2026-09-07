struct Scene { float4x4 viewProjection; float4x4 lightProjection; float4 eye; float4 lightDirection; };
[[vk::binding(0,0)]] ConstantBuffer<Scene> scene;
struct Vertex { float4 position; float4 normal; float4 uv; };
[[vk::binding(1,0)]] StructuredBuffer<Vertex> vertices;
[[vk::binding(2,0)]] StructuredBuffer<uint> boneIds;
[[vk::binding(3,0)]] StructuredBuffer<float> weights;
[[vk::binding(4,0)]] StructuredBuffer<float4> matrices;
[[vk::binding(5,0)]] StructuredBuffer<float4> cloth;
[[vk::binding(6,0)]] StructuredBuffer<float4> clothNormals;
struct Draw { float4 offset; float4 color; uint kind; uint actor; uint influences; uint count; uint shadow; uint material; uint boneCount; uint gray; };
[[vk::push_constant]] ConstantBuffer<Draw> draw;
struct Output { float4 position:SV_Position; [[vk::location(0)]] float3 world:TEXCOORD0; [[vk::location(1)]] float3 normal:NORMAL0; [[vk::location(2)]] float2 uv:TEXCOORD1; [[vk::location(3)]] float4 shadow:TEXCOORD2; };
Output main(uint id:SV_VertexID) {
    Output o;float3 p=0,n=0;float2 uv=0;
    if(draw.kind==0) {
        Vertex v=vertices[id];uv=float2(v.uv.x,1-v.uv.y);
        for(uint k=0;k<draw.influences;++k) {
            uint slot=id*draw.influences+k;float w=weights[slot];if(w<=0)continue;
            uint b=(draw.actor*draw.boneCount+boneIds[slot])*3;
            p+=float3(dot(matrices[b],v.position),dot(matrices[b+1],v.position),dot(matrices[b+2],v.position))*w;
            n+=float3(dot(matrices[b].xyz,v.normal.xyz),dot(matrices[b+1].xyz,v.normal.xyz),dot(matrices[b+2].xyz,v.normal.xyz))*w;
        }
    }else if(draw.kind==1){uint slot=draw.actor*draw.count+id;p=lerp(cloth[slot+3*draw.count].xyz,cloth[slot].xyz,draw.offset.w);n=clothNormals[slot].xyz;uv=float2(p.x,p.y);}
    else {const float2 corners[4]={float2(-1,-1),float2(1,-1),float2(1,1),float2(-1,1)};p=float3(corners[id].x*80,0,corners[id].y*80);n=float3(0,1,0);uv=p.xz;}
    p+=draw.offset.xyz;o.world=p;o.normal=normalize(n);o.uv=uv;o.shadow=mul(scene.lightProjection,float4(p,1));
    o.position=draw.shadow?o.shadow:mul(scene.viewProjection,float4(p,1));return o;
}
