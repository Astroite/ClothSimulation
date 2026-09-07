struct Scene { float4x4 viewProjection; float4x4 lightProjection; float4 eye; float4 lightDirection; };
[[vk::binding(0,0)]] ConstantBuffer<Scene> scene;
[[vk::binding(8,0)]] Texture2D bodyTexture;
[[vk::binding(9,0)]] Texture2D headTexture;
[[vk::binding(10,0)]] SamplerState colorSampler;
[[vk::binding(11,0)]] Texture2D<float> shadowTexture;
[[vk::binding(12,0)]] SamplerComparisonState shadowSampler;
struct Draw { float4 offset; float4 color; uint kind; uint actor; uint influences; uint count; uint shadow; uint material; uint boneCount; uint gray; };
[[vk::push_constant]] ConstantBuffer<Draw> draw;
struct Input { float4 position:SV_Position; [[vk::location(0)]] float3 world:TEXCOORD0; [[vk::location(1)]] float3 normal:NORMAL0; [[vk::location(2)]] float2 uv:TEXCOORD1; [[vk::location(3)]] float4 shadow:TEXCOORD2; };
float4 main(Input i,bool front:SV_IsFrontFace):SV_Target {
    float3 n=normalize(i.normal)*(front?1:-1);float3 base=draw.color.rgb;
    if(draw.kind==0&&draw.gray==0) base*=draw.material?headTexture.Sample(colorSampler,i.uv).rgb:bodyTexture.Sample(colorSampler,i.uv).rgb;
    if(draw.kind==2){float2 g=abs(frac(i.uv*.5-.5)-.5)/max(fwidth(i.uv*.5),.0001);float gridLine=1-saturate(min(g.x,g.y));base=lerp(float3(.19,.22,.25),float3(.25,.28,.31),gridLine*.35);}
    float3 l=normalize(-scene.lightDirection.xyz),v=normalize(scene.eye.xyz-i.world),h=normalize(l+v);
    float ndl=saturate(dot(n,l));float3 s=i.shadow.xyz/i.shadow.w;float2 suv=s.xy*.5+.5;
    float visibility=1;
    if(all(suv>0)&&all(suv<1)&&s.z>0&&s.z<1){visibility=0;for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x)visibility+=shadowTexture.SampleCmpLevelZero(shadowSampler,suv+float2(x,y)/2048,s.z-.0007);visibility/=9;}
    float rough=draw.kind==1?.75:.55;float spec=pow(saturate(dot(n,h)),lerp(90,8,rough))*.08;
    float3 color=base*(.3+.12*saturate(n.y)+.85*ndl*visibility)+spec*visibility;
    // Neutral filmic response followed by gamma; the swapchain uses UNORM.
    color=color/(color+float3(.6,.6,.6));return float4(pow(saturate(color),1/2.2),1);
}
