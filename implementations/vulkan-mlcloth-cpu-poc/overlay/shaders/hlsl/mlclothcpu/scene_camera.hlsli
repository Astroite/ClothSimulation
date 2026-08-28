// The one declaration of the camera block, included by every stage that reads it.
//
// It is shared rather than repeated because the block grew: the sky needs to turn a pixel back
// into a world-space ray, which takes the inverse of the view-projection and the eye position,
// and a stage holding a stale copy of this struct would read the wrong 64 bytes rather than
// fail to compile. `CameraUniform` in mlclothcpu.cpp is the fourth copy and the one that
// writes it -- keep it in step.
//
// `inverseViewProjection` is the inverse of the same two matrices above it, computed on the
// host once per frame. Deriving it here is not an option: HLSL has no matrix inverse, and
// approximating it from the parts would silently disagree with the transform the geometry
// actually used.

#ifndef MLCLOTH_SCENE_CAMERA_HLSLI
#define MLCLOTH_SCENE_CAMERA_HLSLI

struct CameraParams {
    float4x4 projection;
    float4x4 view;
    float4x4 inverseViewProjection;
    float4 cameraPositionM;     // world metres; .w unused
};
cbuffer cameraParams : register(b0) { CameraParams camera; };

#endif
