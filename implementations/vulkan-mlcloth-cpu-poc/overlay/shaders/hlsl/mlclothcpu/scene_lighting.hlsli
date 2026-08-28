// The scene's backdrop and its light rig, in one place because the two have to agree.
//
// Both the sky pass and the surface shading read this file. That is the point: a surface's
// ambient term is supposed to be the light arriving from the same sky that is drawn behind it,
// so if the two drifted apart a garment could read as too dark for a reason that is nowhere in
// the simulation. Everything below is one hemisphere model used twice -- once integrated over a
// normal for shading, once evaluated along a view ray for the background.
//
// Radiance here is linear and unbounded. The swapchain is UNORM (VulkanSwapChain.cpp prefers
// B8G8R8A8_UNORM over any sRGB format), so nothing downstream encodes for us and every stage
// has to finish with `tonemapAndEncode`. Writing linear values straight to an 8-bit UNORM
// surface is what makes a lit scene look murky no matter how much light is added to it.
//
// The rig is three lights and no shadows, deliberately: a key with a visible source in the sky,
// a cool fill from behind on the other side so an unlit silhouette does not go to black, and a
// hemisphere ambient. What it has to do is make folds and limbs read; what it must not do is
// invent contrast that could be mistaken for geometry, which is why there is no shadow term and
// the specular is one weak broad lobe.

#ifndef MLCLOTH_SCENE_LIGHTING_HLSLI
#define MLCLOTH_SCENE_LIGHTING_HLSLI

// --- the sky ---------------------------------------------------------------------------------
// Saturated rather than bright. The tone curve compresses hard at this exposure, so a lit body
// and the sky behind it land within a few percent of each other in *value* whatever is done to
// either -- the separation that survives is hue, which is why the backdrop is distinctly cool
// and the key light distinctly warm. A silhouette against the horizon reads on colour.
static const float3 kSkyZenith  = float3(0.10, 0.21, 0.50);
static const float3 kSkyHorizon = float3(0.40, 0.50, 0.66);

// --- the lights ------------------------------------------------------------------------------
static const float3 kSunDirection  = normalize(float3(0.38, 0.72, 0.55));
static const float3 kSunColour     = float3(1.00, 0.97, 0.90) * 1.55;
static const float3 kFillDirection = normalize(float3(-0.60, 0.25, -0.70));
static const float3 kFillColour    = float3(0.34, 0.40, 0.50) * 0.45;
// The hemisphere ambient, as the two halves it interpolates between. These are the sky and the
// floor as seen by a surface, so they track the two above rather than being tuned separately --
// which is what puts every shadow on the cool side of every lit face.
static const float3 kSkyAmbient    = float3(0.24, 0.30, 0.42);
static const float3 kGroundBounce  = float3(0.100, 0.095, 0.090);

// --- the floor -------------------------------------------------------------------------------
// The character's own floor: the body's rest pose puts its feet on component z = 0 (measured at
// -0.035 cm by body_validate) and world y is component z in metres, so y = 0 is the ground the
// animation was authored against. The grid is there to give the eye a scale and a horizon; the
// character standing on it is what makes a garment's height read at all.
static const float kGroundHeightM = 0.0;
static const float3 kGroundAlbedo = float3(0.115, 0.115, 0.120);
static const float3 kGridAlbedo   = float3(0.20, 0.21, 0.24);
static const float kGridMinorM    = 0.25;
static const float kGridMajorM    = 1.00;

float3 skyRadiance(float3 direction)
{
    // Brighter at the horizon than at the zenith, which is both what a hazy daylight sky does
    // and what puts the most contrast behind a standing figure.
    const float up = saturate(direction.y);
    float3 radiance = lerp(kSkyHorizon, kSkyZenith, pow(up, 0.55));
    // A disc and a halo, so the key light has a source you can point at. Without it the
    // direction of the shading is guesswork.
    const float cosine = saturate(dot(direction, kSunDirection));
    radiance += kSunColour * (pow(cosine, 1400.0) * 5.0 + pow(cosine, 22.0) * 0.14);
    return radiance;
}

float3 hemisphereIrradiance(float3 normal)
{
    return lerp(kGroundBounce, kSkyAmbient, saturate(0.5 + 0.5 * normal.y));
}

// One analytically antialiased grid line set. Dividing by the screen-space derivative is what
// keeps the far field from turning into noise: as the lines converge the width blows up, the
// term saturates, and they fade out on their own instead of aliasing.
float gridLines(float2 coordinate)
{
    const float2 width = max(fwidth(coordinate), 1.0e-5);
    const float2 distanceToLine = abs(frac(coordinate - 0.5) - 0.5) / width;
    return 1.0 - saturate(min(distanceToLine.x, distanceToLine.y));
}

// Radiance of the floor at a hit point, hazed into the horizon with distance so the plane has
// no visible edge. `facingUp` is false when the camera has orbited below the floor, where the
// underside gets no key light -- otherwise the plane would brighten as you pass through it.
float3 groundRadiance(float3 hitM, float distanceM, bool facingUp)
{
    const float fade = exp(-distanceM * 0.09);
    const float lines = gridLines(hitM.xz / kGridMinorM) * 0.30
                      + gridLines(hitM.xz / kGridMajorM) * 1.00;
    const float3 albedo = kGroundAlbedo + kGridAlbedo * saturate(lines) * fade;
    float3 irradiance = facingUp ? kSkyAmbient : kGroundBounce;
    if (facingUp) {
        irradiance += kSunColour * saturate(kSunDirection.y) + kFillColour * saturate(kFillDirection.y);
    }
    return lerp(kSkyHorizon, albedo * irradiance, exp(-distanceM * 0.035));
}

// What every stage ends with. Reinhard rather than anything filmic: it is monotonic per channel,
// so a brighter surface never reads as darker after tone mapping, which matters when the picture
// is being used to compare three branches by eye.
float3 tonemapAndEncode(float3 radiance)
{
    const float3 mapped = radiance / (1.0 + max(radiance, 0.0));
    return pow(saturate(mapped), 1.0 / 2.2);
}

#endif
