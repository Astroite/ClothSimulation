#ifndef TINYHOOD_MLP_HLSLI
#define TINYHOOD_MLP_HLSLI

static const uint TINY_LATENT = 64;

struct MlpDesc {
    uint w0, b0, w1, b1;
    uint w2, b2, lnWeight, lnBias;
    uint inputDimension, outputDimension, hasLayerNorm, reserved;
};

MlpDesc loadMlp(StructuredBuffer<uint4> table, uint id)
{
    const uint4 a = table[id * 3 + 0];
    const uint4 b = table[id * 3 + 1];
    const uint4 c = table[id * 3 + 2];
    MlpDesc value;
    value.w0 = a.x; value.b0 = a.y; value.w1 = a.z; value.b1 = a.w;
    value.w2 = b.x; value.b2 = b.y; value.lnWeight = b.z; value.lnBias = b.w;
    value.inputDimension = c.x; value.outputDimension = c.y; value.hasLayerNorm = c.z; value.reserved = c.w;
    return value;
}

groupshared float tinyScratchA[TINY_LATENT];
groupshared float tinyScratchB[TINY_LATENT];
groupshared float tinyMean;
groupshared float tinyInvStd;

// Weights are stored transposed as [input][output] by buildGpuModelFor, so that adjacent
// lanes -- which own adjacent output channels -- read adjacent floats. `outputCount` is the
// matrix's output width and therefore its row stride; it is not always TINY_LATENT (the
// decoder's third layer emits 3).
float tinyHiddenLinear(StructuredBuffer<float> weights, uint weightOffset, uint biasOffset, uint output, uint inputCount, uint outputCount, bool useA)
{
    float value = weights[biasOffset + output];
    [loop] for (uint input = 0; input < inputCount; ++input)
        value += weights[weightOffset + input * outputCount + output] * (useA ? tinyScratchA[input] : tinyScratchB[input]);
    return value;
}

void tinyLayerNorm(uint lane, uint count)
{
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
        float mean = 0.0;
        [unroll] for (uint i = 0; i < TINY_LATENT; ++i) if (i < count) mean += tinyScratchA[i];
        mean /= float(count);
        float variance = 0.0;
        [unroll] for (uint j = 0; j < TINY_LATENT; ++j) if (j < count) {
            const float difference = tinyScratchA[j] - mean;
            variance += difference * difference;
        }
        tinyMean = mean;
        tinyInvStd = rsqrt(variance / float(count) + 1.0e-5);
    }
    GroupMemoryBarrierWithGroupSync();
}

#endif
