// Train.cs.slang - Training kernel for the image-learn example.
// One dispatch call processes a batch of randomly sampled pixels.

import TSNN.CoopVec;
import TSNN.Losses;
import TSNN.Encodings.FeatureBuffer;
import TSNN.LinearOps;
import Network;

Texture2D<float4> gTarget;
ByteAddressBuffer gParams;
RWByteAddressBuffer gParamGrads;
ByteAddressBuffer gEncodingParams;
RWByteAddressBuffer gEncodingParamGrads;

cbuffer CB
{
    uint2 gFrameDim;
    uint gBatchSize;
    uint gCurrentStep;
    uint gFrameIndex;
}

// PCG hash for per-invocation random pixel selection
uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

[numthreads(64, 1, 1)]
void trainMain(uint3 tid: SV_DispatchThreadID)
{
    uint sampleIndex = tid.x;
    if (sampleIndex >= gBatchSize)
        return;

    // Deterministic-but-varied random pixel per sample
    uint seed = pcg(sampleIndex ^ (gFrameIndex * 1664525u) ^ (gCurrentStep * 1013904223u));
    uint2 pixel = uint2(seed % gFrameDim.x, (seed >> 16u) % gFrameDim.y);
    float2 uv = (float2(pixel) + 0.5f) / float2(gFrameDim);

    let target = gTarget.Load(int3(pixel, 0)).xyz;
    let targetVec = CoopVecFromVector<half>(target);

    let pred = forward(gEncodingParams, gParams, uv);

    let scale = CoopVec<half, 3>(half(1.0f / float(max(1u, gBatchSize))));
    let outputGrad = RelativeL2Luminance<half>::deriv(targetVec, pred, scale);

    var encBuf = FeatureBuffer(gEncodingParams);
    var encGrads = FeatureBufferGrad(gEncodingParamGrads);
    var wBuf = MatrixBiasBuffer(gParams);
    var wGrads = MatrixBiasBufferGrad(gParamGrads);

    bwd_diff(forward)(DifferentialPtrPair<FeatureBuffer>(encBuf, encGrads),
                      DifferentialPtrPair<MatrixBiasBuffer>(wBuf, wGrads), uv, outputGrad);
}
