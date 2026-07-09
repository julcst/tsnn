// Train.cs.slang - Training kernel for the image-learn example.
// One dispatch call processes a batch of randomly sampled pixels.

import TSNN.CoopVec;
import TSNN.Losses;
import TSNN.Encodings.FeatureBuffer;
import TSNN.LinearOps;
import TSNN.Utils.Math;
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
    float gLossScale;
}

[numthreads(64, 1, 1)]
void trainMain(uint3 tid: SV_DispatchThreadID)
{
    uint sampleIndex = tid.x;
    if (sampleIndex >= gBatchSize)
        return;

    // Deterministic-but-varied random pixel per sample
    let rand = pcg3d(uint3(gCurrentStep, gFrameIndex, sampleIndex));
    uint2 pixel = rand.xy % gFrameDim;
    float2 uv = (float2(pixel) + 0.5f) / float2(gFrameDim);

    let target = gTarget.Load(int3(pixel, 0)).xyz;
    let targetVec = CoopVecFromVector<half>(target);

    let pred = forward(gEncodingParams, gParams, uv);

    // Pre-scale by gLossScale before the fp16 gradient atomics (weight grads,
    // and backprop through them) so per-sample contributions don't land in
    // fp16's subnormal range; Optimize.cs.slang's Adam divides it back out.
    let scale = CoopVec<half, 3>(half(gLossScale / float(max(1u, gBatchSize))));
    let outputGrad = RelativeL2Luminance<half>::deriv(targetVec, pred, scale);

    var encBuf = FeatureBuffer(gEncodingParams);
    var encGrads = FeatureBufferGrad(gEncodingParamGrads);
    var wBuf = MatrixBiasBuffer(gParams);
    var wGrads = MatrixBiasBufferGrad(gParamGrads);

    bwd_diff(forward)(DifferentialPtrPair<FeatureBuffer>(encBuf, encGrads),
                      DifferentialPtrPair<MatrixBiasBuffer>(wBuf, wGrads), uv, outputGrad);
}
