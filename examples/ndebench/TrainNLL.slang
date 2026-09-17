import TSNN.CoopVec;
import TSNN.LinearOps;
import architectures.TMM;

ByteAddressBuffer gParams;
RWByteAddressBuffer gParamGrads;
StructuredBuffer<float2> gSamples;

// Keep the differentiated graph at module scope, as in examples/hggrid.  The
// namespace remains the public architecture API for inference and setup.
[Differentiable]
float forward_TMM(MatrixBiasBuffer weights, no_diff float2 x)
{
    let ctx = TMM.makeContext();

    let params = CoopVecToArray(TMM.kMLP.forward<TMM.Activation>(weights, ctx));
    return TMM.kGMM.eval(params, x);
}

// A kernel entry point with scalar `uniform` parameters (as opposed to fields
// inside a named cbuffer) reproducibly corrupts this environment's Vulkan
// device / silently reads back as zero -- see Optimize.slang's `InitCB`
// comment for the full writeup. `TrainCB` is the fix here, mirroring the
// cbuffer already used by every other kernel in this file's siblings.
cbuffer TrainCB
{
    uint  gOffset;
    uint  gCount;
    float gWeight;   // gLossScale / gBatchSize
}

[numthreads(256, 1, 1)]
void trainNLL_TMM(uint3 tid : SV_DispatchThreadID)
{
    let idx = gOffset + tid.x;
    if (tid.x >= gCount || idx >= gSamples.getCount()) return;

    let sample = gSamples[idx];
    float outputGrad = -gWeight;

    var wBuf = MatrixBiasBuffer(gParams);
    var wGrads = MatrixBiasBufferGrad(gParamGrads);
    bwd_diff(forward_TMM)(
        DifferentialPtrPair<MatrixBiasBuffer>(wBuf, wGrads), sample, outputGrad);
}
