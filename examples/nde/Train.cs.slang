// examples/nde/Train.cs.slang
// Training kernel for the NDE example.
//
// Each thread picks a random sample from the pre-generated data buffer,
// evaluates the flow's log-probability, and backpropagates the NLL gradient
// (-1/N per sample) into the parameter gradient buffer.
//
// The three-kernel training loop:
//   1. trainMain   – accumulate ∂L/∂params into gParamGrads  (this file)
//   2. optimizeMain – Adam step on gParams using gParamGrads  (Optimize.cs.slang)
//   3. evalMain     – render learned density to texture        (Infer.cs.slang)

import TSNN.CoopVec;
import TSNN.LinearOps;
import TSNN.Utils.Math;
import Network;

// gSamples holds a flat array of float2 data samples in [-tailBound, tailBound]²,
// pre-generated on the CPU from the target 2-D PDF.
ByteAddressBuffer    gSamples;
ByteAddressBuffer    gParams;
RWByteAddressBuffer  gParamGrads;

cbuffer CB
{
    uint  gNumSamples;   // total samples in gSamples
    uint  gBatchSize;    // threads per dispatch
    uint  gCurrentStep;
    float gLossScale;    // gradient pre-scale (undone in Optimize.cs); see below
}

[numthreads(64, 1, 1)]
void trainMain(uint3 tid : SV_DispatchThreadID)
{
    uint sampleIdx = tid.x;
    if (sampleIdx >= gBatchSize)
        return;

    // Deterministic but varied sample selection
    let rand   = pcg3d(uint3(gCurrentStep, sampleIdx, 0u));
    uint dataIdx = rand.x % gNumSamples;

    // Load float2 sample from buffer (2 × float32 = 8 bytes per sample)
    float2 x;
    x.x = asfloat(gSamples.Load(dataIdx * 8u + 0u));
    x.y = asfloat(gSamples.Load(dataIdx * 8u + 4u));

    // NLL loss gradient: d(-log p(x)) / d(log p(x)) = -1/N.
    // We scale by gLossScale so the per-sample contribution sits comfortably
    // inside float16's normal range: -1/16384 ≈ 6.1e-5 is at the float16
    // min-normal, so grad·activation would otherwise underflow to zero in the
    // float16 gradient buffer.  Optimize.cs UNDOES this scale (in float32) before
    // the optimiser step, so it only buys accumulation precision and does not
    // change the update — use the debug counters to pick a value that neither
    // underflows (too small) nor overflows the float16 accumulator (too large).
    float outputGrad = -gLossScale / float(max(1u, gBatchSize));

    var wBuf   = MatrixBiasBuffer(gParams);
    var wGrads = MatrixBiasBufferGrad(gParamGrads);

    bwd_diff(forward)(
        DifferentialPtrPair<MatrixBiasBuffer>(wBuf, wGrads),
        x,
        outputGrad
    );
}
