// examples/nde/Train.cs.slang
// Two training kernels for the NDE example.
//
// trainMain    – MLE / NLL: pick a random sample from the data buffer,
//                backpropagate -1/N.  Standard maximum-likelihood training.
//
// trainKLMain  – KL divergence (Mueller et al. 2019, §3.1): draw x ~ q_θ from
//                the current model via the inverse flow, look up unnormalized
//                target f(x) from a flat luminance buffer, then backpropagate
//                with importance weight  -f(x) / q_θ(x) / N.
//
//                ∇_θ KL(p‖q_θ) = E_{x~q_θ}[ -f(x)/q_θ(x) · ∇_θ log q_θ(x) ]
//
//                The global scale of f cancels in Adam, so f can be raw luminance
//                (no explicit normalisation needed).
//
// Resources shared by both kernels:  gParams, gParamGrads.
// Each kernel also has its own data source and cbuffer.

import TSNN.CoopVec;
import TSNN.LinearOps;
import TSNN.Utils.Math;
import Network;

// ─── Shared resources ─────────────────────────────────────────────────────────

ByteAddressBuffer    gParams;
RWByteAddressBuffer  gParamGrads;

// ─── MLE kernel ───────────────────────────────────────────────────────────────

ByteAddressBuffer gSamples;

cbuffer CB
{
    uint  gNumSamples;   // total samples in gSamples
    uint  gBatchSize;    // threads per dispatch
    uint  gCurrentStep;
    float gLossScale;    // gradient pre-scale (undone in Optimize.cs)
}

[numthreads(64, 1, 1)]
void trainMain(uint3 tid : SV_DispatchThreadID)
{
    uint sampleIdx = tid.x;
    if (sampleIdx >= gBatchSize)
        return;

    let rand     = pcg3d(uint3(gCurrentStep, sampleIdx, 0u));
    uint dataIdx = rand.x % gNumSamples;
    float2 x     = gSamples.LoadAligned<float2>(dataIdx * sizeof(float2));

    float outputGrad = -gLossScale / float(max(1u, gBatchSize));

    var wBuf   = MatrixBiasBuffer(gParams);
    var wGrads = MatrixBiasBufferGrad(gParamGrads);
    bwd_diff(forward)(DifferentialPtrPair<MatrixBiasBuffer>(wBuf, wGrads), x, outputGrad);
}

// ─── KL kernel (Mueller et al. 2019, §3.1) ────────────────────────────────────

// H×W float32 luminance, row-major.  Row 0 = top of image = x.y = -kDomain,
// matching the convention used by evalMain and image_to_samples in main.py.
ByteAddressBuffer gTargetBuf;

cbuffer KLCB
{
    uint  gKLBatchSize;
    uint  gKLCurrentStep;
    float gKLLossScale;
    uint  gTexWidth;
    uint  gTexHeight;
}

[numthreads(64, 1, 1)]
void trainKLMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gKLBatchSize)
        return;

    // x ~ q_θ and log q_θ(x) in one inverse pass — avoids a separate forward() call.
    let r    = pcg3d(uint3(gKLCurrentStep * 65521u + 1u, idx, 13u));
    float2 u = uintToUnitFloat(r.xy);
    float2 z = samplePrior(u);
    let    s = sampleFlowLogProb(gParams, z);           // (x, log q_θ(x))
    float2 x = clamp(s.x, float2(-kDomain), float2(kDomain));

    // Map world coordinates to image pixel.
    // image_to_samples maps image row 0 (top) → y = +kDomain, so high y is the
    // top of the image.  Row 0 of gTargetBuf is the top of the PNG, so we must
    // flip: row = (1 − uv.y) × H so that y=+1 → row 0 (top of buffer).
    float2 uv = (x / kDomain) * 0.5f + 0.5f;                          // [0, 1]
    uint   px = clamp(uint(uv.x * float(gTexWidth)),             0u, gTexWidth  - 1u);
    uint   py = clamp(uint((1.0f - uv.y) * float(gTexHeight)),   0u, gTexHeight - 1u);
    float  f  = gTargetBuf.LoadAligned<float>((py * gTexWidth + px) * sizeof(float));

    float w = f / max(exp(s.logProb), 1e-10f);

    // Score-function gradient:  -w/N  (eq. 4, Mueller et al. 2019)
    float outputGrad = -w * gKLLossScale / float(max(1u, gKLBatchSize));

    var wBuf   = MatrixBiasBuffer(gParams);
    var wGrads = MatrixBiasBufferGrad(gParamGrads);
    bwd_diff(forward)(DifferentialPtrPair<MatrixBiasBuffer>(wBuf, wGrads), x, outputGrad);
}
