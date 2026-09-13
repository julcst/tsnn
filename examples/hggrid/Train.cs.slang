// examples/hggrid/Train.cs.slang
// Two training kernels for the HGGrid example.
//
// trainMain    – MLE / NLL: pick a random sample from the data buffer,
//                backpropagate -lossScale/batch.  Standard maximum-likelihood
//                training.
//
// trainKLMain  – KL divergence (Mueller et al. 2019, §3.1): draw x ~ q_θ from
//                the current model via sampleModel's single ancestral pass,
//                look up unnormalized target f(x) from a flat luminance
//                buffer, then backpropagate with importance weight
//                -f(x) / q_θ(x) / N.
//
//                ∇_θ KL(p‖q_θ) = E_{x~q_θ}[ -f(x)/q_θ(x) · ∇_θ log q_θ(x) ]
//
//                The global scale of f cancels in Adam, so f can be raw
//                luminance (no explicit normalisation needed).
//
// Domain is the unit square [0,1]^2 (TruncatedGMM's native truncation), so
// unlike examples/nde there is no ±domain rescale and no row flip: row 0 of
// the luminance buffer already corresponds to x.y = 0, matching evalMain and
// image_to_samples in main.py.
//
// Resources shared by both kernels: gHistLogits/gHistLogitGrads (coarse),
// gParams/gParamGrads (fine MLP).  Each kernel also has its own data source
// and cbuffer.

import TSNN.CoopVec;
import TSNN.LinearOps;
import TSNN.Utils.PCG32;
import TSNN.Utils.Math;
import Network;

// ─── Shared resources ─────────────────────────────────────────────────────────

ByteAddressBuffer    gHistLogits;
RWByteAddressBuffer  gHistLogitGrads;
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

    var histBuf   = HistLogitBuffer(gHistLogits);
    var histGrads = HistLogitBufferGrad(gHistLogitGrads);
    var wBuf      = MatrixBiasBuffer(gParams);
    var wGrads    = MatrixBiasBufferGrad(gParamGrads);
    bwd_diff(forward)(
        DifferentialPtrPair<HistLogitBuffer>(histBuf, histGrads),
        DifferentialPtrPair<MatrixBiasBuffer>(wBuf, wGrads),
        x, outputGrad);
}

// ─── Deterministic full-batch gradient (drives main.py's --check-grad) ────────
//
// Same accumulation target buffers and bwd_diff(forward) call as trainMain,
// but every thread reads sample `tid.x` directly instead of trainMain's
// pcg3d-hashed index.  That makes the exact batch a Python caller uploads to
// gCheckSamples the exact batch the backward pass differentiates, so
// evalNLLMain (Infer.cs.slang) can finite-difference against the same buffer
// with no need to reproduce Train.cs's RNG on the CPU.  Methodology mirrors
// test_parameter_gradients in tests/truncGMM/test_truncated_gmm.py.

ByteAddressBuffer gCheckSamples;

cbuffer CheckCB
{
    uint  gCheckBatchSize;
    float gCheckLossScale;
}

[numthreads(64, 1, 1)]
void trainCheckMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gCheckBatchSize)
        return;

    float2 x = gCheckSamples.LoadAligned<float2>(idx * sizeof(float2));
    float outputGrad = -gCheckLossScale / float(max(1u, gCheckBatchSize));

    var histBuf   = HistLogitBuffer(gHistLogits);
    var histGrads = HistLogitBufferGrad(gHistLogitGrads);
    var wBuf      = MatrixBiasBuffer(gParams);
    var wGrads    = MatrixBiasBufferGrad(gParamGrads);
    bwd_diff(forward)(
        DifferentialPtrPair<HistLogitBuffer>(histBuf, histGrads),
        DifferentialPtrPair<MatrixBiasBuffer>(wBuf, wGrads),
        x, outputGrad);
}

// ─── KL kernel (Mueller et al. 2019, §3.1) ────────────────────────────────────

// H×W float32 luminance, row-major.  Row 0 = top of image = x.y = 0.
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

    var rng = PCG32(uint64_t(gKLCurrentStep) * 65521u + uint64_t(idx) + 1u, uint64_t(idx) * 2u + 1u);
    let s    = sampleModel(gHistLogits, gParams, rng);   // (x, log q_θ(x)) in one pass
    float2 x = s.x;

    uint px = clamp(uint(x.x * float(gTexWidth)),  0u, gTexWidth  - 1u);
    uint py = clamp(uint(x.y * float(gTexHeight)), 0u, gTexHeight - 1u);
    float f = gTargetBuf.LoadAligned<float>((py * gTexWidth + px) * sizeof(float));

    float w = f / max(exp(s.logProb), 1e-10f);

    // Score-function gradient:  -w/N  (eq. 4, Mueller et al. 2019)
    float outputGrad = -w * gKLLossScale / float(max(1u, gKLBatchSize));

    var histBuf   = HistLogitBuffer(gHistLogits);
    var histGrads = HistLogitBufferGrad(gHistLogitGrads);
    var wBuf      = MatrixBiasBuffer(gParams);
    var wGrads    = MatrixBiasBufferGrad(gParamGrads);
    bwd_diff(forward)(
        DifferentialPtrPair<HistLogitBuffer>(histBuf, histGrads),
        DifferentialPtrPair<MatrixBiasBuffer>(wBuf, wGrads),
        x, outputGrad);
}
