// examples/hggrid/Optimize.cs.slang
// Adam optimiser kernel for the HGGrid example.  Two independent parameter
// branches, both sanitized and Adam-stepped with the same invLossScale (a
// single bwd_diff(forward) seed fans out to both, so both see the identical
// pre-scaled cotangent):
//
//   · Histogram logits (gHistLogits) — plain float32, 64 scalars, no half
//     mirror.  Never touched by CoopVec matmul hardware, so the mixed-
//     precision dance that the fine MLP needs buys nothing here.
//   · Fine MLP weights (gParams / gParamsMaster) — float16 compute mirror +
//     float32 Adam master, exactly as examples/nde/Optimize.cs.slang: a
//     typical Adam step (~lr) is far below float16's ULP near |w|=1, so a
//     float16-only master would silently round small updates back to their
//     old value.
//
// resetMain  – histogram logits zeroed (exactly uniform coarse start); fine
//              MLP given a He-normal init with its last layer zeroed (so the
//              GMM starts predicting the zero-parameter mixture: one wide,
//              centred, equally-weighted Gaussian per component).
// optimizeMain – one Adam step per branch, sanitized (sanitizeDebug under
//              DEBUG_COUNTERS, flushed to two separate counter blocks so a
//              dead branch — e.g. the one-hot path not wired to the MLP — is
//              visible immediately instead of averaged away).

import TSNN.Optimizers.Adam;
import TSNN.Optimizers.Sanitizer;
import TSNN.Utils.RWBuffer;
import TSNN.Utils.Math;
import TSNN.Utils.MLP;
import Network;

#ifndef DEBUG_COUNTERS
#define DEBUG_COUNTERS 0
#endif

RWByteAddressBuffer gHistLogits;       // float32 (no half mirror)
RWByteAddressBuffer gHistLogitGrads;
RWByteAddressBuffer gHistMoments1;
RWByteAddressBuffer gHistMoments2;

RWByteAddressBuffer gParams;        // float16 mirror (read by forward/backward)
RWByteAddressBuffer gParamsMaster;  // float32 master (updated by Adam)
RWByteAddressBuffer gParamGrads;
RWByteAddressBuffer gMoments1;
RWByteAddressBuffer gMoments2;

#if DEBUG_COUNTERS
// Two 4×uint32 tallies back to back: [0..15] = histogram branch,
// [16..31] = fine-MLP branch.  Each block is [nonFinite, clipped, zero, total].
RWByteAddressBuffer gDebugCounters;
#endif

cbuffer CB
{
    float gLearningRate;
    float gCurrentStep;
    uint  gDispatchThreadCount;
    float gLossScale;    // gradients were pre-scaled by this in Train.cs
    float gGradClip;     // per-element clip, in TRUE (unscaled) gradient units
    float gWeightDecay;  // 0 → Adam, > 0 → AdamW (decoupled decay)
}

[numthreads(256, 1, 1)]
void resetMain(uint3 tid : SV_DispatchThreadID)
{
    // Exactly-uniform coarse start: every cell equally likely before any
    // gradient signal arrives.
    for (uint i = tid.x; i < kNumCells; i += gDispatchThreadCount)
        gHistLogits.StoreAligned<float>(i * sizeof(float), 0.0f);

    // Identity-ish init for the fine head: He hidden layers, zeroed last
    // layer (each cell starts predicting the zero-parameter GMM).
    kFineMLP.initParams<half, float>(gParams, gParamsMaster, tid.x, gDispatchThreadCount, MLPInit::HeNormal, true);
    // Adam moments are zeroed host-side (clear_buffer in main.py).
}

[numthreads(256, 1, 1)]
void optimizeMain(uint3 tid : SV_DispatchThreadID)
{
    let invLossScale = 1.0f / gLossScale;

    AdamConfig cfg = AdamConfig(gLearningRate);
    cfg.invLossScale = invLossScale;
    cfg.weightDecay  = gWeightDecay;
    let opt = Adam(gCurrentStep, cfg);

#if DEBUG_COUNTERS
    SanitizeDebugCounter histDbg = SanitizeDebugCounter();
    SanitizeDebugCounter mlpDbg  = SanitizeDebugCounter();
#endif

    // Histogram branch: 64 plain float32 scalars.
    for (uint base = tid.x * 4u; base < kNumCells; base += gDispatchThreadCount * 4u)
    {
        var p    = LoadFloat4(gHistLogits, base);
        let gRaw = LoadFloat4(gHistLogitGrads, base);
#if DEBUG_COUNTERS
        let g = sanitizeDebug(gRaw, gGradClip, histDbg);
#else
        let g = sanitize(gRaw, gGradClip);
#endif
        var m1 = LoadFloat4(gHistMoments1, base);
        var m2 = LoadFloat4(gHistMoments2, base);
        opt.step(p, m1, m2, g);
        StoreFloat4(gHistMoments1, base, m1);
        StoreFloat4(gHistMoments2, base, m2);
        StoreFloat4(gHistLogits,   base, p);
    }

    // Fine-MLP branch: float16 mirror + float32 master.
    for (uint base = tid.x * 4u; base < kParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p     = LoadFloat4(gParamsMaster, base);            // float32 master
        let gRaw  = LoadHalf4(gParamGrads, base);               // scaled, float16
#if DEBUG_COUNTERS
        let g     = sanitizeDebug(gRaw, gGradClip, mlpDbg);
#else
        let g     = sanitize(gRaw, gGradClip);
#endif
        var m1    = LoadFloat4(gMoments1, base);
        var m2    = LoadFloat4(gMoments2, base);
        opt.step(p, m1, m2, g);
        StoreFloat4(gMoments1,     base, m1);
        StoreFloat4(gMoments2,     base, m2);
        StoreFloat4(gParamsMaster, base, p);                   // keep full-precision master
        StoreHalf4(gParams,        base, p);                   // mirror to float16 for compute
    }

#if DEBUG_COUNTERS
    histDbg.flush(gDebugCounters, 0u);
    mlpDbg.flush(gDebugCounters, 16u);
#endif
}
