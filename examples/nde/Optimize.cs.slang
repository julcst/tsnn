// examples/nde/Optimize.cs.slang
// Adam optimiser kernel for the NDE example.
//
// Mixed-precision training (the right way):
//   · gParamsMaster — float32 master copy of the weights; Adam updates this.
//   · gParams       — float16 mirror used by the forward/backward GPU compute.
// Storing the master in float32 is essential: a typical Adam step is ~LR, which
// is far below the float16 ULP (≈1e-3 near |w|=1), so a float16 master would
// silently round every small update back to its old value — frozen weights.
//
// resetMain  – initialises both buffers and zero-fills moment buffers.
//   · Hidden-layer weights/biases: small He/Xavier random
//   · Output-layer weights and biases: zero → initial affine is identity (s=0, t=0)
//
// optimizeMain – one Adam step on the float32 master, then mirror to float16.

import TSNN.Optimizers.Adam;
import TSNN.Optimizers.Sanitizer;
import TSNN.Utils.PCG32;
import TSNN.Utils.RWBuffer;
import TSNN.Utils.Math;
import Network;

// Adam vs AdamW is now selected purely by gWeightDecay (0 → Adam, > 0 → AdamW);
// no separate kernel variant is needed.
// Debug instrumentation: count gradient under/overflows to tune the loss scale.
#ifndef DEBUG_COUNTERS
#define DEBUG_COUNTERS 0
#endif

RWByteAddressBuffer gParams;        // float16 mirror (read by forward/backward)
RWByteAddressBuffer gParamsMaster;  // float32 master (updated by Adam)
RWByteAddressBuffer gParamGrads;
RWByteAddressBuffer gMoments1;
RWByteAddressBuffer gMoments2;

#if DEBUG_COUNTERS
// 4 × uint atomic tally: [0]=non-finite (overflow), [1]=clipped (saturation),
// [2]=exactly-zero (underflow proxy), [3]=elements processed.
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
    PCG32 rng = PCG32(uint64_t(tid.x), uint64_t(tid.x) ^ 0x9e3779b97f4a7c15ULL);

    float hiddenScale = sqrt(2.0f / float(kHidden));

    for (uint i = tid.x; i < kParamElementCount; i += gDispatchThreadCount)
    {
        uint byteOff   = i * sizeof(half);
        uint localByte = byteOff % kMLPBlockBytes;

        float val;
        if (localByte >= kOutputWeightStart)
        {
            // Output-layer weights and biases → zero so the initial transform is identity
            val = 0.0f;
        }
        else
        {
            // Hidden layers → small He/Xavier random
            val = (rng.nextFloat() * 2.0f - 1.0f) * hiddenScale;
        }

        gParamsMaster.StoreAligned<float>(i * sizeof(float), val);
        gParams.StoreAligned<half>(byteOff, half(val));
        gMoments1.StoreAligned<float>(i * sizeof(float), 0.0f);
        gMoments2.StoreAligned<float>(i * sizeof(float), 0.0f);
    }
}

[numthreads(256, 1, 1)]
void optimizeMain(uint3 tid : SV_DispatchThreadID)
{
    // Loss-scale correction: gradients were scaled up by gLossScale in Train.cs
    // so they accumulate inside float16's normal range.  We UNDO that here, in
    // float32, *before* clipping and before the optimiser sees them — so the
    // clip threshold and Adam's epsilon both act in true gradient units and are
    // independent of the loss scale.  (The previous code clipped the scaled
    // gradient, which silently made the effective clip = gGradClip/gLossScale
    // and the effective epsilon = eps/gLossScale.)
    let invLossScale = 1.0f / gLossScale;

    // Use the defaulting constructor so beta1/beta2/epsilon get their defaults
    // (a plain `AdamConfig cfg;` would leave epsilon = 0 → 0/0 NaN on zero grads).
    AdamConfig cfg = AdamConfig(gLearningRate);
    cfg.invLossScale = invLossScale;    // applied inside Adam.step
    cfg.weightDecay  = gWeightDecay;    // 0 → Adam, > 0 → AdamW (decoupled decay)
    let opt = Adam(gCurrentStep, cfg);

#if DEBUG_COUNTERS
    SanitizeDebugCounter dbg = SanitizeDebugCounter();
#endif

    for (uint base = tid.x * 4u; base < kParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p     = LoadFloat4(gParamsMaster, base);            // float32 master
        let gRaw  = LoadHalf4(gParamGrads, base);               // scaled, float16
#if DEBUG_COUNTERS
        let g     = sanitizeDebug(gRaw, gGradClip, dbg);        // scrub + clip + tally
#else
        let g     = sanitize(gRaw, gGradClip);                  // scrub + clip
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
    dbg.flush(gDebugCounters);
#endif
}
