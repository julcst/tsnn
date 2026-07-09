// Optimize.cs.slang - Adam optimizer kernel for the image-learn example.
// Vectorized float4/half4 processing for maximum memory bandwidth.

import TSNN.Optimizers.Adam;
import TSNN.Optimizers.Sanitizer;
import TSNN.Utils.RWBuffer;
import TSNN.Utils.MLP;
import TSNN.Encodings.FeatureBuffer;
import Network;

// Mixed-precision training: gParamsMaster holds the float32 authoritative MLP
// weights; Adam reads/updates this buffer, and gParams is the float16 compute
// mirror forward/backward reads from. A typical Adam step is ~lr = 1e-3; the
// float16 ULP at |w|=1 is ~9.8e-4 (barely one ULP), so weight-only-fp16
// storage rounds most updates back to their old value once |w| grows past
// ~0.5, freezing training. See NISTracer/Optimization.cs.slang for the same
// pattern and its derivation.
RWByteAddressBuffer gParams;
RWByteAddressBuffer gParamsMaster;
RWByteAddressBuffer gParamGrads;
RWByteAddressBuffer gMoments1;
RWByteAddressBuffer gMoments2;
RWByteAddressBuffer gEncodingParams;
RWByteAddressBuffer gEncodingParamGrads;
RWByteAddressBuffer gEncodingMoments1;
RWByteAddressBuffer gEncodingMoments2;

cbuffer CB
{
    float gLearningRate;
    float gCurrentStep;
    uint gDispatchThreadCount;
    float gLossScale;
}

[numthreads(256, 1, 1)]
void resetMain(uint3 tid: SV_DispatchThreadID)
{
    // SOTA He init for the MLP weights (dual-writes the float16 compute mirror
    // and float32 master from the same draw) and tiny-random init for the hash
    // grid; each helper owns its coalesced loop.  Adam moments are zeroed
    // host-side (clear_buffer in main.py).
    kMLP.initParams<half, float>(gParams, gParamsMaster, tid.x, gDispatchThreadCount);
    initFeatures<half>(gEncodingParams, kEncodingParamElementCount, tid.x, gDispatchThreadCount);
}

[numthreads(256, 1, 1)]
void optimizeMain(uint3 tid: SV_DispatchThreadID)
{
    var adamConfig = AdamConfig(gLearningRate);
    adamConfig.invLossScale = 1.0f / gLossScale;
    let adam = Adam(gCurrentStep, adamConfig);

    for (uint base = tid.x * 4u; base < kParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p = LoadFloat4(gParamsMaster, base);
        let g = sanitize(LoadHalf4(gParamGrads, base));
        var m1 = LoadFloat4(gMoments1, base);
        var m2 = LoadFloat4(gMoments2, base);
        adam.step(p, m1, m2, g);
        StoreFloat4(gMoments1, base, m1);
        StoreFloat4(gMoments2, base, m2);
        StoreFloat4(gParamsMaster, base, p); // keep full-precision master
        StoreHalf4(gParams, base, p);        // mirror to float16 for compute
    }

    // Always store: this used to be gated on `dot(step,step) > 1e-6` as a
    // bandwidth-saving skip for the oversized table's untouched entries, but
    // that also silently dropped small-but-real updates for entries that
    // *were* touched (especially late in training, as gradients shrink
    // approaching convergence) -- capping achievable PSNR ~10dB below tcnn
    // at the same step count instead of just slowing convergence.
    for (uint base = tid.x * 4u; base < kEncodingParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p = LoadHalf4(gEncodingParams, base);
        let g = sanitize(LoadFloat4(gEncodingParamGrads, base));
        var m1 = LoadFloat4(gEncodingMoments1, base);
        var m2 = LoadFloat4(gEncodingMoments2, base);
        adam.step(p, m1, m2, g);
        StoreFloat4(gEncodingMoments1, base, m1);
        StoreFloat4(gEncodingMoments2, base, m2);
        StoreHalf4(gEncodingParams, base, p);
    }
}
