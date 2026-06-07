// Optimize.cs.slang - Adam optimizer kernel for the image-learn example.
// Vectorized float4/half4 processing for maximum memory bandwidth.

import TSNN.Optimizers.Adam;
import TSNN.Optimizers.Sanitizer;
import TSNN.Utils.RWBuffer;
import TSNN.Utils.MLP;
import TSNN.Encodings.FeatureBuffer;
import Network;

RWByteAddressBuffer gParams;
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
}

[numthreads(256, 1, 1)]
void resetMain(uint3 tid: SV_DispatchThreadID)
{
    // SOTA He init for the MLP weights and tiny-random init for the hash grid;
    // each helper owns its coalesced loop.  Adam moments are zeroed host-side
    // (clear_buffer in main.py).
    kMLP.initParams<half>(gParams, tid.x, gDispatchThreadCount);
    initFeatures<half>(gEncodingParams, kEncodingParamElementCount, tid.x, gDispatchThreadCount);
}

[numthreads(256, 1, 1)]
void optimizeMain(uint3 tid: SV_DispatchThreadID)
{
    let adamConfig = AdamConfig(gLearningRate);
    let adam = Adam(gCurrentStep, adamConfig);

    for (uint base = tid.x * 4u; base < kParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p = LoadHalf4(gParams, base);
        let g = sanitize(LoadHalf4(gParamGrads, base));
        var m1 = LoadFloat4(gMoments1, base);
        var m2 = LoadFloat4(gMoments2, base);
        adam.step(p, m1, m2, g);
        StoreFloat4(gMoments1, base, m1);
        StoreFloat4(gMoments2, base, m2);
        StoreHalf4(gParams, base, p);
    }

    for (uint base = tid.x * 4u; base < kEncodingParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p = LoadHalf4(gEncodingParams, base);
        let g = sanitize(LoadFloat4(gEncodingParamGrads, base));
        var m1 = LoadFloat4(gEncodingMoments1, base);
        var m2 = LoadFloat4(gEncodingMoments2, base);
        let updated = adam.step(p, m1, m2, g);
        StoreFloat4(gEncodingMoments1, base, m1);
        StoreFloat4(gEncodingMoments2, base, m2);
        if (updated)
            StoreHalf4(gEncodingParams, base, p);
    }
}
