// Optimize.cs.slang - Adam optimizer kernel for the image-learn example.
// Vectorized float4/half4 processing for maximum memory bandwidth.

import TSNN.Optimizers.Adam;
import TSNN.Utils.PCG32;
import TSNN.Utils.RWBuffer;
import TSNN.Utils.Math;
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
    PCG32 rng = PCG32(uint64_t(tid.x), uint64_t(tid.x) ^ 0x9e3779b97f4a7c15ULL);

    for (uint i = tid.x; i < kParamElementCount; i += gDispatchThreadCount)
    {
        float scale = sqrt(2.0f / float(64));
        float p = rng.nextFloat() * 2.0f - 1.0f;
        gParams.StoreAligned<half>(i * sizeof(half), half(p * scale));
        gMoments1.StoreAligned<float>(i * sizeof(float), 0.0f);
        gMoments2.StoreAligned<float>(i * sizeof(float), 0.0f);
    }
    for (uint i = tid.x; i < kEncodingParamElementCount; i += gDispatchThreadCount)
    {
        float p = rng.nextFloat() * 2.0f - 1.0f;
        gEncodingParams.StoreAligned<half>(i * sizeof(half), half(p * 1e-3f));
        gEncodingMoments1.StoreAligned<float>(i * sizeof(float), 0.0f);
        gEncodingMoments2.StoreAligned<float>(i * sizeof(float), 0.0f);
    }
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
