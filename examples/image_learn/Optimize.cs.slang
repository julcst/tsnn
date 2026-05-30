// Optimize.cs.slang - Adam optimizer kernel for the image-learn example.
// Vectorized float4/half4 processing for maximum memory bandwidth.

import TSNN.Optimizers.Adam;
import TSNN.Utils.PCG32;
import MLP;

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
    uint  gDispatchThreadCount;
}

float4 loadFloat4(RWByteAddressBuffer b, uint base) { return b.LoadAligned<float4>(base * sizeof(float)); }
void   storeFloat4(RWByteAddressBuffer b, uint base, float4 v) { b.StoreAligned<float4>(base * sizeof(float), v); }
float4 loadHalf4(RWByteAddressBuffer b, uint base) { return float4(b.LoadAligned<half4>(base * sizeof(half))); }
void   storeHalf4(RWByteAddressBuffer b, uint base, float4 v) { b.StoreAligned<half4>(base * sizeof(half), half4(v)); }

float4 sanitize4(float4 v)
{
    float4 r;
    r.x = (isnan(v.x) || isinf(v.x)) ? 0.f : v.x;
    r.y = (isnan(v.y) || isinf(v.y)) ? 0.f : v.y;
    r.z = (isnan(v.z) || isinf(v.z)) ? 0.f : v.z;
    r.w = (isnan(v.w) || isinf(v.w)) ? 0.f : v.w;
    return r;
}

[numthreads(256, 1, 1)]
void resetMain(uint3 tid : SV_DispatchThreadID)
{
    PCG32 rng = PCG32(uint64_t(tid.x), uint64_t(tid.x) ^ 0x9e3779b97f4a7c15ULL);

    for (uint i = tid.x; i < kParamElementCount; i += gDispatchThreadCount)
    {
        float scale = sqrt(2.0f / float(kHiddenSize));
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
void optimizeMain(uint3 tid : SV_DispatchThreadID)
{
    let adam = Adam(gLearningRate, kAdamBeta1, kAdamBeta2, kAdamEpsilon, gCurrentStep, kOptimizerLossScale);

    for (uint base = tid.x * 4u; base < kParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p  = loadHalf4(gParams, base);
        let g  = sanitize4(loadHalf4(gParamGrads, base));
        var m1 = loadFloat4(gMoments1, base);
        var m2 = loadFloat4(gMoments2, base);
        adam.step(p, m1, m2, g);
        storeFloat4(gMoments1, base, m1);
        storeFloat4(gMoments2, base, m2);
        storeHalf4(gParams, base, p);
    }

    for (uint base = tid.x * 4u; base < kEncodingParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p  = loadHalf4(gEncodingParams, base);
        let g  = sanitize4(loadFloat4(gEncodingParamGrads, base));
        var m1 = loadFloat4(gEncodingMoments1, base);
        var m2 = loadFloat4(gEncodingMoments2, base);
        let updated = adam.step(p, m1, m2, g);
        storeFloat4(gEncodingMoments1, base, m1);
        storeFloat4(gEncodingMoments2, base, m2);
        if (updated) storeHalf4(gEncodingParams, base, p);
    }
}
