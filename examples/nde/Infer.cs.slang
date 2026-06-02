// examples/nde/Infer.cs.slang
// Inference kernels for the NDE example.
//
// evalMain   – evaluates the learned density on a 2-D grid and writes
//              exp(log p(x)) to a float texture (for visualisation).
//
// sampleMain – generates samples by mapping Gaussian noise through the
//              inverse flow and writes them as float2 into a buffer.

import TSNN.CoopVec;
import TSNN.Utils.Math;
import Network;

// ─── Density evaluation ───────────────────────────────────────────────────────

ByteAddressBuffer   gParams;
RWTexture2D<float4> gDensityTex;  // output: learned density (all four channels = p(x))

[numthreads(8, 8, 1)]
void evalMain(uint3 tid : SV_DispatchThreadID, uniform uint2 frameDim)
{
    uint2 pixel = tid.xy;
    if (any(pixel >= frameDim))
        return;

    // Map pixel centre to domain [-kTailBound, kTailBound]²
    float2 uv = (float2(pixel) + 0.5f) / float2(frameDim);
    float2 x  = (uv * 2.0f - 1.0f) * kTailBound;

    float logp  = forward(gParams, x);
    float density = exp(logp);

    gDensityTex[pixel] = float4(density, density, density, 1.0f);
}

// ─── NLL evaluation on a held-out batch ───────────────────────────────────────

ByteAddressBuffer    gTestSamples;
RWByteAddressBuffer  gLogProbsOut;

cbuffer NLLCB { uint gNLLCount; }

[numthreads(64, 1, 1)]
void evalNLLMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gNLLCount) return;

    float2 x;
    x.x = asfloat(gTestSamples.Load(idx * 8u + 0u));
    x.y = asfloat(gTestSamples.Load(idx * 8u + 4u));

    float logp = forward(gParams, x);
    gLogProbsOut.Store(idx * 4u, asuint(logp));
}

// ─── Sampling ─────────────────────────────────────────────────────────────────

RWByteAddressBuffer gSampleOut;  // output: float2 samples (8 bytes each)

cbuffer SampleCB
{
    uint gNumSamples;
    uint gSeed;
}

[numthreads(64, 1, 1)]
void sampleMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gNumSamples)
        return;

    // Box-Muller transform from uniform PCG noise → bivariate Gaussian
    let r    = pcg3d(uint3(gSeed, idx, 1u));
    float u1 = (float(r.x) + 0.5f) / 4294967296.0f;
    float u2 = (float(r.y) + 0.5f) / 4294967296.0f;
    float mag = sqrt(-2.0f * log(max(u1, 1e-7f)));
    float phi = 2.0f * float.getPi() * u2;
    float2 z  = float2(mag * cos(phi), mag * sin(phi));

    float2 x = sampleFlow(gParams, z);

    gSampleOut.Store(idx * 8u + 0u, asuint(x.x));
    gSampleOut.Store(idx * 8u + 4u, asuint(x.y));
}
