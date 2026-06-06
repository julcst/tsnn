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

    // Map pixel centre to the normalized domain [-kDomain, kDomain]²
    float2 uv = (float2(pixel) + 0.5f) / float2(frameDim);
    float2 x  = (uv * 2.0f - 1.0f) * kDomain;

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

    let x = gTestSamples.LoadAligned<float2>(idx * sizeof(float2));  // Load float2 sample (8 bytes) from buffer

    float logp = forward(gParams, x);
    gLogProbsOut.StoreAligned(idx * sizeof(float), logp);
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

    // Draw a base latent from the configured prior, then run the inverse flow.
    let r    = pcg3d(uint3(gSeed, idx, 1u));
    float2 u01 = uintToUnitFloat(r.xy);
    float2 z   = samplePrior(u01);

    float2 x = sampleFlow(gParams, z);

    gSampleOut.StoreAligned(idx * sizeof(float2), x);
}