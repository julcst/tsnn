// examples/hggrid/Infer.cs.slang
// Inference kernels for the HGGrid example.
//
// evalMain    – evaluates the learned density on a 2-D grid and writes
//               exp(log p(x)) to a float texture (for visualisation).
// evalNLLMain – log p(x) on a held-out batch; also the probe used by
//               --check-grad to finite-difference the GPU gradient.
// sampleMain  – ancestral samples via sampleModel, written as float2 into a
//               buffer.

import TSNN.CoopVec;
import TSNN.Utils.Math;
import TSNN.Utils.PCG32;
import Network;

// ─── Density evaluation ───────────────────────────────────────────────────────

ByteAddressBuffer   gHistLogits;
ByteAddressBuffer   gParams;
RWTexture2D<float4> gDensityTex;  // output: learned density (all four channels = p(x))

[numthreads(8, 8, 1)]
void evalMain(uint3 tid : SV_DispatchThreadID, uniform uint2 frameDim)
{
    uint2 pixel = tid.xy;
    if (any(pixel >= frameDim))
        return;

    // Map pixel centre to the unit square [0,1]^2 -- HGGrid's native domain,
    // so unlike examples/nde there is no [-domain, domain] rescale.
    float2 x = (float2(pixel) + 0.5f) / float2(frameDim);

    float logp    = forward(gHistLogits, gParams, x);
    float density = exp(logp);

    gDensityTex[pixel] = float4(density, density, density, 1.0f);
}

// ─── NLL evaluation on a held-out batch (also drives --check-grad) ────────────

ByteAddressBuffer    gTestSamples;
RWByteAddressBuffer  gLogProbsOut;

cbuffer NLLCB { uint gNLLCount; }

[numthreads(64, 1, 1)]
void evalNLLMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gNLLCount) return;

    let x = gTestSamples.LoadAligned<float2>(idx * sizeof(float2));  // Load float2 sample (8 bytes) from buffer

    float logp = forward(gHistLogits, gParams, x);
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

    var rng = PCG32(uint64_t(idx) + 1u, uint64_t(gSeed));
    let s = sampleModel(gHistLogits, gParams, rng);

    gSampleOut.StoreAligned(idx * sizeof(float2), s.x);
}
