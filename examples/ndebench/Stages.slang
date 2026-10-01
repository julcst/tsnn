// Instrumented kernels for layers_viz/plot_real_layers.py -- not used by
// benchmark.py or any other kernel, purely a diagnostic side door onto the
// real GPU-trained networks:
//
//   hero_<Arch>            one deterministic "typical" sample trajectory
//                          (prior -> after layer 1 -> final), via each
//                          architecture's sampleTrajectoryFixed() (every
//                          random draw sample() would make is fixed at the
//                          median quantile 0.5 instead).
//   stage1Marginal_<Arch>  the exact after-layer-1 density, with no sampling
//                          noise at all: DFN16/HDF/HGGrid's layer 1 doesn't
//                          depend on any continuous input (IArchitecture's
//                          ctx is a trivial constant), so it's a small
//                          softmax read-back; NSF-Linear's layer 1 does
//                          depend on a continuous (still-latent) z1, so it's
//                          evaluated at the same fixed z1=0 the hero_
//                          trajectory itself uses (layer 2 simply isn't
//                          applied): invert that one coupling step and sum
//                          log-prior + logAbsDet, same as evalLogPDF itself
//                          does per block. No integration over z1 either.
import TSNN.CoopVec;
import TSNN.LinearOps;

import architectures.IArchitecture;
import architectures.DFN;
import architectures.HDF;
import architectures.HGGrid;
import architectures.NSFLinear;

ByteAddressBuffer gParams;
StructuredBuffer<uint2> gLayout;

RWStructuredBuffer<float2> gHeroPrior;
RWStructuredBuffer<float2> gHeroAfterLayer1;
RWStructuredBuffer<float2> gHeroFinal;

#define NDEBENCH_HERO_KERNEL(ARCHITECTURE) \
    [numthreads(1, 1, 1)] \
    void hero_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        if (tid.x != 0) return; \
        let ctx = ARCHITECTURE.makeContext(); \
        float2 prior, afterLayer1, outFinal; \
        ARCHITECTURE.sampleTrajectoryFixed(MatrixBiasBuffer(gParams), gLayout, ctx, prior, afterLayer1, outFinal); \
        gHeroPrior[0] = prior; \
        gHeroAfterLayer1[0] = afterLayer1; \
        gHeroFinal[0] = outFinal; \
    }

NDEBENCH_HERO_KERNEL(NSFLinear)
NDEBENCH_HERO_KERNEL(DFN16)
NDEBENCH_HERO_KERNEL(HDF)
NDEBENCH_HERO_KERNEL(HGGrid)

RWStructuredBuffer<float> gStage1Marginal;  // sized for the largest use (RES for NSFLinear; only [0,64) used otherwise)

cbuffer Stage1CB
{
    uint gStage1Count;       // NSFLinear only: number of x0 query points (RES)
}

[numthreads(16, 1, 1)]
void stage1Marginal_DFN16(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    let ctx = DFN16.makeContext();
    let probs = DFN16.stage1MarginalX(MatrixBiasBuffer(gParams), gLayout, ctx);
    [ForceUnroll]
    for (uint i = 0; i < 16; ++i) gStage1Marginal[i] = probs[i];
}

[numthreads(64, 1, 1)]
void stage1Marginal_HDF(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    let ctx = HDF.makeContext();
    let probs = HDF.stage1MarginalCoarse(MatrixBiasBuffer(gParams), gLayout, ctx);
    [ForceUnroll]
    for (uint i = 0; i < 64; ++i) gStage1Marginal[i] = probs[i];
}

[numthreads(64, 1, 1)]
void stage1Marginal_HGGrid(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    let ctx = HGGrid.makeContext();
    let probs = HGGrid.stage1MarginalCoarse(MatrixBiasBuffer(gParams), gLayout, ctx);
    [ForceUnroll]
    for (uint i = 0; i < 64; ++i) gStage1Marginal[i] = probs[i];
}

[numthreads(256, 1, 1)]
void stage1Marginal_NSFLinear(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gStage1Count) return;
    let ctx = NSFLinear.makeContext();
    float z0Query = (float(idx) + 0.5f) / float(gStage1Count) - 0.5f;
    // Log-density, matching InferEval.slang's inferEval (exp'd host-side) --
    // not a raw density like the other three architectures' stage1Marginal_
    // kernels, since this one goes through kPrior.logProb + logAbsDet.
    gStage1Marginal[idx] = NSFLinear.stage1LogDensityX0(MatrixBiasBuffer(gParams), gLayout, ctx, z0Query);
}
