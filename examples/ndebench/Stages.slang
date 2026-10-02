import TSNN.CoopVec;
import TSNN.LinearOps;

import architectures.IArchitecture;
#include "Configurations.slangh"

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
NDEBENCH_HERO_KERNEL(HDF_G8L2_H32D2)
NDEBENCH_HERO_KERNEL(HGGrid_G4L2_H16D3_K8)

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

#define NDEBENCH_STAGE1_GRID(NAME, COUNT) \
    [numthreads(64, 1, 1)] \
    void stage1Marginal_ ## NAME(uint3 tid : SV_DispatchThreadID) \
    { \
        if (tid.x != 0) return; \
        let probs = NAME.stage1MarginalCoarse(MatrixBiasBuffer(gParams), gLayout, NAME.makeContext()); \
        [ForceUnroll] for (uint i = 0; i < COUNT; ++i) gStage1Marginal[i] = probs[i]; \
    }

NDEBENCH_STAGE1_GRID(HDF_G8L2_H32D2, 64)
NDEBENCH_STAGE1_GRID(HGGrid_G4L2_H16D3_K8, 16)

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
