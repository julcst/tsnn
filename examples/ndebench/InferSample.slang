import TSNN.CoopVec;
import TSNN.LinearOps;
import TSNN.Utils.PCG32;

import architectures.IArchitecture;
import architectures.TMM;
import architectures.HGGrid;
import architectures.DFN;
import architectures.DFL;
import architectures.NSFLinear;
import architectures.NSFQuadratic;
import architectures.NSFRQS;
import architectures.HDF;

ByteAddressBuffer gParams;
RWStructuredBuffer<float2> gSamples;

// A lone scalar `uniform` shader-entry-point parameter (as opposed to a named
// cbuffer field) reproducibly hangs the GPU on the dispatch *following* it --
// see Optimize.slang's `InitCB` for the full writeup.  `gSeed` goes through a
// cbuffer for the same reason `init_TMM`'s thread count does.
cbuffer SampleCB
{
    uint gSeed;
}

void inferSample<T : IArchitecture>(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gSamples.getCount()) return;
    var rand = PCG32(gSeed, idx);
    gSamples[idx] = T.sample(MatrixBiasBuffer(gParams), rand);
}

/// Generates inferSample kernel for a given architecture.
#define NDEBENCH_INFER_SAMPLE_KERNEL(ARCHITECTURE) \
    [numthreads(256, 1, 1)] \
    void inferSample_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        inferSample<ARCHITECTURE>(tid); \
    }

NDEBENCH_INFER_SAMPLE_KERNEL(TMM)
NDEBENCH_INFER_SAMPLE_KERNEL(HGGrid)
NDEBENCH_INFER_SAMPLE_KERNEL(DFN)
NDEBENCH_INFER_SAMPLE_KERNEL(DFL)
NDEBENCH_INFER_SAMPLE_KERNEL(NSFLinear)
NDEBENCH_INFER_SAMPLE_KERNEL(NSFQuadratic)
NDEBENCH_INFER_SAMPLE_KERNEL(NSFRQS)
NDEBENCH_INFER_SAMPLE_KERNEL(HDF)
