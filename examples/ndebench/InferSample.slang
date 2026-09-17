import TSNN.CoopVec;
import TSNN.LinearOps;
import TSNN.Utils.PCG32;
import architectures.TMM;

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

// `TMM` becomes `inferSample_TMM`.
#define NDEBENCH_INFER_SAMPLE_KERNEL(ARCHITECTURE) \
    [numthreads(256, 1, 1)] \
    void inferSample_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        uint idx = tid.x; \
        if (idx >= gSamples.getCount()) return; \
        var rand = PCG32(gSeed, idx); \
        gSamples[idx] = ARCHITECTURE.sample(MatrixBiasBuffer(gParams), rand); \
    }

NDEBENCH_INFER_SAMPLE_KERNEL(TMM)
