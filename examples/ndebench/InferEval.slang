import TSNN.CoopVec;
import TSNN.LinearOps;

import architectures.IArchitecture;
#include "Configurations.slangh"

ByteAddressBuffer gParams;
StructuredBuffer<float2> gSamples;
RWStructuredBuffer<float> gLogPDFs;
StructuredBuffer<uint2> gLayout;

void inferEval<T : IArchitecture>(uint3 tid : SV_DispatchThreadID)
{
    let idx = tid.x;
    if (idx >= gSamples.getCount()) return;
    gLogPDFs[idx] = T.evalLogPDF(MatrixBiasBuffer(gParams), gLayout, gSamples[idx]);
}

/// Generates inferEval kernel for a given architecture.
#define NDEBENCH_INFER_EVAL_KERNEL(ARCHITECTURE) \
    [numthreads(256, 1, 1)] \
    void inferEval_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        inferEval<ARCHITECTURE>(tid); \
    }


#define NDEBENCH_MODEL(NAME) NDEBENCH_INFER_EVAL_KERNEL(NAME)
#define NDEBENCH_GRID(NAME, G, L, H, D, K) NDEBENCH_MODEL(NAME)
#include "Models.slangh"
#undef NDEBENCH_GRID
#undef NDEBENCH_MODEL
