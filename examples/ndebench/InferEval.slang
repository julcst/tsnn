import TSNN.CoopVec;
import TSNN.LinearOps;

import architectures.IArchitecture;
import architectures.TMM;
import architectures.HGGrid;

ByteAddressBuffer gParams;
StructuredBuffer<float2> gSamples;
RWStructuredBuffer<float> gLogPDFs;

void inferEval<T : IArchitecture>(uint3 tid : SV_DispatchThreadID)
{
    let idx = tid.x;
    if (idx >= gSamples.getCount()) return;
    gLogPDFs[idx] = T.evalLogPDF(MatrixBiasBuffer(gParams), gSamples[idx]);
}

/// Generates inferEval kernel for a given architecture.
#define NDEBENCH_INFER_EVAL_KERNEL(ARCHITECTURE) \
    [numthreads(256, 1, 1)] \
    void inferEval_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        inferEval<ARCHITECTURE>(tid); \
    }

NDEBENCH_INFER_EVAL_KERNEL(TMM)
NDEBENCH_INFER_EVAL_KERNEL(HGGrid)
