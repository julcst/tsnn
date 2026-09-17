import TSNN.CoopVec;
import TSNN.LinearOps;
import architectures.TMM;

ByteAddressBuffer gParams;
StructuredBuffer<float2> gSamples;
RWStructuredBuffer<float> gLogPDFs;

/// Generates inferEval kernel for a given architecture.
#define NDEBENCH_INFER_EVAL_KERNEL(ARCHITECTURE) \
    [numthreads(256, 1, 1)] \
    void inferEval_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        let idx = tid.x; \
        if (idx >= gSamples.getCount()) return; \
        gLogPDFs[idx] = ARCHITECTURE.evalLogPDF(MatrixBiasBuffer(gParams), gSamples[idx]); \
    }

NDEBENCH_INFER_EVAL_KERNEL(TMM)
