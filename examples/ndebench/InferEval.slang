import TSNN.CoopVec;
import TSNN.LinearOps;

import architectures.IArchitecture;
import architectures.TMM;
import architectures.HGGrid;
import architectures.DFN;
import architectures.DFL;
import architectures.NSFLinear;
import architectures.NSFQuadratic;
import architectures.NSFRQS;
import architectures.HDF;
import architectures.HDFLevels;
import architectures.HGGridLevels;

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

NDEBENCH_INFER_EVAL_KERNEL(TMM)
NDEBENCH_INFER_EVAL_KERNEL(HGGrid)
NDEBENCH_INFER_EVAL_KERNEL(DFN)
NDEBENCH_INFER_EVAL_KERNEL(DFL)
NDEBENCH_INFER_EVAL_KERNEL(DFN16)
NDEBENCH_INFER_EVAL_KERNEL(DFL16)
NDEBENCH_INFER_EVAL_KERNEL(NSFLinear)
NDEBENCH_INFER_EVAL_KERNEL(NSFQuadratic)
NDEBENCH_INFER_EVAL_KERNEL(NSFRQS)
NDEBENCH_INFER_EVAL_KERNEL(HDF)
NDEBENCH_INFER_EVAL_KERNEL(HDFG4L3)
NDEBENCH_INFER_EVAL_KERNEL(HDFG3L4)
NDEBENCH_INFER_EVAL_KERNEL(HDFG4L3B)
NDEBENCH_INFER_EVAL_KERNEL(HDFG3L4B)
NDEBENCH_INFER_EVAL_KERNEL(HDFG4L3S)
NDEBENCH_INFER_EVAL_KERNEL(HGGridG2L3)
NDEBENCH_INFER_EVAL_KERNEL(HGGridG2L3B)
NDEBENCH_INFER_EVAL_KERNEL(HGGridG4L2)
NDEBENCH_INFER_EVAL_KERNEL(HGGridG4L2B)
NDEBENCH_INFER_EVAL_KERNEL(HGGridG4L2S)
NDEBENCH_INFER_EVAL_KERNEL(HDFFastest)
NDEBENCH_INFER_EVAL_KERNEL(HDFFast)
NDEBENCH_INFER_EVAL_KERNEL(HGGridFastest)
NDEBENCH_INFER_EVAL_KERNEL(HGGridFast)
