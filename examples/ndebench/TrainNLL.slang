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

ByteAddressBuffer gParams;
RWByteAddressBuffer gParamGrads;
StructuredBuffer<float2> gSamples;

// A kernel entry point with scalar `uniform` parameters (as opposed to fields
// inside a named cbuffer) reproducibly corrupts this environment's Vulkan
// device / silently reads back as zero -- see Optimize.slang's `InitCB`
// comment for the full writeup. `TrainCB` is the fix here, mirroring the
// cbuffer already used by every other kernel in this file's siblings.
cbuffer TrainCB
{
    uint  gCount;
    float gWeight;   // gLossScale / gBatchSize
}

[numthreads(256, 1, 1)]
void trainNLL<T : IArchitecture>(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;

    let sample = gSamples[tid.x];
    float outputGrad = -gWeight;

    var wBuf = MatrixBiasBuffer(gParams);
    var wGrads = MatrixBiasBufferGrad(gParamGrads);
    bwd_diff(T.evalLogPDF)(DifferentialPtrPair<MatrixBiasBuffer>(wBuf, wGrads), sample, outputGrad);
}

/// Generates trainNLL kernel for a given architecture.
#define NDEBENCH_TRAIN_NLL_KERNEL(ARCHITECTURE) \
    [numthreads(256, 1, 1)] \
    void trainNLL_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        trainNLL<ARCHITECTURE>(tid); \
    }

NDEBENCH_TRAIN_NLL_KERNEL(TMM)
NDEBENCH_TRAIN_NLL_KERNEL(HGGrid)
NDEBENCH_TRAIN_NLL_KERNEL(DFN)
NDEBENCH_TRAIN_NLL_KERNEL(DFL)
NDEBENCH_TRAIN_NLL_KERNEL(NSFLinear)
NDEBENCH_TRAIN_NLL_KERNEL(NSFQuadratic)
NDEBENCH_TRAIN_NLL_KERNEL(NSFRQS)
NDEBENCH_TRAIN_NLL_KERNEL(HDF)
