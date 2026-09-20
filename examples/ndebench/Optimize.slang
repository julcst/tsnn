import TSNN.Optimizers.Adam;
import TSNN.Optimizers.Sanitizer;
import TSNN.Utils.MLP;
import TSNN.Encodings.FeatureBuffer;

import architectures.IArchitecture;
import architectures.TMM;
import architectures.HGGrid;
import architectures.DFN;
import architectures.DFL;
import architectures.NSFLinear;
import architectures.NSFQuadratic;
import architectures.NSFRQS;
import architectures.HDF;

/// float16 gradients written by train, read-only
StructuredBuffer<half4> gParamGrads;

/// float16 mirror for forward/backward compute
RWStructuredBuffer<half4> gParams;
/// float32 authoritative master for Adam
RWStructuredBuffer<float4> gParamsMaster;
/// float32 Adam moments for gParamsMaster
RWStructuredBuffer<float4> gMoments1;
RWStructuredBuffer<float4> gMoments2;

// Byte-address aliases are used only by initialization.  The optimizer uses
// the vector views above, while MLP initialization needs element-stride-aware
// byte-address stores.  Both aliases are bound to the same allocations.
RWByteAddressBuffer gParamsInit;
RWByteAddressBuffer gParamsMasterInit;

#if DEBUG_COUNTERS
// 4 × uint atomic: [0]=non-finite (overflow), [1]=clipped (saturation),
// [2]=exactly-zero (underflow proxy), [3]=elements processed.
RWByteAddressBuffer gDebugCounters;
#endif

cbuffer CB
{
    float gLearningRate;
    float gCurrentStep;
    float gLossScale;
    float gGradClip;
    uint gParamElementCount;
    uint gDispatchThreadCount;
}

// A compute kernel entry point with exactly one scalar `uniform` parameter
// (as opposed to a field inside a named cbuffer) reproducibly corrupts this
// environment's Vulkan device: the dispatch *following* it hangs the GPU and
// takes the device down as Xid 109 (CTX SWITCH TIMEOUT), regardless of which
// kernel or buffers that next dispatch touches. Verified in isolation with a
// standalone shader with no TSNN/ndebench code at all -- this is a
// SlangPy/slang-rhi/driver push-constant-layout bug on this GPU/driver, not
// an application bug. hggrid/image_learn's reset kernels never hit this
// because they already route their dispatch-thread-count through a cbuffer
// (see examples/hggrid/Optimize.cs.slang's `CB.gDispatchThreadCount`); this
// one used a lone `uniform uint threadCount` instead, which is what init_TMM
// used to declare here. `InitCB` below is the fix: it makes the payload a
// named-cbuffer field like every other kernel in this file already uses.
cbuffer InitCB
{
    uint gInitThreadCount;
}

void init<T : IArchitecture>(uint3 tid : SV_DispatchThreadID)
{
    T.initParams<half, float>(gParamsInit, gParamsMasterInit, tid.x, gInitThreadCount);
}

// `optimize` is architecture-independent.  Only reset is specialized because
// parameter initialization belongs to the architecture contract.
// `TMM` becomes `init_TMM`.
#define NDEBENCH_RESET_KERNEL(ARCHITECTURE) \
    [numthreads(256, 1, 1)] \
    void init_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        init<ARCHITECTURE>(tid); \
    }

NDEBENCH_RESET_KERNEL(TMM)
NDEBENCH_RESET_KERNEL(HGGrid)
NDEBENCH_RESET_KERNEL(DFN)
NDEBENCH_RESET_KERNEL(DFL)
NDEBENCH_RESET_KERNEL(NSFLinear)
NDEBENCH_RESET_KERNEL(NSFQuadratic)
NDEBENCH_RESET_KERNEL(NSFRQS)
NDEBENCH_RESET_KERNEL(HDF)

[numthreads(256, 1, 1)]
void optimize(uint3 tid: SV_DispatchThreadID)
{
    var adamConfig = AdamConfig(gLearningRate);
    adamConfig.invLossScale = 1.0f / gLossScale;
    let adam = Adam(gCurrentStep, adamConfig);

#if DEBUG_COUNTERS
    SanitizeDebugCounter dbg = SanitizeDebugCounter();
#endif

    for (uint i = tid.x; i < gParamElementCount / 4; i += gDispatchThreadCount)
    {
        var p = gParamsMaster[i];
        let gRaw = float4(gParamGrads[i]);
#if DEBUG_COUNTERS
        let g = sanitizeDebug(gRaw, gGradClip, dbg);
#else
        let g = sanitize(gRaw, gGradClip);
#endif
        var m1 = gMoments1[i];
        var m2 = gMoments2[i];
        adam.step(p, m1, m2, g);
        gMoments1[i] = m1;
        gMoments2[i] = m2;
        gParamsMaster[i] = p; // update master
        gParams[i] = half4(p); // mirror to float16 for compute
    }

#if DEBUG_COUNTERS
    dbg.flush(gDebugCounters);
#endif
}
