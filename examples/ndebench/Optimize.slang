import TSNN.Optimizers.Adam;
import TSNN.Optimizers.Sanitizer;
import TSNN.Utils.MLP;
import TSNN.Encodings.FeatureBuffer;

import architectures.IArchitecture;
#include "Configurations.slangh"

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
StructuredBuffer<uint2> gLayout;

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

// Keep scalar dispatch arguments in named cbuffers. A lone scalar `uniform`
// entry-point argument hangs the following Vulkan dispatch on this driver.
cbuffer InitCB
{
    uint gInitThreadCount;
}

void init<T : IArchitecture>(uint3 tid : SV_DispatchThreadID)
{
    T.initParams<half, float>(gParamsInit, gParamsMasterInit, gLayout, tid.x, gInitThreadCount);
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

#define NDEBENCH_MODEL(NAME) NDEBENCH_RESET_KERNEL(NAME)
#define NDEBENCH_GRID(NAME, G, L, H, D, K) NDEBENCH_MODEL(NAME)
#include "Models.slangh"
#undef NDEBENCH_GRID
#undef NDEBENCH_MODEL
