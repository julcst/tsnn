// examples/nde/Optimize.cs.slang
// Adam optimiser kernel for the NDE example.
//
// resetMain  – initialises parameters and zero-fills moment buffers.
//   · Hidden-layer weights: small Xavier random  (±√(2/kHidden))
//   · Output-layer weights: zero  (flow starts near-identity)
//   · Width/height biases: zero   (uniform-bin softmax)
//   · Derivative biases: kRQSBoundaryUnnorm ≈ 0.5413
//     (ensures initial knot derivatives ≈ 1, i.e. identity slope)
//
// optimizeMain – one Adam step; vectorised float4/half4 reads for bandwidth.

import TSNN.Optimizers.Adam;
import TSNN.Utils.PCG32;
import TSNN.Utils.RWBuffer;
import TSNN.Utils.Math;
import TSNN.Flows.RQS;
import Network;

RWByteAddressBuffer gParams;
RWByteAddressBuffer gParamGrads;
RWByteAddressBuffer gMoments1;
RWByteAddressBuffer gMoments2;

cbuffer CB
{
    float gLearningRate;
    float gCurrentStep;
    uint  gDispatchThreadCount;
}

[numthreads(256, 1, 1)]
void resetMain(uint3 tid : SV_DispatchThreadID)
{
    PCG32 rng = PCG32(uint64_t(tid.x), uint64_t(tid.x) ^ 0x9e3779b97f4a7c15ULL);

    float hiddenScale = sqrt(2.0f / float(kHidden));

    for (uint i = tid.x; i < kParamElementCount; i += gDispatchThreadCount)
    {
        uint byteOff  = i * sizeof(half);

        // Determine position within the repeating MLP block
        uint localByte = byteOff % kMLPBlockBytes;

        float val;
        if (localByte >= kOutputWeightStart && localByte < kOutputWeightEnd)
        {
            // Output-layer weights → zero (near-identity flow at start)
            val = 0.0f;
        }
        else if (localByte >= kDerivBiasStart && localByte < kDerivBiasEnd)
        {
            // Derivative biases → softplus_inv(1 - kRQSMinDerivative)
            // so initial knot derivatives ≈ 1.0
            val = kRQSBoundaryUnnorm;
        }
        else
        {
            // Hidden-layer weights and all other biases: small Xavier random
            val = (rng.nextFloat() * 2.0f - 1.0f) * hiddenScale;
        }

        gParams.StoreAligned<half>(byteOff, half(val));
        gMoments1.StoreAligned<float>(i * sizeof(float), 0.0f);
        gMoments2.StoreAligned<float>(i * sizeof(float), 0.0f);
    }
}

[numthreads(256, 1, 1)]
void optimizeMain(uint3 tid : SV_DispatchThreadID)
{
    let adamConfig = AdamConfig(gLearningRate);
    let adam       = Adam(gCurrentStep, adamConfig);

    for (uint base = tid.x * 4u; base < kParamElementCount; base += gDispatchThreadCount * 4u)
    {
        var p  = LoadHalf4(gParams,    base);
        // Clamp gradient elements to ±1 to suppress rare spikes from sharp RQS bins
        let g  = clamp(sanitize(LoadHalf4(gParamGrads, base)), float4(-1.0f), float4(1.0f));
        var m1 = LoadFloat4(gMoments1, base);
        var m2 = LoadFloat4(gMoments2, base);
        adam.step(p, m1, m2, g);
        StoreFloat4(gMoments1, base, m1);
        StoreFloat4(gMoments2, base, m2);
        StoreHalf4(gParams,    base, p);
    }
}
