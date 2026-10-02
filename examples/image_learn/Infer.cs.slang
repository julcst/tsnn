// Infer.cs.slang - Inference kernel for the image-learn example.
// Evaluates the MLP at every pixel and writes to a float texture.

import TSNN.CoopVec;
import Network;

ByteAddressBuffer gParams;
ByteAddressBuffer gEncodingParams;
RWTexture2D<float4> gOutput;

void infer<let INFERENCE_OPTIMAL : bool>(uint3 tid, uint2 frameDim)
{
    uint2 pixel = tid.xy;
    if (any(pixel >= frameDim))
        return;

    float2 uv = (float2(pixel) + 0.5f) / float2(frameDim);
    let predVec = INFERENCE_OPTIMAL
        ? forwardInference(gEncodingParams, gParams, uv)
        : forward(gEncodingParams, gParams, uv);
    let pred = VectorFromCoopVec(predVec);
    gOutput[pixel] = float4(pred, 1.0f);
}

[numthreads(8, 8, 1)]
void inferMain(uint3 tid: SV_DispatchThreadID, uniform uint2 frameDim)
{
    infer<true>(tid, frameDim);
}

// Reference path for validating and benchmarking conversion on identical weights.
[numthreads(8, 8, 1)]
void inferTrainingMain(uint3 tid: SV_DispatchThreadID, uniform uint2 frameDim)
{
    infer<false>(tid, frameDim);
}
