// Infer.cs.slang - Inference kernel for the image-learn example.
// Evaluates the MLP at every pixel and writes to a float texture.

import TSNN.CoopVec;
import MLP;

ByteAddressBuffer gParams;
ByteAddressBuffer gEncodingParams;
RWTexture2D<float4> gOutput;

[numthreads(8, 8, 1)]
void inferMain(uint3 tid: SV_DispatchThreadID, uniform uint2 frameDim)
{
    uint2 pixel = tid.xy;
    if (any(pixel >= frameDim))
        return;

    float2 uv = (float2(pixel) + 0.5f) / float2(frameDim);
    let predVec = forward(gEncodingParams, gParams, uv);
    let pred = VectorFromCoopVec(predVec);
    // let pred = float3(1.0f, 0.0f, 0.0f);
    gOutput[pixel] = float4(pred, 1.0f);
}
