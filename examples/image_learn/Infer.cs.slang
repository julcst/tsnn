// Infer.cs.slang - Inference kernel for the image-learn example.
// Evaluates the MLP at every pixel and writes to a float texture.

import TSNN.CoopVec;
import MLP;

ByteAddressBuffer    gParams;
ByteAddressBuffer    gEncodingParams;
RWTexture2D<float4>  gOutput;

cbuffer CB { uint2 gFrameDim; }

[numthreads(8, 8, 1)]
void inferMain(uint3 tid : SV_DispatchThreadID)
{
    uint2 pixel = tid.xy;
    if (any(pixel >= gFrameDim)) return;

    float2 uv = (float2(pixel) + 0.5f) / float2(gFrameDim);
    let predVec = forward(gEncodingParams, gParams, uv);
    let pred = VectorFromCoopVec(predVec);
    gOutput[pixel] = float4(saturate(pred), 1.0f);
}
