import TSNN.Mixtures.DiscretizedGMM;
import TSNN.Utils.PCG32;

StructuredBuffer<float> gParams;
RWStructuredBuffer<float> gOutputs;

[Differentiable]
float loss(float params[10], uint bin)
{
    DiscretizedGMM<2> model;
    return -model.evalBin(params, bin);
}

[numthreads(64, 1, 1)]
void evalMain(uint3 tid : SV_DispatchThreadID)
{
    uint bin = tid.x;
    if (bin >= 64) return;
    float params[10];
    for (uint i = 0; i < 10; ++i) params[i] = gParams[i];
    DiscretizedGMM<2> model;
    gOutputs[bin] = exp(model.evalBin(params, bin));
    float2 x = (float2(bin / 8, bin % 8) + 0.5f) / 8.0f;
    gOutputs[64 + bin] = exp(model.eval(params, x));
    var dp = diffPair(params);
    bwd_diff(loss)(dp, bin, 1.0f);
    for (uint j = 0; j < 10; ++j) gOutputs[128 + bin * 10 + j] = dp.d[j];
    DiscretizedGMM<2, 1> single;
    gOutputs[768 + bin] = single.evalBin(params, 0);
}

[numthreads(64, 1, 1)]
void sampleMain(uint3 tid : SV_DispatchThreadID)
{
    float params[10];
    for (uint i = 0; i < 10; ++i) params[i] = gParams[i];
    PCG32 rng = PCG32(tid.x, 42);
    DiscretizedGMM<2> model;
    float2 x = model.sample(params, rng);
    gOutputs[2 * tid.x] = x.x;
    gOutputs[2 * tid.x + 1] = x.y;
}
