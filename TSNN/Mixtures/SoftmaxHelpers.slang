

// TODO: Maybe manual analytic derivative would be better
[Differentiable]
float logSumExp<T : IFloat, uint N>(T input[N])
{
    float maxVal = -1.0#INF;
    [ForceUnroll]
    for (uint i = 0; i < N; ++i)
    {
        maxVal = max(maxVal, input[i].toFloat());
    }

    float sumExp = 0.0;
    [ForceUnroll]
    for (uint i = 0; i < N; ++i)
    {
        sumExp += exp(input[i].toFloat() - maxVal);
    }

    return maxVal + log(sumExp);
}

float[N] stableSoftmax<T : IFloat, uint N>(T input[N])
{
    float maxVal = -1.0#INF;
    [ForceUnroll]
    for (uint i = 0; i < N; ++i)
    {
        maxVal = max(maxVal, input[i].toFloat());
    }

    float output[N];
    float sumExp = 0.0;
    [ForceUnroll]
    for (uint i = 0; i < N; ++i)
    {
        output[i] = exp(input[i].toFloat() - maxVal);
        sumExp += output[i];
    }

    float invSumExp = 1.0 / sumExp;
    [ForceUnroll]
    for (uint i = 0; i < N; ++i)
    {
        output[i] *= invSumExp;
    }
    return output;
}

uint sampleSoftmax<T : IFloat, uint N>(T input[N], float u)
{
    float maxVal = -1.0#INF;
    [ForceUnroll]
    for (uint i = 0; i < N; ++i)
    {
        maxVal = max(maxVal, input[i].toFloat());
    }

    float output[N];
    float sumExp = 0.0;
    [ForceUnroll]
    for (uint i = 0; i < N; ++i)
    {
        output[i] = exp(input[i].toFloat() - maxVal);
        sumExp += output[i];
    }

    var r = u * sumExp;
    for (uint i = 0; i < N; ++i)
    {
        r -= output[i];
        if (r < 0.0) return i;
    }
    return N - 1;
}