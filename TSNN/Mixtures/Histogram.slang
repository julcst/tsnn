struct Histogram<uint N> {

    [Differentiable]
    float eval<T : IFloat>(T logWeights[N], uint bin)
    {
        float maxLog = -1.0#INF;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            maxLog = no_diff max(maxLog, logWeights[i].toFloat());
        }

        float output[N];
        float sumExp = 0.0;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            output[i] = exp(logWeights[i].toFloat() - maxLog);
            sumExp += output[i];
        }

        return output[bin] / sumExp;
    }

    uint sample<T : IFloat>(T logWeights[N], float u)
    {
        float maxLog = -1.0#INF;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            maxLog = max(maxLog, logWeights[i].toFloat());
        }

        float output[N];
        float sumExp = 0.0;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            output[i] = exp(logWeights[i].toFloat() - maxLog);
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
}