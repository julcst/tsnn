uint grid2bin(uint2 bin, uint dim) {
    return bin.x * dim + bin.y;
}

uint2 bin2grid(uint bin, uint dim) {
    return uint2(bin / dim, bin % dim);
}

/**
 * A histogram is a discrete probability distribution over a fixed number of bins.
 * The bins are indexed from 0 to N-1, where N is the number of bins.
 * Input is in logits and output is log probability.
 */
struct Histogram<uint N> {

    /**
     * Evaluates the log probability of a given bin index.
     * @param logits The unnormalized log probabilities for each bin.
     * @param bin The index of the bin to evaluate.
     * @return The log probability of the specified bin.
     */
    [Differentiable]
    float eval<T : IFloat>(T logWeights[N], uint bin)
    {
        float maxLog = -1.0#INF;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            maxLog = max(maxLog, logWeights[i].toFloat());
        }

        float sumExp = 0.0;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            sumExp += exp(logWeights[i].toFloat() - maxLog);
        }

        return logWeights[bin].toFloat() - maxLog - log(sumExp);
    }

    /**
     * Samples a bin index from the histogram using the provided uniform random variable.
     * @param logits The unnormalized log probabilities for each bin.
     * @param u A uniform random variable in the range [0, 1).
     * @return The sampled bin index.
     */
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