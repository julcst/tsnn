uint grid2bin(uint2 bin, uint dim) {
    return bin.x * dim + bin.y;
}

uint2 bin2grid(uint bin, uint dim) {
    return uint2(bin / dim, bin % dim);
}

// Clamp raw (pre-softmax) logits to a bounded range before any softmax math,
// mirroring TSNN.Flows.RQS's kRQSMaxRaw clamp on its own raw conditioner
// output -- Histogram had no analog. This is a real, observed failure mode,
// not a hypothetical: a runaway conditioner MLP can (and, on the DF-N
// benchmark architecture, does) emit a literal +-inf logit. Once any logit
// is +-inf, `maxLog - logWeights[i]` produces inf-inf = NaN, and NaN then
// silently reduces to a finite (but meaningless) value through every
// `max(x, 1e-30f)`-style guard in this codebase, confirmed empirically on
// this Slang/Vulkan backend's `max`: `max(NaN, y) == y` whenever NaN is the
// FIRST argument (the second argument wins), so `log(max(p, 1e-30f))`
// quietly becomes log(1e-30f) = -69.08 instead of NaN or a crash. That let a
// genuinely diverged model keep training and reporting a plausible-looking
// but bogus NLL for thousands of further steps instead of surfacing the real
// bug (see FINDING.md). Clamping the logits up front prevents inf/NaN from
// ever entering this struct's math in the first place -- simpler and more
// robust than flooring the output probability after the fact (the previous
// version of this fix), and it stays entirely in log space: no extra exp/
// log/division round trip.
static const float kHistMaxLogit = 15.0f; // matches TSNN.Flows.RQS's kRQSMaxRaw

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
        float w[N];
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            w[i] = clamp(logWeights[i].toFloat(), -kHistMaxLogit, kHistMaxLogit);
        }

        float maxLog = -1.0#INF;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            maxLog = max(maxLog, w[i]);
        }

        float sumExp = 0.0;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            sumExp += exp(w[i] - maxLog);
        }

        return w[bin] - maxLog - log(sumExp);
    }

    /**
     * Samples a bin index from the histogram using the provided uniform random variable.
     * @param logits The unnormalized log probabilities for each bin.
     * @param u A uniform random variable in the range [0, 1).
     * @return The sampled bin index.
     */
    uint sample<T : IFloat>(T logWeights[N], float u)
    {
        float w[N];
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            w[i] = clamp(logWeights[i].toFloat(), -kHistMaxLogit, kHistMaxLogit);
        }

        float maxLog = -1.0#INF;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            maxLog = max(maxLog, w[i]);
        }

        float output[N];
        float sumExp = 0.0;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            output[i] = exp(w[i] - maxLog);
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