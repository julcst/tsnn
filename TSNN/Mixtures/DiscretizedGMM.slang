import TSNN.Mixtures.GaussianHelpers;
import TSNN.Mixtures.TruncatedGMM; // Shared logSumExp and sampleSoftmax helpers.
import TSNN.Utils.PCG32;

/**
 * A diagonal GMM discretized onto a uniform GridDim x GridDim grid.
 * N and GridDim must be positive. Each component uses the same five parameters
 * as TruncatedGMM: weight logit, mean.x, mean.y, logSigma.x, logSigma.y.
 * Gaussian tails are absorbed into the boundary bins, without truncation.
 * Bin indices use x * GridDim + y, matching Histogram's grid2bin.
 */
struct DiscretizedGMM<uint N, uint GridDim = 8>
{
    static const uint D = 2;
    static const uint STRIDE = 1 + 2 * D;
    static const uint NUM_BINS = GridDim * GridDim;
    typealias Output = vector<float, D>;

    [Differentiable]
    float logAxisMass(float mean, float logSigma, uint bin)
    {
        if (GridDim == 1) return 0.0f;
        let dim = float(GridDim);
        if (bin == 0)
            return logNormalCDF(1.0f / dim, mean, logSigma);
        if (bin == GridDim - 1)
            return logNormalCDF(mean - float(bin) / dim, 0.0f, logSigma);

        // Map this finite interval to [0,1] to reuse stable log-CDF
        // differences and their analytic backward derivative.
        return logTruncMass(mean * dim - float(bin), logSigma + log(dim));
    }

    /** Log probability mass of a bin, suitable for a categorical loss. */
    [Differentiable]
    float evalBin<T : IFloat>(T input[N * STRIDE], uint bin)
    {
        if (bin >= NUM_BINS) return -1.0#INF;
        uint2 cell = uint2(bin / GridDim, bin % GridDim);
        float logits[N];
        float logWeights[N];
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            let offset = i * STRIDE;
            let logWeight = input[offset].toFloat();
            logits[i] = logWeight
                + logAxisMass(input[offset + 1].toFloat(), input[offset + 3].toFloat(), cell.x)
                + logAxisMass(input[offset + 2].toFloat(), input[offset + 4].toFloat(), cell.y);
            logWeights[i] = logWeight;
        }
        return logSumExp(logits) - logSumExp(logWeights);
    }

    /** Log piecewise-constant density on [0,1]^2; includes inverse bin area. */
    [Differentiable]
    float eval<T : IFloat>(T input[N * STRIDE], no_diff Output x)
    {
        if (any(x < Output(0.0)) || any(x > Output(1.0))) return -1.0#INF;
        uint2 cell = min(uint2(floor(x * float(GridDim))), uint2(GridDim - 1));
        return evalBin(input, cell.x * GridDim + cell.y) + 2.0f * log(float(GridDim));
    }

    uint sampleBin<T : IFloat>(T input[N * STRIDE], inout PCG32 rand)
    {
        float logWeights[N];
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
            logWeights[i] = input[i * STRIDE].toFloat();
        let offset = sampleSoftmax(logWeights, rand.nextFloat()) * STRIDE;

        // Conditional on the component, the two axes are independent.
        uint2 cell;
        [ForceUnroll]
        for (uint d = 0; d < D; ++d)
        {
            float masses[GridDim];
            [ForceUnroll]
            for (uint b = 0; b < GridDim; ++b)
                masses[b] = logAxisMass(input[offset + 1 + d].toFloat(), input[offset + 3 + d].toFloat(), b);
            cell[d] = sampleSoftmax(masses, rand.nextFloat());
        }
        return cell.x * GridDim + cell.y;
    }

    /** Sample uniformly inside a sampled bin, matching eval's density. */
    Output sample<T : IFloat>(T input[N * STRIDE], inout PCG32 rand)
    {
        let bin = sampleBin(input, rand);
        let cell = uint2(bin / GridDim, bin % GridDim);
        return (Output(cell) + Output(rand.nextFloat(), rand.nextFloat())) / float(GridDim);
    }
}
