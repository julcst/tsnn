import TSNN.Mixtures.GaussianHelpers;
import TSNN.Mixtures.SoftmaxHelpers;
import TSNN.Utils.PCG32;

/**
* Truncated Gaussian Mixture Model (GMM) implementation.
* This model represents a mixture of N Gaussian components, each parametrized by its log weight, mean, and log sigma, truncated to [0, 1]^2.
*/
struct TruncatedGMM<uint N>
{
    static const uint D = 2; // Dimensionality of the Gaussian components
    static const uint STRIDE = 1 + 2 * D; // weight + mean + variance for each component
    typealias Output = vector<float, D>;

    // logSigma is an unconstrained MLP output; clamping keeps sigma =
    // exp(logSigma) in [exp(-6), exp(6)] ~= [0.0025, 403] -- generous for a
    // sub-cell distribution over [0,1]^2, but keeps z = (x-mean)/sigma well
    // away from the float32-overflow regime an unconstrained head eventually
    // drifts into (logTruncMass/logTruncGaussianPDF's z*z and erfcxPositive's
    // fitted input range both degrade to inf/NaN there). Fixes this file's own
    // former "TODO: test logSigma floor" -- see FINDING.md's TruncatedGMM
    // logSigma-floor entry for the training divergence this was causing.
    static const float kLogSigmaMax = 6.0f;

    /**
    * Evaluates the log-probability density function (PDF) of the truncated GMM at a given sample point.
    * @param input An array containing the weights, means, and sigmas
    * @return log pdf
    */
    [Differentiable]
    float eval<T : IFloat>(T input[N * STRIDE], Output x)
    {
        if (any(x < Output(0.0)) || any(x > Output(1.0))) return -1.0#INF; // Sample is outside the unit square, return zero probability

        float logits[N];
        float logWeights[N];

        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            let offset = i * STRIDE;
            let logWeight = input[offset + 0].toFloat(); // normalized weight
            let mean = Output(input[offset + 1].toFloat(), input[offset + 2].toFloat()); // uncontrained mean
            let logSigma = Output(
                clamp(input[offset + 3].toFloat(), -kLogSigmaMax, kLogSigmaMax),
                clamp(input[offset + 4].toFloat(), -kLogSigmaMax, kLogSigmaMax));

            logits[i] = logWeight + logTruncGaussianPDF(x, mean, logSigma);
            logWeights[i] = logWeight;
        }

        return logSumExp(logits) - logSumExp(logWeights); // Normalize by the sum of weights
    }

    /**
    * Closed-form log probability MASS of this truncated GMM inside an
    * axis-aligned box [lo, hi] subset of [0,1]^2 (e.g. one cell of a
    * discretization grid) -- the continuous-GMM analog of a learned
    * histogram's per-bin log-probability, but with a parameter count that
    * doesn't scale with the grid resolution: only the box's own boundaries
    * change per cell, not this GMM's K components. Per component, the mass
    * factorizes into independent per-axis interval masses (the joint
    * Gaussian's covariance is diagonal) divided by that same component's own
    * [0,1] truncation mass (logTruncMass) -- exactly logTruncGaussianPDF's
    * own normalization term, just replacing the pointwise Gaussian PDF/
    * exponent with GaussianHelpers.logIntervalMass's own box-mass term.
    * @param input An array containing the weights, means, and sigmas
    * @param lo Box lower corner, component-wise >= 0
    * @param hi Box upper corner, component-wise <= 1 and > lo
    * @return log P(lo <= X <= hi)
    */
    [Differentiable]
    float evalCellMass<T : IFloat>(T input[N * STRIDE], no_diff Output lo, no_diff Output hi)
    {
        float logits[N];
        float logWeights[N];

        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            let offset = i * STRIDE;
            let logWeight = input[offset + 0].toFloat();
            let mean = Output(input[offset + 1].toFloat(), input[offset + 2].toFloat());
            let logSigma = Output(
                clamp(input[offset + 3].toFloat(), -kLogSigmaMax, kLogSigmaMax),
                clamp(input[offset + 4].toFloat(), -kLogSigmaMax, kLogSigmaMax));

            float logMass = 0.0f;
            [ForceUnroll]
            for (uint d = 0; d < D; ++d)
                logMass += logIntervalMass(mean[d], logSigma[d], lo[d], hi[d]) - logTruncMass(mean[d], logSigma[d]);

            logits[i] = logWeight + logMass;
            logWeights[i] = logWeight;
        }

        return logSumExp(logits) - logSumExp(logWeights);
    }

    Output sample<T : IFloat>(T input[N * STRIDE], inout PCG32 rand)
    {
        float[N] logWeights;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            logWeights[i] = input[i * STRIDE].toFloat();
        }
        let bin = sampleSoftmax(logWeights, rand.nextFloat());

        // Sample from the selected Gaussian component
        let offset = bin * STRIDE;
        let mean = Output(input[offset + 1].toFloat(), input[offset + 2].toFloat());
        let logSigma = Output(
            clamp(input[offset + 3].toFloat(), -kLogSigmaMax, kLogSigmaMax),
            clamp(input[offset + 4].toFloat(), -kLogSigmaMax, kLogSigmaMax));

        // Sample from the truncated Gaussian
        Output sample;
        [ForceUnroll]
        for (uint d = 0; d < D; ++d)
        {
            let cdfMin = normalCDF(0.0, mean[d], logSigma[d]);
            let cdfMax = normalCDF(1.0, mean[d], logSigma[d]);
            let u = cdfMin + rand.nextFloat() * (cdfMax - cdfMin); // Map u to the truncated CDF range
            sample[d] = normalCDFInverse(u, mean[d], logSigma[d]);
        }

        return sample;
    }

    // Deterministic counterpart of sample(): takes explicit uniform values
    // instead of drawing from `inout PCG32 rand`, so a caller can ask for a
    // reproducible "typical" draw (e.g. uComponent=uAxis=0.5 -> the
    // highest-cumulative-probability-quantile component's own median point)
    // instead of a random realization. Used by HGGrid's layers_viz
    // instrumentation (architectures/HGGrid.slang's sampleTrajectoryFixed).
    Output sampleAt<T : IFloat>(T input[N * STRIDE], float uComponent, Output uAxis)
    {
        float[N] logWeights;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            logWeights[i] = input[i * STRIDE].toFloat();
        }
        let bin = sampleSoftmax(logWeights, uComponent);

        let offset = bin * STRIDE;
        let mean = Output(input[offset + 1].toFloat(), input[offset + 2].toFloat());
        let logSigma = Output(
            clamp(input[offset + 3].toFloat(), -kLogSigmaMax, kLogSigmaMax),
            clamp(input[offset + 4].toFloat(), -kLogSigmaMax, kLogSigmaMax));

        Output sample;
        [ForceUnroll]
        for (uint d = 0; d < D; ++d)
        {
            let cdfMin = normalCDF(0.0, mean[d], logSigma[d]);
            let cdfMax = normalCDF(1.0, mean[d], logSigma[d]);
            let u = cdfMin + uAxis[d] * (cdfMax - cdfMin);
            sample[d] = normalCDFInverse(u, mean[d], logSigma[d]);
        }
        return sample;
    }
}
