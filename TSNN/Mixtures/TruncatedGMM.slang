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
            let logSigma = Output(input[offset + 3].toFloat(), input[offset + 4].toFloat()); // positive sigma
            logits[i] = logWeight + logTruncGaussianPDF(x, mean, logSigma);
            logWeights[i] = logWeight;
        }

        return logSumExp(logits) - logSumExp(logWeights); // Normalize by the sum of weights
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
        let logSigma = Output(input[offset + 3].toFloat(), input[offset + 4].toFloat());

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
}
