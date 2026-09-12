import TSNN.Mixtures.GaussianHelpers;
import TSNN.Utils.PCG32;

[Differentiable]
float logTruncGaussianPDF(float2 x, float2 mean, float2 logSigma)
{
    let inverseSigma = exp(-logSigma);

    var area = 1.0;
    [ForceUnroll]
    for (uint d = 0; d < 2; ++d)
    {
        let k = inverseSigma[d] / sqrt(2.0);
        area *= erf((1.0 - mean[d]) * k) + erf(mean[d] * k);
    }

    let diff = (x - mean) * inverseSigma;
    let exponent = 0.5 * dot(diff, diff);
    return log(2.0 / float.getPi()) - exponent - logSigma.x - logSigma.y - log(area);
}

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
        float weights[N];
        float totalWeight = 0.0;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            let w = exp(input[i * STRIDE + 0].toFloat()); // weight
            weights[i] = w;
            totalWeight += w;
        }

        var r = rand.nextFloat() * totalWeight;
        uint componentIndex = 0;
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            r -= weights[i];
            if (r <= 0.0)
            {
                componentIndex = i;
                break;
            }
        }

        // Sample from the selected Gaussian component
        let offset = componentIndex * STRIDE;
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
