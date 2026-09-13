import TSNN.Mixtures.GaussianHelpers;
import TSNN.Utils.PCG32;

// Stable log P(0 <= X <= 1) for X ~ N(mean, sigma^2), sigma = exp(logSigma).
// This normalizer used to come from erf((1-mean)*k)+erf(mean*k) directly (see
// logTruncGaussianPDF below), which underflows to exactly 0 -- log(0) = -inf,
// then NaN gradients -- once the standardized interval [zLo, zHi] sits
// entirely in one tail. This keeps the interior case's cheap erf evaluation
// (both CDF values are O(1) there, no cancellation) but routes both tail
// cases through logNormalCDF and a logsubexp identity so the subtraction of
// two CDF values never happens in linear (collapsible) space: log(a - b) =
// log(a) + log1mexp(log(b) - log(a)) for a >= b > 0, where log(a) and log(b)
// individually stay finite (they cancel the common -z^2/2 term algebraically
// instead of forming exp(-z^2/2) and rounding it to zero first).
[Differentiable]
float logTruncMass(float mean, float logSigma)
{
    let invSigma = exp(-logSigma);
    let zLo = -mean * invSigma;         // (0 - mean) / sigma
    let zHi = (1.0f - mean) * invSigma; // (1 - mean) / sigma; zHi > zLo always

    if (zHi <= 0.0f)
    {
        // Both quantiles in the left tail (mean well above the [0,1] interval).
        let logHi = logNormalCDF(zHi);
        let logLo = logNormalCDF(zLo);
        return logHi + log1mexp(logLo - logHi);
    }
    if (zLo >= 0.0f)
    {
        // Both quantiles in the right tail (mean well below the [0,1] interval):
        // reflect to the survival function SF(z) = Phi(-z), decreasing in z, so
        // mass = SF(zLo) - SF(zHi) with SF(zLo) >= SF(zHi) >= 0.
        let logLo = logNormalCDF(-zLo);
        let logHi = logNormalCDF(-zHi);
        return logLo + log1mexp(logHi - logLo);
    }
    // Interior: mean falls inside (or near) the interval, zLo < 0 < zHi -- both
    // CDF values are O(1) here, so the direct erf difference is well-behaved.
    let invSqrt2 = 0.70710678118654752f;
    return log(0.5f * (erf(zHi * invSqrt2) - erf(zLo * invSqrt2)));
}

[Differentiable]
float logTruncGaussianPDF(float2 x, float2 mean, float2 logSigma)
{
    let inverseSigma = exp(-logSigma);
    let diff = (x - mean) * inverseSigma;
    let exponent = 0.5 * dot(diff, diff);

    float logMass = 0.0;
    [ForceUnroll]
    for (uint d = 0; d < 2; ++d)
        logMass += logTruncMass(mean[d], logSigma[d]);

    return -log(2.0f * float.getPi()) - exponent - logSigma.x - logSigma.y - logMass;
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
