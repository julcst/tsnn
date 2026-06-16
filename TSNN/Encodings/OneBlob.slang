/// One-blob encoding (Müller et al. 2019, "Neural Importance Sampling", §one-blob).
///
/// A generalization of one-hot encoding: a Gaussian kernel (σ = 1/K) is placed at
/// the scalar s ∈ [0,1] and discretized into K equal bins. Unlike one-hot, the
/// encoding is lossless — it captures the exact position of s — and lets the
/// downstream ReLU network specialise on sub-domains of the input, which sharpens
/// piecewise-polynomial coupling transforms substantially.
///
/// The kernel is sampled at each bin centre c_i = (i + 0.5)/K. Output is left
/// un-normalised (the network learns the scale); this matches the common
/// tiny-cuda-nn convention. Non-differentiable: the encoded scalar is a constant
/// input to the conditioner network.

import TSNN.CoopVec;

CoopVec<T, K> encodeOneBlob<T : __BuiltinFloatingPointType, let K : int>(float s)
{
    const float invK  = 1.0f / float(K);
    const float sigma = invK;
    // 1 / (2 σ²), folded into the Gaussian exponent.
    const float invTwoSigma2 = 0.5f / (sigma * sigma);

    float arr[K];
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        float center = (float(i) + 0.5f) * invK;
        float d      = center - s;
        arr[i]       = exp(-d * d * invTwoSigma2);
    }
    return CoopVecFromArray<T, K>(arr);
}
