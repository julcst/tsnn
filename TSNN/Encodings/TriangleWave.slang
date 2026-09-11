/// Triangle-wave encoding (Figueiredo et al. 2025, "Neural Path Guiding with Distribution
/// Factorization", the conditioning-coordinate encoding for their second/conditional PDF
/// network).
///
/// F octaves of a unit-amplitude triangle wave at doubling frequency, sampled at the scalar
/// s in [0,1]:
///     level l:  tri(2^l * s),   tri(z) = 2 * |z - floor(z + 0.5)|  in [0,1]
/// tri(z) is the 1-periodic triangle wave with tri(0) = 0, tri(0.5) = 1 -- unlike a sine
/// encoding it is C0 but not C1, which is what lets a ReLU MLP fit it exactly. Output is
/// left un-normalised, matching TSNN's `Frequency`/`OneBlob` convention: the downstream MLP
/// learns the scale. Non-differentiable: the encoded scalar (here, the marginal draw fed to
/// the conditional network) is a constant input, never optimised through.

import TSNN.CoopVec;

CoopVec<T, F> encodeTriangleWave<T : __BuiltinFloatingPointType, let F : int>(float s)
{
    float arr[F];
    [ForceUnroll]
    for (int l = 0; l < F; ++l)
    {
        float z = float(1u << uint(l)) * s;
        arr[l] = 2.0f * abs(z - floor(z + 0.5f));
    }
    return CoopVecFromArray<T, F>(arr);
}
