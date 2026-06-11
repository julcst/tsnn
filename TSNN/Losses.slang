import TSNN.CoopVec;
import TSNN.Utils.Math;

// A loss exposes both the scalar loss value (`eval`, for logging) and the
// gradient of the loss w.r.t. the prediction (`deriv`, fed straight into the
// backward pass).  `scale` is a per-element weight (e.g. 1/batchSize, or a
// loss-scale factor) folded into both.  All losses are element-wise and generic
// over the channel count K, so the same struct works for an RGB image (K=3) or
// any other width; `RelativeL2Luminance` is the one K=3 specialisation.
interface ILoss<T : __BuiltinFloatingPointType, let K : int>
{
    static CoopVec<T, K> eval(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale);
    static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale);
};

// ─── L2 / mean-squared error ──────────────────────────────────────────────────
// eval  = scale·(p − t)²        deriv = scale·2·(p − t)
struct L2<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
{
    static CoopVec<T, K> eval(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale)
    {
        let r = predicted - target;
        return scale * r * r;
    }

    static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale)
    {
        return scale * (predicted - target) * T(2);
    }
};

// ─── L1 / mean-absolute error ─────────────────────────────────────────────────
// eval  = scale·|p − t|         deriv = scale·sign(p − t)
struct L1<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
{
    static CoopVec<T, K> eval(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale)
    {
        return scale * abs(predicted - target);
    }

    static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale)
    {
        return scale * copysign(CoopVec<T, K>(T(1)), predicted - target);
    }
};

// ─── Huber / smooth-L1 ────────────────────────────────────────────────────────
// Quadratic for |r| ≤ δ, linear beyond — L2's stability near 0 with L1's
// robustness to outliers.  δ is fixed at 1 (canonical smooth-L1); rescale your
// residuals if you need a different knee.
//   eval  = scale·( |r|≤δ ? ½r²        : δ(|r| − ½δ) )
//   deriv = scale·( |r|≤δ ? r          : δ·sign(r)   )   = scale·clamp(r, −δ, δ)
struct Huber<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
{
    static CoopVec<T, K> eval(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale)
    {
        let delta = T(1);
        let r = predicted - target;
        CoopVec<T, K> o;
        [ForceUnroll]
        for (int i = 0; i < K; ++i)
        {
            let a = abs(r[i]);
            o[i] = scale[i] * (a <= delta ? T(0.5) * r[i] * r[i]
                                          : delta * (a - T(0.5) * delta));
        }
        return o;
    }

    static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale)
    {
        let delta = T(1);
        let r = predicted - target;
        CoopVec<T, K> o;
        [ForceUnroll]
        for (int i = 0; i < K; ++i)
            o[i] = scale[i] * clamp(r[i], -delta, delta);
        return o;
    }
};

// ─── Relative-L2 (per channel) ────────────────────────────────────────────────
// MSE normalised by the prediction's own magnitude, so bright and dark regions
// contribute comparably (Lehtinen et al.'s relative loss, per channel).
//   eval  = scale·(p − t)² / (p² + ε)     deriv = scale·2(p − t) / (p² + ε)
struct RelativeL2<T : __BuiltinFloatingPointType, let K : int> : ILoss<T, K>
{
    static CoopVec<T, K> eval(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale)
    {
        let r = predicted - target;
        let denom = predicted * predicted + CoopVec<T, K>(T(1e-2));
        return scale * r * r / denom;
    }

    static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale)
    {
        let denom = predicted * predicted + CoopVec<T, K>(T(1e-2));
        return scale * (predicted - target) * T(2) / denom;
    }
};

// ─── Relative-L2 normalised by luminance (RGB only) ───────────────────────────
// Like RelativeL2 but the denominator is a single shared luminance², so the hue
// is not distorted by per-channel normalisation — the standard choice for
// learning HDR radiance.
struct RelativeL2Luminance<T : __BuiltinFloatingPointType> : ILoss<T, 3>
{
    static CoopVec<T, 3> eval(CoopVec<T, 3> target, CoopVec<T, 3> predicted, CoopVec<T, 3> scale)
    {
        let lum = luminance(VectorFromCoopVec(predicted));
        let denom = lum * lum + 1e-2;
        return scale * (predicted - target) * (predicted - target) * T(1 / denom);
    }

    static CoopVec<T, 3> deriv(CoopVec<T, 3> target, CoopVec<T, 3> predicted, CoopVec<T, 3> scale)
    {
        let lum = luminance(VectorFromCoopVec(predicted));
        let denom = lum * lum + 1e-2;
        return scale * (predicted - target) * T(2.f / denom);
    }
};
