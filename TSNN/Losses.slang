import CoopVec;
import Utils.Math;

interface ILoss<T : __BuiltinFloatingPointType, let K : int>
{
    static CoopVec<T, K> eval(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale);
    static CoopVec<T, K> deriv(CoopVec<T, K> target, CoopVec<T, K> predicted, CoopVec<T, K> scale);
};

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