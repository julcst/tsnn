import TSNN.LinearOps;
import TSNN.Utils.PCG32;
import TSNN.CoopVec;

typealias Context = CoopVec<half, 1>;

/// Defines an NDE model
interface IArchitecture
{
    static uint getParamCount();

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params,
        RWByteAddressBuffer master,
        uint tid,
        uint threadCount);

    static Context makeContext()
    {
        float context[1] = { 1.0f };
        return CoopVecFromArray<half>(context);
    }

    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, Context ctx, no_diff float2 x);

    static float2 sample(MatrixBiasBuffer weights, Context ctx, inout PCG32 rand);

    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff float2 x) {
        let ctx = makeContext();
        return evalLogPDF(weights, ctx, x);
    }

    static float2 sample(MatrixBiasBuffer weights, inout PCG32 rand) {
        let ctx = makeContext();
        return sample(weights, ctx, rand);
    }
}
