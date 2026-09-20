import TSNN.LinearOps;
import TSNN.Utils.PCG32;
import TSNN.CoopVec;

typealias Context = CoopVec<half, 1>;

// Per-layer (weightOffset, biasOffset) byte pairs for every MLP block an
// architecture owns, packed back-to-back in MLP-declaration order (each MLP
// contributing DEPTH+1 entries). Computed host-side via
// Device.get_coop_vec_matrix_size for TrainingOptimal weight matrices --
// see TSNN.Utils.MLP's `__init(StructuredBuffer<uint2>, uint)` overload's
// doc comment for why this can't be a Slang-side compile-time constant.
typealias Layout = StructuredBuffer<uint2>;

/// Defines an NDE model
interface IArchitecture
{
    static uint getParamCount(no_diff Layout layout);

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params,
        RWByteAddressBuffer master,
        no_diff Layout layout,
        uint tid,
        uint threadCount);

    static Context makeContext()
    {
        float context[1] = { 1.0f };
        return CoopVecFromArray<half>(context);
    }

    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float2 x);

    static float2 sample(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, inout PCG32 rand);

    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff Layout layout, no_diff float2 x) {
        let ctx = makeContext();
        return evalLogPDF(weights, layout, ctx, x);
    }

    static float2 sample(MatrixBiasBuffer weights, no_diff Layout layout, inout PCG32 rand) {
        let ctx = makeContext();
        return sample(weights, layout, ctx, rand);
    }
}
