import TSNN.LinearOps;
import TSNN.Utils.PCG32;

/// Defines an NDE model
interface IArchitecture
{
    static uint getParamCount();

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params,
        RWByteAddressBuffer master,
        uint tid,
        uint threadCount);

    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff float2 x);

    static float2 sample(MatrixBiasBuffer weights, inout PCG32 rand);
}
