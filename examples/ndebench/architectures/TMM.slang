import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;
import architectures.IArchitecture;

import TSNN.Mixtures.TruncatedGMM;

struct TMM : IArchitecture
{
    static constexpr const uint K = 16;

    typealias Net = MLP<1, 32, 3, K * 5>;
    typealias Activation = LeakyReLU;
    typealias GMM = TruncatedGMM<K>;

    static constexpr const GMM kGMM = GMM();

    // Base index into the host-supplied `Layout` buffer (see
    // IArchitecture.slang and TSNN.Utils.MLP's static from-layout methods'
    // doc comment): the MLP contributes DEPTH+1 = 4 entries, mirroring
    // benchmark.py's own layout builder for this architecture.
    static constexpr const uint kBase = 0u;

    static uint getParamCount(no_diff Layout layout) {
        // The benchmark allocator and Adam kernel operate on fp16 element counts.
        return Net.paramCount(layout, kBase);
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, no_diff Layout layout, uint tid, uint threadCount)
    {
        Net.initParams<TMirror, TMaster>(params, master, layout, kBase, tid, threadCount);
    }

    [Differentiable]
    static float[K * 5] forward(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx) {
        return CoopVecToArray(Net.forward<Activation>(weights, layout, kBase, ctx));
    }

    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float2 x) {
        let params = forward(weights, layout, ctx);
        return kGMM.eval(params, x);
    }

    static float2 sample(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, inout PCG32 rand) {
        let params = forward(weights, layout, ctx);
        return kGMM.sample(params, rand);
    }
}
