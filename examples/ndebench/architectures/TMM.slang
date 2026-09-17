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

    static constexpr const Net kMLP = Net();
    static constexpr const GMM kGMM = GMM();

    static uint getParamCount() {
        // `end` is a byte offset in the fp16 parameter layout.  The benchmark
        // allocator and Adam kernel operate on fp16 element counts.
        return kMLP.end / uint(sizeof(half));
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, uint tid, uint threadCount)
    {
        kMLP.initParams<TMirror, TMaster>(params, master, tid, threadCount);
    }

    [Differentiable]
    static float[K * 5] forward(MatrixBiasBuffer weights, Context ctx) {
        return CoopVecToArray(kMLP.forward<Activation>(weights, ctx));
    }

    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, Context ctx, no_diff float2 x) {
        let params = forward(weights, ctx);
        return kGMM.eval(params, x);
    }

    static float2 sample(MatrixBiasBuffer weights, Context ctx, inout PCG32 rand) {
        let params = forward(weights, ctx);
        return kGMM.sample(params, rand);
    }
}
