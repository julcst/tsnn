import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;

import TSNN.Mixtures.TruncatedGMM;
namespace TMM
{
    static constexpr const uint K = 8;

    typealias Net = MLP<1, 32, 3, K * 5>;
    typealias Activation = LeakyReLU;
    typealias GMM = TruncatedGMM<K>;

    static constexpr const Net kMLP = Net();
    static constexpr const GMM kGMM = GMM();

    uint getParamCount() {
        // `end` is a byte offset in the fp16 parameter layout.  The benchmark
        // allocator and Adam kernel operate on fp16 element counts.
        return kMLP.end / uint(sizeof(half));
    }

    void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, uint tid, uint threadCount)
    {
        kMLP.initParams<TMirror, TMaster>(params, master, tid, threadCount);
    }

    CoopVec<half, 1> makeContext()
    {
        float context[1] = { 1.0f };
        return CoopVecFromArray<half>(context);
    }

    [Differentiable]
    float[K * 5] forward(MatrixBiasBuffer weights) {
        let ctx = makeContext();
        return CoopVecToArray(kMLP.forward<Activation>(weights, ctx));
    }

    [Differentiable]
    float evalLogPDF(MatrixBiasBuffer weights, no_diff float2 x) {
        let params = forward(weights);
        return kGMM.eval(params, x);
    }

    float2 sample(MatrixBiasBuffer weights, inout PCG32 rand) {
        let params = forward(weights);
        return kGMM.sample(params, rand);
    }
}
