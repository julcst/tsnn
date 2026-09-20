import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;
import architectures.IArchitecture;

import TSNN.Mixtures.Histogram;
import TSNN.Mixtures.TruncatedGMM;
import TSNN.Encodings.OneHot;
import TSNN.Encodings.Compose;

/// HGGrid: mlp -> histogram -> sample bin -> encode bin -> mlp -> tgmm
struct HGGrid : IArchitecture
{
    static constexpr const uint kGridDim    = 8;
    static constexpr const uint kNumCells   = kGridDim * kGridDim;
    static constexpr const uint kEncodedDim = 2u * kGridDim; // one-hot(cell.x) ++ one-hot(cell.y)
    static constexpr const uint kContextDim = 1u; // must match IArchitecture.Context's CoopVec width
    static constexpr const uint K = 4; // Gaussians per cell

    typealias CoarseNet = MLP<kContextDim, 32, 3, kNumCells>;
    // Context ++ one-hot(cell)
    typealias FineNet   = MLP<kContextDim + kEncodedDim, 32, 3, K * 5>;
    typealias Activation = LeakyReLU;
    typealias GMM = TruncatedGMM<K>;

    static constexpr const GMM kGMM = GMM();

    // Base indices into the host-supplied `Layout` buffer (see
    // IArchitecture.slang and TSNN.Utils.MLP's `__init(Layout, uint)`
    // overload's doc comment): each MLP contributes DEPTH+1 = 4 entries,
    // chained in declaration order, mirroring benchmark.py's own layout
    // builder for this architecture.
    static constexpr const uint kCoarseBase = 0u;
    static constexpr const uint kFineBase = kCoarseBase + 4u;

    static CoarseNet makeCoarseNet(no_diff Layout layout) { return CoarseNet(layout, kCoarseBase); }
    static FineNet makeFineNet(no_diff Layout layout) { return FineNet(layout, kFineBase); }

    static uint getParamCount(no_diff Layout layout) {
        // `end` is a byte offset in the fp16 parameter layout; both MLPs are
        // chained into one buffer, so the fine MLP's end covers both blocks.
        return makeFineNet(layout).end / uint(sizeof(half));
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, no_diff Layout layout, uint tid, uint threadCount)
    {
        makeCoarseNet(layout).initParams<TMirror, TMaster>(params, master, tid, threadCount);
        makeFineNet(layout).initParams<TMirror, TMaster>(params, master, tid, threadCount);
    }

    static half[kEncodedDim] encodeCell(no_diff uint2 cell) {
        let ex = oneHot<kGridDim>(cell.x);
        let ey = oneHot<kGridDim>(cell.y);

        half enc[kEncodedDim];
        [ForceUnroll]
        for (uint i = 0; i < kGridDim; ++i)
        {
            enc[i]            = half(ex[i]);
            enc[kGridDim + i] = half(ey[i]);
        }
        return enc;
    }

    [Differentiable]
    static float[kNumCells] forwardCoarse(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx) {
        return CoopVecToArray(makeCoarseNet(layout).forward<Activation>(weights, ctx));
    }

    [Differentiable]
    static float[K * 5] forwardFine(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff uint2 cell) {
        let input = ComposeCoopVecPlus(ctx, encodeCell(cell));
        let params = CoopVecToArray(makeFineNet(layout).forward<Activation>(weights, input));
        // Clip logSigma?
        return params;
    }

    // log p(x) = logHist(bin) + logFine(t) + 2*log(kGridDim): the Jacobian of
    // (bin, t) -> x, mapping the unit square into a kGridDim x kGridDim grid
    // of unit cells, scales area by 1/kGridDim^2.
    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float2 x) {
        float2 xs = x * float(kGridDim);
        uint2 cell = clamp(uint2(floor(xs)), uint2(0u), uint2(kGridDim - 1u));
        float2 t = xs - float2(cell);
        uint bin = grid2bin(cell, kGridDim);

        Histogram<kNumCells> hist;
        let histLogits = forwardCoarse(weights, layout, ctx);
        float logHist = hist.eval(histLogits, bin);

        let fineParams = forwardFine(weights, layout, ctx, cell);
        float logFine = kGMM.eval(fineParams, t);

        return logHist + logFine + 2.0f * log(float(kGridDim));
    }

    static float2 sample(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, inout PCG32 rand) {
        Histogram<kNumCells> hist;
        let histLogits = forwardCoarse(weights, layout, ctx);
        uint bin = hist.sample(histLogits, rand.nextFloat());
        uint2 cell = bin2grid(bin, kGridDim);

        let fineParams = forwardFine(weights, layout, ctx, cell);
        float2 t = kGMM.sample(fineParams, rand);

        return (float2(cell) + t) / float(kGridDim);
    }
}
