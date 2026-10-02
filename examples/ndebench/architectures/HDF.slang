import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;
import architectures.IArchitecture;

import TSNN.Mixtures.Histogram;
import TSNN.Encodings.OneHot;
import TSNN.Encodings.Compose;

/// HDF: Hierarchical Distribution Factorization (user's lab proposal,
/// ~/Developer/lab-neural-importance-sampling/tex/content.tex, SS"Hierarchical
/// Distribution Factorization (HDF)"): p(xi|ctx) = p_C(C|ctx) * p_F(F|C,ctx) *
/// U(xi|C,F), C,F in {0..7}^2.  Mirrors HGGrid's coarse/fine two-MLP structure
/// and its encodeCell/grid2bin/bin2grid helpers, but the fine stage is a second
/// Histogram<64> (categorical) instead of a TruncatedGMM, plus a final
/// uniform-jitter sample within the chosen fine cell.
struct HDF : IArchitecture
{
    static constexpr const uint kCoarseDim       = 8;
    static constexpr const uint kFineDim         = 8;
    static constexpr const uint kNumCoarseCells  = kCoarseDim * kCoarseDim; // 64
    static constexpr const uint kNumFineCells    = kFineDim * kFineDim;     // 64
    static constexpr const uint kEncodedDim      = 2u * kCoarseDim;         // one-hot(cell.x) ++ one-hot(cell.y)
    static constexpr const uint kContextDim      = 1u; // must match IArchitecture.Context's CoopVec width
    static constexpr const uint kFinestRes       = kCoarseDim * kFineDim;   // 64: per-axis resolution at the finest level

    typealias CoarseNet = MLP<kContextDim, 32, 3, kNumCoarseCells>;
    // Context ++ one-hot(coarse cell)
    typealias FineNet   = MLP<kContextDim + kEncodedDim, 32, 3, kNumFineCells>;
    typealias Activation = LeakyReLU;

    // Base indices into the host-supplied `Layout` buffer (see
    // IArchitecture.slang and TSNN.Utils.MLP's static from-layout methods'
    // doc comment): each MLP contributes DEPTH+1 = 4 entries, chained in
    // declaration order, mirroring benchmark.py's own layout builder for
    // this architecture.
    static constexpr const uint kCoarseBase = 0u;
    static constexpr const uint kFineBase = kCoarseBase + 4u;

    static uint getParamCount(no_diff Layout layout) {
        // FineNet's end covers both blocks: both MLPs are chained into one buffer.
        return FineNet.paramCount(layout, kFineBase);
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, no_diff Layout layout, uint tid, uint threadCount)
    {
        CoarseNet.initParams<TMirror, TMaster>(params, master, layout, kCoarseBase, tid, threadCount);
        FineNet.initParams<TMirror, TMaster>(params, master, layout, kFineBase, tid, threadCount);
    }

    static half[kEncodedDim] encodeCell(no_diff uint2 cell) {
        let ex = oneHot<kCoarseDim>(cell.x);
        let ey = oneHot<kCoarseDim>(cell.y);

        half enc[kEncodedDim];
        [ForceUnroll]
        for (uint i = 0; i < kCoarseDim; ++i)
        {
            enc[i]              = half(ex[i]);
            enc[kCoarseDim + i] = half(ey[i]);
        }
        return enc;
    }

    [Differentiable]
    static float[kNumCoarseCells] forwardCoarse(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx) {
        return CoopVecToArray(CoarseNet.forwardStatic<NDEOffsets, Activation>(weights, ctx, kCoarseBase));
    }

    [Differentiable]
    static float[kNumFineCells] forwardFine(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff uint2 coarseCell) {
        let input = ComposeCoopVecPlus(ctx, encodeCell(coarseCell));
        let params = CoopVecToArray(FineNet.forwardStatic<NDEOffsets, Activation>(weights, input, kFineBase));
        return params;
    }

    // log p(x) = logCoarseHist(binC) + logFineHist(binF) + 2*log(kFinestRes):
    // U(x|C,F) is uniform over a (1/kFinestRes)^2 fine cell, density
    // kFinestRes^2 = exp(2*log(kFinestRes)); both histograms are discrete
    // categoricals (log-probability sums to 1 over their own 64 bins), so
    // their product with the uniform-jitter density gives the full continuous
    // density over x -- the same Jacobian HGGrid adds for its single-level
    // grid, doubled for HDF's two nested 8x subdivisions per axis.
    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float2 x) {
        float2 xs = x * float(kCoarseDim);
        uint2 coarseCell = clamp(uint2(floor(xs)), uint2(0u), uint2(kCoarseDim - 1u));
        float2 t = xs - float2(coarseCell);

        float2 fs = t * float(kFineDim);
        uint2 fineCell = clamp(uint2(floor(fs)), uint2(0u), uint2(kFineDim - 1u));

        uint coarseBin = grid2bin(coarseCell, kCoarseDim);
        uint fineBin   = grid2bin(fineCell, kFineDim);

        Histogram<kNumCoarseCells> coarseHist;
        let coarseLogits = forwardCoarse(weights, layout, ctx);
        float logCoarse = coarseHist.eval(coarseLogits, coarseBin);

        Histogram<kNumFineCells> fineHist;
        let fineLogits = forwardFine(weights, layout, ctx, coarseCell);
        float logFine = fineHist.eval(fineLogits, fineBin);

        return logCoarse + logFine + 2.0f * log(float(kFinestRes));
    }

    static float2 sample(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, inout PCG32 rand) {
        Histogram<kNumCoarseCells> coarseHist;
        let coarseLogits = forwardCoarse(weights, layout, ctx);
        uint coarseBin = coarseHist.sample(coarseLogits, rand.nextFloat());
        uint2 coarseCell = bin2grid(coarseBin, kCoarseDim);

        Histogram<kNumFineCells> fineHist;
        let fineLogits = forwardFine(weights, layout, ctx, coarseCell);
        uint fineBin = fineHist.sample(fineLogits, rand.nextFloat());
        uint2 fineCell = bin2grid(fineBin, kFineDim);

        float2 jitter = float2(rand.nextFloat(), rand.nextFloat());
        return (float2(coarseCell) * float(kFineDim) + float2(fineCell) + jitter) / float(kFinestRes);
    }

    // Deterministic mirror of sample() for layers_viz's Stages.slang: every
    // "random" draw sample() would make is instead the fixed quantile 0.5,
    // so the traced trajectory is a reproducible "typical" sample instead of
    // one realization of many. Not used by evalLogPDF/sample/training.
    static void sampleTrajectoryFixed(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx,
        out float2 prior, out float2 afterLayer1, out float2 outFinal)
    {
        prior = float2(0.5f, 0.5f);

        Histogram<kNumCoarseCells> coarseHist;
        let coarseLogits = forwardCoarse(weights, layout, ctx);
        uint coarseBin = coarseHist.sample(coarseLogits, 0.5f);
        uint2 coarseCell = bin2grid(coarseBin, kCoarseDim);
        afterLayer1 = (float2(coarseCell) + float2(0.5f, 0.5f)) / float(kCoarseDim);

        Histogram<kNumFineCells> fineHist;
        let fineLogits = forwardFine(weights, layout, ctx, coarseCell);
        uint fineBin = fineHist.sample(fineLogits, 0.5f);
        uint2 fineCell = bin2grid(fineBin, kFineDim);
        float2 jitter = float2(0.5f, 0.5f);
        outFinal = (float2(coarseCell) * float(kFineDim) + float2(fineCell) + jitter) / float(kFinestRes);
    }

    // Exact (no sampling) coarse-cell probability mass, after layer 1 only --
    // forwardCoarse's logits don't depend on any continuous input (ctx is
    // IArchitecture's trivial constant), so this is a single small softmax
    // read-back, not a histogram of samples.
    static float[kNumCoarseCells] stage1MarginalCoarse(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx)
    {
        let logits = forwardCoarse(weights, layout, ctx);
        Histogram<kNumCoarseCells> hist;
        float probs[kNumCoarseCells];
        [ForceUnroll]
        for (uint i = 0; i < kNumCoarseCells; ++i) probs[i] = exp(hist.eval(logits, i));
        return probs;
    }
}
