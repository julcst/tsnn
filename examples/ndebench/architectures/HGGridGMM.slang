import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;
import architectures.IArchitecture;

import TSNN.Mixtures.TruncatedGMM;
import TSNN.Encodings.OneHot;
import TSNN.Encodings.Compose;

/// HGGridGMM<G,HID,KC,KF,DEPTH>: the discrete "which cell" decision is made by
/// a small CONTINUOUS TruncatedGMM<KC> (the "coarse" GMM) instead of a
/// learned NxN histogram (what architectures.HGGrid/HGGridLevels use) --
/// replaces the histogram outright, it does not sit alongside one.
///
/// Two things follow from a GMM instead of a histogram supplying "P(cell)":
///  - EVAL: a cell's probability mass is TruncatedGMM.evalCellMass's closed-
///    form box integral (per-axis truncated-Gaussian CDF differences, see
///    TSNN/Mixtures/TruncatedGMM.slang), a function of the coarse GMM's KC
///    components only -- NOT of the grid resolution G. A histogram head's
///    output width (and CoopVec matrix row count) scales with G*G and hits
///    this hardware's 128-row CoopVec cap at G=16 (confirmed live trying
///    architectures.HGGridLevels<16u,1u,...>); the coarse GMM here has the
///    same KC*5-wide head at G=8, G=16, or G=128 alike.
///  - SAMPLE: draw a continuous point from the coarse GMM directly and round
///    it onto the G x G grid (floor(x*G)) to get the discrete cell -- exactly
///    "sample from the GMM, round to the grid". This is an exact draw from
///    the P(cell) distribution above (by definition of probability mass),
///    with no discrete categorical sampler (Histogram::sample's O(G*G)
///    unrolled search) at all.
///
/// The chosen cell then hands off to a second, FINE TruncatedGMM<KF>
/// conditioned on it (one-hot(cell), exactly HGGrid's own FineNet/encodeCell)
/// for the within-cell shape -- unchanged from HGGrid's own design; only the
/// coarse stage's mechanism (GMM box-mass vs. histogram softmax) differs.
struct HGGridGMM<let G : uint, let HID : uint, let KC : uint, let KF : uint, let DEPTH : uint = 3u> : IArchitecture
{
    static constexpr const uint kContextDim = 1u; // must match IArchitecture.Context's CoopVec width
    static constexpr const uint kFineEncodedDim = 2u * G; // one-hot(cell.x) ++ one-hot(cell.y)

    typealias CoarseNet = MLP<kContextDim, HID, DEPTH, KC * 5>;
    typealias FineNet   = MLP<kContextDim + kFineEncodedDim, HID, DEPTH, KF * 5>;
    typealias Activation = LeakyReLU;
    typealias CoarseGMM = TruncatedGMM<KC>;
    typealias FineGMM   = TruncatedGMM<KF>;

    static constexpr const CoarseGMM kCoarseGMM = CoarseGMM();
    static constexpr const FineGMM kFineGMM = FineGMM();

    // Base indices into the host-supplied `Layout` buffer (see
    // IArchitecture.slang and TSNN.Utils.MLP's static from-layout methods'
    // doc comment): each MLP contributes DEPTH+1 entries, chained in
    // declaration order, mirroring benchmark.py's own layout builder.
    static uint coarseBase() { return 0u; }
    static uint fineBase() { return coarseBase() + (DEPTH + 1u); }

    static uint getParamCount(no_diff Layout layout) {
        // FineNet's end covers both blocks: both MLPs are chained into one buffer.
        return FineNet.paramCount(layout, fineBase());
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, no_diff Layout layout, uint tid, uint threadCount)
    {
        CoarseNet.initParams<TMirror, TMaster>(params, master, layout, coarseBase(), tid, threadCount);
        FineNet.initParams<TMirror, TMaster>(params, master, layout, fineBase(), tid, threadCount);
    }

    // [ctx | enc], spelled out as a plain (non-generic) static method instead
    // of TSNN.Encodings.Compose's ComposeCoopVecPlus: FineNet declares its
    // INPUT as the single cast `kContextDim + kFineEncodedDim`, and Slang's
    // generic-argument unifier does not normalize that against
    // ComposeCoopVecPlus's separately-inferred `int(N)+int(M)` -- confirmed
    // live (`expected CoopVec<half, int(N+1)>, got CoopVec<half, int(N)+1>`).
    // Same fix architectures.HGGridLevels' own composeInput doc comment
    // already documents for the identical issue there.
    [BackwardDerivative(composeFineInput_bwd)]
    static CoopVec<half, kContextDim + kFineEncodedDim> composeFineInput(Context ctx, no_diff half[kFineEncodedDim] enc) {
        CoopVec<half, kContextDim + kFineEncodedDim> output;
        [ForceUnroll] for (uint i = 0; i < kContextDim; ++i) output[i] = ctx[i];
        [ForceUnroll] for (uint i = 0; i < kFineEncodedDim; ++i) output[kContextDim + i] = enc[i];
        return output;
    }

    static void composeFineInput_bwd(
        inout DifferentialPair<Context> ctx, no_diff half[kFineEncodedDim] enc, CoopVec<half, kContextDim + kFineEncodedDim> dOutput) {
        Context dCtx;
        [ForceUnroll] for (uint i = 0; i < kContextDim; ++i) dCtx[i] = dOutput[i];
        ctx = diffPair(ctx.p, dCtx);
    }

    static half[kFineEncodedDim] encodeCell(no_diff uint2 cell) {
        let ex = oneHot<half, G>(cell.x);
        let ey = oneHot<half, G>(cell.y);
        half enc[kFineEncodedDim];
        [ForceUnroll]
        for (uint i = 0; i < G; ++i)
        {
            enc[i]     = ex[i];
            enc[G + i] = ey[i];
        }
        return enc;
    }

    [Differentiable]
    static float[KC * 5] forwardCoarse(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx) {
        return CoopVecToArray(CoarseNet.forward<Activation>(weights, layout, coarseBase(), ctx));
    }

    [Differentiable]
    static float[KF * 5] forwardFine(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff uint2 cell) {
        let input = composeFineInput(ctx, encodeCell(cell));
        return CoopVecToArray(FineNet.forward<Activation>(weights, layout, fineBase(), input));
    }

    // log p(x) = logCellMass(cell) + logFine(t) + 2*log(G): logCellMass is
    // the coarse GMM's own closed-form box-probability (already a properly
    // normalized log-probability, no extra Jacobian of its own), and the
    // 2*log(G) term converts that per-cell MASS into a density over the
    // cell's area (1/G)^2 for the fine GMM's within-cell shape to multiply
    // against -- same convention as HGGrid's 2*log(kGridDim).
    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float2 x) {
        float2 xs = x * float(G);
        uint2 cell = clamp(uint2(floor(xs)), uint2(0u), uint2(G - 1u));
        float2 t = xs - float2(cell);
        float2 lo = float2(cell) / float(G);
        float2 hi = float2(cell + 1u) / float(G);

        let coarseParams = forwardCoarse(weights, layout, ctx);
        float logCellMass = kCoarseGMM.evalCellMass(coarseParams, lo, hi);

        let fineParams = forwardFine(weights, layout, ctx, cell);
        float logFine = kFineGMM.eval(fineParams, t);

        return logCellMass + logFine + 2.0f * log(float(G));
    }

    static float2 sample(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, inout PCG32 rand) {
        let coarseParams = forwardCoarse(weights, layout, ctx);
        float2 xc = kCoarseGMM.sample(coarseParams, rand); // continuous draw from the coarse GMM
        uint2 cell = clamp(uint2(floor(xc * float(G))), uint2(0u), uint2(G - 1u)); // round to the grid

        let fineParams = forwardFine(weights, layout, ctx, cell);
        float2 t = kFineGMM.sample(fineParams, rand);

        return (float2(cell) + t) / float(G);
    }

    // Deterministic mirror of sample() for layers_viz's Stages.slang -- see
    // HGGrid.slang's own sampleTrajectoryFixed for the convention. Uses each
    // GMM's own sampleAt(0.5, (0.5,0.5)) "typical" quantile instead of a
    // random draw at both the coarse-round and fine stages.
    static void sampleTrajectoryFixed(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx,
        out float2 prior, out float2 afterLayer1, out float2 outFinal)
    {
        prior = float2(0.5f, 0.5f);

        let coarseParams = forwardCoarse(weights, layout, ctx);
        float2 xc = kCoarseGMM.sampleAt(coarseParams, 0.5f, float2(0.5f, 0.5f));
        uint2 cell = clamp(uint2(floor(xc * float(G))), uint2(0u), uint2(G - 1u));
        afterLayer1 = (float2(cell) + float2(0.5f, 0.5f)) / float(G);

        let fineParams = forwardFine(weights, layout, ctx, cell);
        float2 t = kFineGMM.sampleAt(fineParams, 0.5f, float2(0.5f, 0.5f));
        outFinal = (float2(cell) + t) / float(G);
    }
}

// The literal "coarse GMM rounds to a 16x16 grid, truncated GMM in that cell"
// architecture. HID/depth match HGGrid's own coarse-net capacity; KC=8
// coarse / KF=8 fine components per the proven GMM-count lever (FINDING.md's
// HGGrid optimal-config entry).
typealias HGGridGMM16 = HGGridGMM<16u, 32u, 8u, 8u>;
// Same 16x16 grid, HGGridFast's cheaper HID=16/depth=2 profile.
typealias HGGridGMM16Fast = HGGridGMM<16u, 16u, 8u, 8u, 2u>;
// Same geometry as HGGridGMM16, coarse GMM widened 8->32 components: tests
// whether a smooth few-component coarse GMM's quality gap vs. a freely
// parameterized per-cell histogram is closable just by adding more coarse
// components (cheap here -- KC only widens the coarse head's KC*5 output,
// unlike a histogram where reaching more cells needs more OUTPUT CELLS,
// i.e. exactly the G*G scaling this architecture exists to avoid).
typealias HGGridGMM16KC24 = HGGridGMM<16u, 32u, 24u, 8u>; // KC*5=120, max under the 128-row CoopVec cap
