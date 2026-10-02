import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;
import TSNN.Encodings.TriangleWave;
import architectures.IArchitecture;

import TSNN.Mixtures.Histogram;
import TSNN.Encodings.OneHot;
import TSNN.Encodings.Compose;

/// DFN<K>: mlp -> histogram -> sample bin -> encode bin -> mlp -> histogram -> sample bin -> encode bin.
/// Generic over per-axis bin count K (FINDING.md's DF-vs-NSF speed investigation: DF's
/// Histogram<K>::eval/sample fully unroll 4 O(K) passes, autodiff'd every training step,
/// so K is directly comparable to NSFLinear's kNumBins=16 -- DFN16/DFL16 below match it).
struct DFNImpl<let K : uint> : IArchitecture
{
    static constexpr const uint kX = K; // x resolution
    static constexpr const uint kY = K; // y resolution
    static constexpr const uint kEncodedDim = 12; // TriangleWave encoding
    static constexpr const uint kContextDim = 1u; // must match IArchitecture.Context's CoopVec width

    typealias XNet = MLP<kContextDim, kX, 3, kX>;
    typealias YNet   = MLP<kContextDim + kEncodedDim, kY, 3, kY>;
    typealias Activation = LeakyReLU;

    // Base indices into the host-supplied `Layout` buffer (see
    // IArchitecture.slang and TSNN.Utils.MLP's static from-layout methods'
    // doc comment): each MLP contributes DEPTH+1 = 4 entries, chained in
    // declaration order, mirroring benchmark.py's own layout builder for
    // this architecture.
    static constexpr const uint kXBase = 0u;
    static constexpr const uint kYBase = kXBase + 4u;

    static uint getParamCount(no_diff Layout layout) {
        // YNet's end covers both blocks: both MLPs are chained into one buffer.
        return YNet.paramCount(layout, kYBase);
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, no_diff Layout layout, uint tid, uint threadCount)
    {
        XNet.initParams<TMirror, TMaster>(params, master, layout, kXBase, tid, threadCount);
        YNet.initParams<TMirror, TMaster>(params, master, layout, kYBase, tid, threadCount);
    }

    [Differentiable]
    static float[kX] forwardX(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx) {
        return CoopVecToArray(XNet.forwardStatic<NDEOffsets, Activation>(weights, ctx, kXBase));
    }

    [Differentiable]
    static float[kY] forwardY(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float x) {
        // ComposeCoop needs half[] on both sides; encodeTriangleWave returns a
        // CoopVec, so route it through CoopVecToArray first (safe here: x is
        // no_diff data, not a learnable upstream feature, so there is no
        // gradient for that conversion to drop). ctx stays a CoopVec and is
        // composed directly via ComposeCoopVecPlus, same as HGGrid.encodeCell.
        let encFloats = CoopVecToArray(no_diff encodeTriangleWave<half, kEncodedDim>(x));
        half enc[kEncodedDim];
        [ForceUnroll]
        for (uint i = 0; i < kEncodedDim; ++i)
            enc[i] = half(encFloats[i]);

        let input = ComposeCoopVecPlus<kContextDim, kEncodedDim>(ctx, enc);
        let params = CoopVecToArray(YNet.forwardStatic<NDEOffsets, Activation>(weights, input, kYBase));
        return params;
    }

    // log p(x) = logHistX(binX) + logHistY(binY) + log(kX) + log(kY): each
    // histogram's log-probability sums to 1 over its bins, but a bin has
    // continuous width 1/kX (resp. 1/kY), so converting to a density over x
    // needs the same per-axis resolution Jacobian HGGrid adds for its grid.
    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float2 x) {
        uint2 cell = uint2(x * float2(kX, kY));
        var logPDF = 0.0f;

        Histogram<kX> histX;
        let histXLogits = forwardX(weights, layout, ctx);
        logPDF += histX.eval(histXLogits, cell.x);

        Histogram<kY> histY;
        let histYLogits = forwardY(weights, layout, ctx, x.x);
        logPDF += histY.eval(histYLogits, cell.y);

        return logPDF + log(float(kX)) + log(float(kY));
    }

    static float2 sample(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, inout PCG32 rand)
    {
        // Draw in bin-X, jitter-X, bin-Y, jitter-Y order.
        let u = float4(rand.nextFloat(), rand.nextFloat(), rand.nextFloat(), rand.nextFloat());
        float2 prior, afterLayer1;
        return sampleAt(weights, layout, ctx, u, prior, afterLayer1);
    }

    static float2 sampleAt(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, float4 u,
        out float2 prior, out float2 afterLayer1)
    {
        prior = float2(0.5f);
        Histogram<kX> histX;
        let histXLogits = forwardX(weights, layout, ctx);
        let binX = histX.sample(histXLogits, u.x);
        // Jitter X before conditioning Y on its continuous coordinate.
        let x = (float(binX) + u.y) / float(kX);
        afterLayer1 = float2(x, prior.y);

        Histogram<kY> histY;
        let histYLogits = forwardY(weights, layout, ctx, x);
        let binY = histY.sample(histYLogits, u.z);
        let y = (float(binY) + u.w) / float(kY);
        return float2(x, y);
    }

    static void sampleTrajectoryFixed(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx,
        out float2 prior, out float2 afterLayer1, out float2 outFinal)
    {
        outFinal = sampleAt(weights, layout, ctx, float4(0.5f), prior, afterLayer1);
    }

    // Exact (no sampling) marginal-over-X probability mass per bin, after
    // layer 1 only -- forwardX's logits don't depend on any continuous
    // input (ctx is IArchitecture's trivial constant), so this is a single
    // small softmax read-back, not a histogram of samples.
    static float[kX] stage1MarginalX(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx)
    {
        let logits = forwardX(weights, layout, ctx);
        Histogram<kX> hist;
        float probs[kX];
        [ForceUnroll]
        for (uint i = 0; i < kX; ++i) probs[i] = exp(hist.eval(logits, i));
        return probs;
    }
}

typealias DFN = DFNImpl<32u>;
// Bin-matched to NSFLinear's kNumBins=16, to isolate "explicit histogram vs. coupling-flow
// spline" from "resolution" when comparing DFN's training cost against NSF-L's.
typealias DFN16 = DFNImpl<16u>;
