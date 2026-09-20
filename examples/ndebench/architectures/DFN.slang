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

/// DFN: mlp -> histogram -> sample bin -> encode bin -> mlp -> histogram -> sample bin -> encode bin
struct DFN : IArchitecture
{
    static constexpr const uint kX = 32; // x resolution
    static constexpr const uint kY = 32; // y resolution
    static constexpr const uint kEncodedDim = 12; // TriangleWave encoding
    static constexpr const uint kContextDim = 1u; // must match IArchitecture.Context's CoopVec width

    typealias XNet = MLP<kContextDim, kX, 3, kX>;
    typealias YNet   = MLP<kContextDim + kEncodedDim, kY, 3, kY>;
    typealias Activation = LeakyReLU;

    static constexpr const XNet kXMLP = XNet(0u);
    static constexpr const YNet kYMLP   = YNet(kXMLP.end);

    static uint getParamCount() {
        // `end` is a byte offset in the fp16 parameter layout; both MLPs are
        // chained into one buffer, so the fine MLP's end covers both blocks.
        return kYMLP.end / uint(sizeof(half));
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, uint tid, uint threadCount)
    {
        kXMLP.initParams<TMirror, TMaster>(params, master, tid, threadCount);
        kYMLP.initParams<TMirror, TMaster>(params, master, tid, threadCount);
    }

    [Differentiable]
    static float[kX] forwardX(MatrixBiasBuffer weights, Context ctx) {
        return CoopVecToArray(kXMLP.forward<Activation>(weights, ctx));
    }

    [Differentiable]
    static float[kY] forwardY(MatrixBiasBuffer weights, Context ctx, no_diff float x) {
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
        let params = CoopVecToArray(kYMLP.forward<Activation>(weights, input));
        return params;
    }

    // log p(x) = logHistX(binX) + logHistY(binY) + log(kX) + log(kY): each
    // histogram's log-probability sums to 1 over its bins, but a bin has
    // continuous width 1/kX (resp. 1/kY), so converting to a density over x
    // needs the same per-axis resolution Jacobian HGGrid adds for its grid.
    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, Context ctx, no_diff float2 x) {
        uint2 cell = uint2(x * float2(kX, kY));
        var logPDF = 0.0f;

        Histogram<kX> histX;
        let histXLogits = forwardX(weights, ctx);
        logPDF += histX.eval(histXLogits, cell.x);

        Histogram<kY> histY;
        let histYLogits = forwardY(weights, ctx, x.x);
        logPDF += histY.eval(histYLogits, cell.y);

        return logPDF + log(float(kX)) + log(float(kY));
    }

    static float2 sample(MatrixBiasBuffer weights, Context ctx, inout PCG32 rand) {
        // histX.sample gives a discrete bin, not a continuous ε1 -- the density
        // is uniform WITHIN each bin (Eq. 4's piecewise-constant p(ε) = v[bin]),
        // so a real draw needs an independent uniform jitter within the chosen
        // bin. The bin alone (equivalently: always jittering to exactly 0) would
        // return only kX*kY fixed grid points, never anything continuous. x must
        // be jittered BEFORE conditioning forwardY, since Y|X is conditioned on
        // the continuous ε1, not on which bin it fell in.
        Histogram<kX> histX;
        let histXLogits = forwardX(weights, ctx);
        let binX = histX.sample(histXLogits, rand.nextFloat());
        let x = (float(binX) + rand.nextFloat()) / float(kX);

        Histogram<kY> histY;
        let histYLogits = forwardY(weights, ctx, x);
        let binY = histY.sample(histYLogits, rand.nextFloat());
        let y = (float(binY) + rand.nextFloat()) / float(kY);

        return float2(x, y);
    }
}
