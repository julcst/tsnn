import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;
import architectures.IArchitecture;
import TSNN.Mixtures.Histogram;
import TSNN.Mixtures.TruncatedGMM;
import TSNN.Encodings.OneHot;

/// L nested G×G histograms, conditioned on one-hot ancestor cells.
/// K=0 gives HDF with uniform leaf cells; K>0 adds a truncated GMM per leaf.
struct HGGrid<let G : uint, let L : uint, let HID : uint, let K : uint, let DEPTH : uint = 3u>
    : IArchitecture
{
    static constexpr const uint kAncestorCount = L - (K == 0u ? 1u : 0u);
    static constexpr const uint kEncodingBins = G * kAncestorCount;
    static constexpr const uint kEncodedDim = 2u * kEncodingBins;
    static constexpr const uint kNumCells = G * G;
    // Keep the unused head type valid for K=0; its calls and storage are omitted.
    static constexpr const uint kFineCount = K > 0u ? K : 1u;
    static constexpr const uint kFineDim = 5u * kFineCount;

    typealias RootNet = MLP<1u, HID, DEPTH, kNumCells>;
    typealias RestNet = MLP<1u + kEncodedDim, HID, DEPTH, kNumCells>;
    typealias FineNet = MLP<1u + kEncodedDim, HID, DEPTH, kFineDim>;
    typealias Activation = LeakyReLU;
    typealias GMM = TruncatedGMM<kFineCount>;
    static constexpr const GMM kGMM = GMM();

    // Each MLP contributes DEPTH+1 entries to the host-supplied layout.
    static uint levelBase(uint level) { return (DEPTH + 1u) * level; }

    // Use the same dimension expression as RestNet/FineNet for Slang's type unifier.
    [BackwardDerivative(composeInput_bwd)]
    static CoopVec<half, 1u + kEncodedDim> composeInput(Context ctx, no_diff half[kEncodedDim] enc)
    {
        CoopVec<half, 1u + kEncodedDim> output;
        output[0] = ctx[0];
        [ForceUnroll]
        for (uint i = 0; i < kEncodedDim; ++i) output[1u + i] = enc[i];
        return output;
    }

    static void composeInput_bwd(
        inout DifferentialPair<Context> ctx, no_diff half[kEncodedDim] enc,
        CoopVec<half, 1u + kEncodedDim> dOutput)
    {
        Context dCtx;
        dCtx[0] = dOutput[0];
        ctx = diffPair(ctx.p, dCtx);
    }

    static uint getParamCount(no_diff Layout layout)
    {
        if (K > 0u) return FineNet.paramCount(layout, levelBase(L));
        if (L == 1u) return RootNet.paramCount(layout, 0u);
        return RestNet.paramCount(layout, levelBase(L - 1u));
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, no_diff Layout layout,
        uint tid, uint threadCount)
    {
        RootNet.initParams<TMirror, TMaster>(params, master, layout, 0u, tid, threadCount);
        [ForceUnroll]
        for (uint i = 1u; i < L; ++i)
            RestNet.initParams<TMirror, TMaster>(params, master, layout, levelBase(i), tid, threadCount);
        if (K > 0u)
            FineNet.initParams<TMirror, TMaster>(params, master, layout, levelBase(L), tid, threadCount);
    }

    // HDF uses L-1 ancestor blocks; the Gaussian head uses all L.
    static half[kEncodedDim] encodeCell(no_diff uint pathX[L], no_diff uint pathY[L], uint count)
    {
        half enc[kEncodedDim];
        [ForceUnroll]
        for (uint j = 0; j < kAncestorCount; ++j)
        {
            let hx = oneHot<half, G>(j < count ? pathX[j] : 0u);
            let hy = oneHot<half, G>(j < count ? pathY[j] : 0u);
            [ForceUnroll]
            for (uint k = 0; k < G; ++k)
            {
                enc[j * G + k] = j < count ? hx[k] : half(0.0f);
                enc[kEncodingBins + j * G + k] = j < count ? hy[k] : half(0.0f);
            }
        }
        return enc;
    }

    [Differentiable]
    static float[kNumCells] forwardLevel(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx,
        no_diff uint pathX[L], no_diff uint pathY[L], uint level)
    {
        if (level == 0u)
            return CoopVecToArray(RootNet.forwardStatic<NDEOffsets, Activation>(weights, ctx, 0u));
        let input = composeInput(ctx, encodeCell(pathX, pathY, level));
        return CoopVecToArray(RestNet.forwardStatic<NDEOffsets, Activation>(weights, input, levelBase(level)));
    }

    [Differentiable]
    static float[kFineDim] forwardFine(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx,
        no_diff uint pathX[L], no_diff uint pathY[L])
    {
        let input = composeInput(ctx, encodeCell(pathX, pathY, L));
        return CoopVecToArray(FineNet.forwardStatic<NDEOffsets, Activation>(weights, input, levelBase(L)));
    }

    [Differentiable]
    static float evalLogPDF(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float2 x)
    {
        uint pathX[L], pathY[L];
        float2 residual = x;
        [ForceUnroll]
        for (uint i = 0; i < L; ++i)
        {
            residual *= float(G);
            uint2 cell = clamp(uint2(floor(residual)), uint2(0u), uint2(G - 1u));
            pathX[i] = cell.x;
            pathY[i] = cell.y;
            residual -= float2(cell);
        }

        float logp = 0.0f;
        [ForceUnroll]
        for (uint i = 0; i < L; ++i)
        {
            Histogram<kNumCells> hist;
            let logits = forwardLevel(weights, layout, ctx, pathX, pathY, i);
            logp += hist.eval(logits, grid2bin(uint2(pathX[i], pathY[i]), G));
        }
        if (K > 0u)
            logp += kGMM.eval(forwardFine(weights, layout, ctx, pathX, pathY), residual);
        // Every subdivision scales area by G².
        return logp + 2.0f * float(L) * log(float(G));
    }

    static float2 sample(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, inout PCG32 rand)
    {
        uint pathX[L], pathY[L];
        float2 origin = float2(0.0f);
        float scale = 1.0f;
        [ForceUnroll]
        for (uint i = 0; i < L; ++i)
        {
            Histogram<kNumCells> hist;
            let logits = forwardLevel(weights, layout, ctx, pathX, pathY, i);
            uint2 cell = bin2grid(hist.sample(logits, rand.nextFloat()), G);
            pathX[i] = cell.x;
            pathY[i] = cell.y;
            scale /= float(G);
            origin += float2(cell) * scale;
        }
        float2 residual;
        if (K > 0u)
            residual = kGMM.sample(forwardFine(weights, layout, ctx, pathX, pathY), rand);
        else
            residual = float2(rand.nextFloat(), rand.nextFloat());
        return origin + residual * scale;
    }

    // Diagnostic views use median quantiles, independent of the random sampler.
    static void sampleTrajectoryFixed(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx,
        out float2 prior, out float2 afterLayer1, out float2 outFinal)
    {
        prior = float2(0.5f);
        afterLayer1 = prior;
        uint pathX[L], pathY[L];
        float2 origin = float2(0.0f);
        float scale = 1.0f;
        [ForceUnroll]
        for (uint i = 0; i < L; ++i)
        {
            Histogram<kNumCells> hist;
            let logits = forwardLevel(weights, layout, ctx, pathX, pathY, i);
            uint2 cell = bin2grid(hist.sample(logits, 0.5f), G);
            pathX[i] = cell.x;
            pathY[i] = cell.y;
            scale /= float(G);
            origin += float2(cell) * scale;
            if (i == 0u) afterLayer1 = origin + prior * scale;
        }
        float2 residual = prior;
        if (K > 0u)
            residual = kGMM.sampleAt(forwardFine(weights, layout, ctx, pathX, pathY), 0.5f, prior);
        outFinal = origin + residual * scale;
    }

    static float[kNumCells] stage1MarginalCoarse(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx)
    {
        let logits = CoopVecToArray(RootNet.forwardStatic<NDEOffsets, Activation>(weights, ctx, 0u));
        Histogram<kNumCells> hist;
        float probs[kNumCells];
        [ForceUnroll]
        for (uint i = 0; i < kNumCells; ++i) probs[i] = exp(hist.eval(logits, i));
        return probs;
    }
}
