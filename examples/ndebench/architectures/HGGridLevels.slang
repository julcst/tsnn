import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;
import architectures.IArchitecture;

import TSNN.Mixtures.Histogram;
import TSNN.Mixtures.TruncatedGMM;
import TSNN.Encodings.OneHot;
import TSNN.Encodings.OneBlob;
import TSNN.Encodings.Compose;

/// HGGridLevels<G,L,HID,ENC,KBINS,K>: generalises architectures.HGGrid's fixed
/// single-level 8x8 coarse grid to L cascaded levels of GxG grids (each level a
/// small Histogram<G*G> MLP head, mirroring architectures.HDFLevels' own
/// generalisation of HDF) before terminating in a continuous
/// TruncatedGMM<K> head over the residual position inside the finest cell --
/// exactly what HGGrid's own single-level FineNet already does, just fed a
/// deeper discrete path instead of one coarse cell. Same experiment as
/// HDFLevels, applied to HGGrid's coarse/fine split instead of HDF's
/// coarse/coarse split: does cascading several small histogram heads (narrower
/// softmax, cheaper gradient scatter) beat one wide head at the same total
/// discrete resolution G^L, and how far can L go before the extra sequential
/// MLPs cost more than the head-width saving?
///
/// ENC selects how level i>0 (and the final GMM head) is conditioned on the
/// cells already chosen at shallower levels (0..count-1, count = this head's
/// ancestor count -- up to L-1 for a histogram level, exactly L for the GMM
/// head, since it comes after all L levels), both no_diff / constant, both
/// exactly kEncodedDim = 2*KBINS wide per call:
///   ENC=0: path one-hot -- per axis, a concatenation of L oneHot<G> blocks,
///     one per possible ancestor level, with only the first `count` filled
///     (deeper blocks zero) -- direct generalisation of HGGrid's encodeCell,
///     which only ever has one level to encode. Requires KBINS == G*L (one
///     block per level, including the deepest, since the GMM head needs all
///     L ancestors -- unlike HDFLevels, whose deepest head only ever needs
///     L-1 ancestors).
///   ENC=1: one-blob -- per axis, encodeOneBlob<half,KBINS> of the center of
///     the sub-cell the first `count` ancestors have narrowed into. KBINS is
///     a free, typically smaller, compressed width.
struct HGGridLevels<
    let G : uint, let L : uint, let HID : uint, let ENC : uint, let KBINS : uint, let K : uint,
    let DEPTH : uint = 3u
> : IArchitecture
{
    static constexpr const uint kNumCells   = G * G;
    static constexpr const uint kEncodedDim = 2u * KBINS;
    static constexpr const uint kContextDim = 1u; // must match IArchitecture.Context's CoopVec width
    static constexpr const uint kGMMStride  = 5u; // TruncatedGMM<K>::STRIDE = 1 + 2*D, D=2 (weight, mean, logSigma per axis)

    typealias RootNet = MLP<kContextDim, HID, DEPTH, kNumCells>;
    // Context ++ encodeCell(ancestors)
    typealias RestNet = MLP<kContextDim + kEncodedDim, HID, DEPTH, kNumCells>;
    // Context ++ encodeCell(all L levels)
    typealias FineNet = MLP<kContextDim + kEncodedDim, HID, DEPTH, K * kGMMStride>;
    typealias Activation = LeakyReLU;
    typealias GMM = TruncatedGMM<K>;

    static constexpr const GMM kGMM = GMM();

    // Base index into the host-supplied `Layout` buffer for level `level`'s
    // MLP: each level contributes DEPTH+1 entries, chained in level order
    // (0..L-1), then the final GMM head, mirroring benchmark.py's own layout
    // builder for this architecture (see architectures.HDFLevels for the
    // shared convention).
    static uint levelBase(uint level) { return (DEPTH + 1u) * level; }
    static uint fineBase() { return levelBase(L); }

    // [ctx | enc]. See architectures.HDFLevels.composeInput's doc comment for
    // why this is spelled out as a plain (non-generic) static method instead
    // of TSNN.Encodings.Compose's ComposeCoopVecPlus: both RestNet and
    // FineNet declare their INPUT as the single cast `kContextDim +
    // kEncodedDim`, and Slang's generic-argument unifier does not normalize
    // that against ComposeCoopVecPlus's separately-inferred `int(N)+int(M)`.
    [BackwardDerivative(composeInput_bwd)]
    static CoopVec<half, kContextDim + kEncodedDim> composeInput(Context ctx, no_diff half[kEncodedDim] enc) {
        CoopVec<half, kContextDim + kEncodedDim> output;
        [ForceUnroll] for (uint i = 0; i < kContextDim; ++i) output[i] = ctx[i];
        [ForceUnroll] for (uint i = 0; i < kEncodedDim; ++i) output[kContextDim + i] = enc[i];
        return output;
    }

    static void composeInput_bwd(
        inout DifferentialPair<Context> ctx, no_diff half[kEncodedDim] enc, CoopVec<half, kContextDim + kEncodedDim> dOutput)
    {
        Context dCtx;
        [ForceUnroll] for (uint i = 0; i < kContextDim; ++i) dCtx[i] = dOutput[i];
        ctx = diffPair(ctx.p, dCtx);
    }

    static uint getParamCount(no_diff Layout layout) {
        // FineNet's end covers every earlier block: all L levels plus the GMM head are chained into one buffer.
        return FineNet.paramCount(layout, fineBase());
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, no_diff Layout layout, uint tid, uint threadCount)
    {
        RootNet.initParams<TMirror, TMaster>(params, master, layout, levelBase(0u), tid, threadCount);
        [ForceUnroll]
        for (uint i = 1u; i < L; ++i)
            RestNet.initParams<TMirror, TMaster>(params, master, layout, levelBase(i), tid, threadCount);
        FineNet.initParams<TMirror, TMaster>(params, master, layout, fineBase(), tid, threadCount);
    }

    // Encoding fed to a head conditioned on `count` ancestors (count in
    // [1, L-1] for a histogram level, count == L for the final GMM head),
    // built from the local GxG cell chosen at each ancestor level j < count
    // (pathX/pathY[j]); entries for j >= count are never read -- callers need
    // not pre-zero pathX/pathY.
    static half[kEncodedDim] encodeCell(no_diff uint pathX[L], no_diff uint pathY[L], uint count) {
        half enc[kEncodedDim];
        if (ENC == 0u) {
            [ForceUnroll]
            for (uint j = 0; j < L; ++j) {
                let hx = (j < count) ? oneHot<half, G>(pathX[j]) : oneHot<half, G>(0u);
                let hy = (j < count) ? oneHot<half, G>(pathY[j]) : oneHot<half, G>(0u);
                [ForceUnroll]
                for (uint k = 0; k < G; ++k) {
                    enc[j * G + k]         = (j < count) ? hx[k] : half(0.0f);
                    enc[KBINS + j * G + k] = (j < count) ? hy[k] : half(0.0f);
                }
            }
        } else {
            // Center, in [0,1) per axis, of the sub-cell the first `count`
            // ancestors have narrowed x/y into: sum_{j<count} pathX[j]/G^(j+1),
            // plus half of that sub-cell's own size (1/G)^count.
            float cx = 0.0f, cy = 0.0f, scale = 1.0f, cellSize = 1.0f;
            [ForceUnroll]
            for (uint j = 0; j < L; ++j) {
                scale /= float(G);
                if (j < count) {
                    cx += float(pathX[j]) * scale;
                    cy += float(pathY[j]) * scale;
                    cellSize = scale;
                }
            }
            cx += 0.5f * cellSize;
            cy += 0.5f * cellSize;
            let bx = CoopVecToArray(encodeOneBlob<half, int(KBINS)>(cx));
            let by = CoopVecToArray(encodeOneBlob<half, int(KBINS)>(cy));
            [ForceUnroll]
            for (uint k = 0; k < KBINS; ++k) {
                enc[k]         = half(bx[k]);
                enc[KBINS + k] = half(by[k]);
            }
        }
        return enc;
    }

    [Differentiable]
    static float[kNumCells] forwardLevel(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx,
        no_diff uint pathX[L], no_diff uint pathY[L], uint level)
    {
        if (level == 0u) {
            return CoopVecToArray(RootNet.forward<Activation>(weights, layout, levelBase(0u), ctx));
        } else {
            let enc = encodeCell(pathX, pathY, level);
            let input = composeInput(ctx, enc);
            return CoopVecToArray(RestNet.forward<Activation>(weights, layout, levelBase(level), input));
        }
    }

    [Differentiable]
    static float[K * kGMMStride] forwardFine(
        MatrixBiasBuffer weights, no_diff Layout layout, Context ctx,
        no_diff uint pathX[L], no_diff uint pathY[L])
    {
        let enc = encodeCell(pathX, pathY, L);
        let input = composeInput(ctx, enc);
        return CoopVecToArray(FineNet.forward<Activation>(weights, layout, fineBase(), input));
    }

    // log p(x) = sum_i logHist_i(bin_i) + logFine(t) + 2*L*log(G): L nested
    // GxG subdivisions per axis reach a residual cell of size (1/G)^L, whose
    // area-scaling Jacobian is 2*L*log(G) (generalising HGGrid.slang's own
    // 2*log(kGridDim) for its single level); the continuous GMM head then
    // supplies the density shape within that residual cell, exactly as
    // HGGrid's FineNet does within its one coarse cell.
    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float2 x) {
        uint pathX[L], pathY[L];
        float2 xs = x;
        [ForceUnroll]
        for (uint i = 0; i < L; ++i) {
            xs *= float(G);
            uint2 cell = clamp(uint2(floor(xs)), uint2(0u), uint2(G - 1u));
            pathX[i] = cell.x;
            pathY[i] = cell.y;
            xs -= float2(cell);
        }

        float logp = 0.0f;
        [ForceUnroll]
        for (uint i = 0; i < L; ++i) {
            Histogram<kNumCells> hist;
            let logits = forwardLevel(weights, layout, ctx, pathX, pathY, i);
            uint bin = grid2bin(uint2(pathX[i], pathY[i]), G);
            logp += hist.eval(logits, bin);
        }

        let fineParams = forwardFine(weights, layout, ctx, pathX, pathY);
        float logFine = kGMM.eval(fineParams, xs);

        return logp + logFine + 2.0f * float(L) * log(float(G));
    }

    static float2 sample(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, inout PCG32 rand) {
        uint pathX[L], pathY[L];
        [ForceUnroll]
        for (uint i = 0; i < L; ++i) { pathX[i] = 0u; pathY[i] = 0u; }

        [ForceUnroll]
        for (uint i = 0; i < L; ++i) {
            Histogram<kNumCells> hist;
            let logits = forwardLevel(weights, layout, ctx, pathX, pathY, i);
            uint bin = hist.sample(logits, rand.nextFloat());
            uint2 cell = bin2grid(bin, G);
            pathX[i] = cell.x;
            pathY[i] = cell.y;
        }

        let fineParams = forwardFine(weights, layout, ctx, pathX, pathY);
        float2 t = kGMM.sample(fineParams, rand);

        float gx = 0.0f, gy = 0.0f, scale = 1.0f;
        [ForceUnroll]
        for (uint i = 0; i < L; ++i) {
            scale /= float(G);
            gx += float(pathX[i]) * scale;
            gy += float(pathY[i]) * scale;
        }
        return float2(gx, gy) + t * scale;
    }
}

typealias HGGridG2L3  = HGGridLevels<2u, 3u, 32u, 0u, 6u, 4u>; // 3 levels of 2x2 (4-bin heads), path one-hot -- iso-resolution (8) HGGrid comparison
typealias HGGridG2L3B = HGGridLevels<2u, 3u, 32u, 1u, 8u, 4u>; // same geometry as HGGridG2L3, one-blob(8) conditioning instead of path one-hot
typealias HGGridG4L2  = HGGridLevels<4u, 2u, 32u, 0u, 8u, 4u>; // 2 levels of 4x4 (16-bin heads), path one-hot -- finer res (16) than HGGrid
typealias HGGridG4L2B = HGGridLevels<4u, 2u, 32u, 1u, 8u, 4u>; // same geometry as HGGridG4L2, one-blob(8) conditioning instead of path one-hot
typealias HGGridG4L2S = HGGridLevels<4u, 2u, 16u, 1u, 8u, 4u>; // HGGridG4L2B with hidden width halved (16 instead of 32)

// Winners of a depth/width/gaussian-count/geometry screening sweep over ~12
// variants (see FINDING.md's HGGrid optimal-config entry for the full table
// and the runner-up configs, including one -- 2 levels of 8x8 with HID=16 --
// that reproducibly diverges: TruncatedGMM's logSigma had no floor/ceiling,
// see TSNN/Mixtures/TruncatedGMM.slang's kLogSigmaMax and FINDING.md's
// TruncatedGMM logSigma-floor entry for the fix). Key finding: raising the
// GMM head's Gaussian count K is a far better lever than a finer discrete
// grid for quality, and MLP depth (3->2) the far bigger lever for speed.
typealias HGGridFast    = HGGridLevels<4u, 2u, 16u, 0u, 8u, 8u>;     // HGGridG4L2 geometry, HID=16, K=8: better KL AND faster than HGGrid -- recommended default
typealias HGGridFastest = HGGridLevels<8u, 1u, 32u, 0u, 8u, 4u, 2u>; // HGGrid's own geometry, depth 2: fastest, KL ~= HGGrid
