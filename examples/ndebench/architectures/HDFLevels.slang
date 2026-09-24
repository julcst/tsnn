import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;
import architectures.IArchitecture;

import TSNN.Mixtures.Histogram;
import TSNN.Encodings.OneHot;
import TSNN.Encodings.OneBlob;
import TSNN.Encodings.Compose;

/// HDFLevels<G,L,HID,ENC,KBINS>: generalises architectures.HDF's fixed 2-level
/// 8x8/8x8 coarse/fine histogram cascade to L levels of GxG grids, each level
/// a small Histogram<G*G> MLP head instead of HDF's 64-bin heads (experiment:
/// does more levels of narrower heads beat HDF's 2-level/64-bin design on
/// per-step cost at the same finest resolution G^L?).
///
/// ENC selects how a level > 0 is conditioned on the cells already chosen at
/// shallower levels (levels 0..count-1, count = this level's index), both
/// no_diff / constant, both exactly kEncodedDim = 2*KBINS wide per call:
///   ENC=0: path one-hot -- per axis, a concatenation of L-1 oneHot<G>
///     blocks, one per possible ancestor level, with only the first `count`
///     filled (deeper blocks zero) -- the direct generalisation of HDF's
///     encodeCell, which only ever has one ancestor to encode. Requires
///     KBINS == G*(L-1) so every block fits exactly.
///   ENC=1: one-blob -- per axis, encodeOneBlob<half,KBINS> of the center of
///     the sub-cell the first `count` ancestors have narrowed into, the
///     NSFRQS.slang condInput call pattern applied to a cell position
///     instead of a coupling coordinate. KBINS is a free, typically smaller,
///     compressed width (nearby cells now share similar encodings instead of
///     being mutually orthogonal, unlike one-hot).
struct HDFLevels<
    let G : uint, let L : uint, let HID : uint, let ENC : uint, let KBINS : uint, let DEPTH : uint = 3u
> : IArchitecture
{
    static constexpr const uint kNumCells   = G * G;
    static constexpr const uint kEncodedDim = 2u * KBINS;
    static constexpr const uint kContextDim = 1u; // must match IArchitecture.Context's CoopVec width

    typealias RootNet = MLP<kContextDim, HID, DEPTH, kNumCells>;
    // Context ++ encodeCell(ancestors)
    typealias RestNet = MLP<kContextDim + kEncodedDim, HID, DEPTH, kNumCells>;
    typealias Activation = LeakyReLU;

    // Base index into the host-supplied `Layout` buffer for level `level`'s
    // MLP: each level contributes DEPTH+1 entries, chained in level order
    // (0..L-1), mirroring benchmark.py's own layout builder for this architecture.
    static uint levelBase(uint level) { return (DEPTH + 1u) * level; }

    // [ctx | enc], RestNet's input. Deliberately NOT TSNN.Encodings.Compose's
    // ComposeCoopVecPlus<let N:int, let M:int>: that infers N and M each as a
    // separately-cast `int` from head/tail's actual widths, giving a result type
    // `int(N)+int(M)`, whereas RestNet's declared INPUT casts the whole uint sum
    // `kContextDim+kEncodedDim` to `int` exactly once -- two distinct type-level terms
    // for the same value that Slang's generic unifier refuses to match. Spelling this out
    // as a plain (non-generic) static method keeps both sides referencing the identical
    // `kContextDim + kEncodedDim` source expression, so both get the same single cast.
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
        // The last level's end covers every earlier block: all L levels are chained into one buffer.
        return RestNet.paramCount(layout, levelBase(L - 1u));
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, no_diff Layout layout, uint tid, uint threadCount)
    {
        RootNet.initParams<TMirror, TMaster>(params, master, layout, levelBase(0u), tid, threadCount);
        [ForceUnroll]
        for (uint i = 1u; i < L; ++i)
            RestNet.initParams<TMirror, TMaster>(params, master, layout, levelBase(i), tid, threadCount);
    }

    // Encoding fed to level `count` (count in [1, L-1]), built from the local
    // GxG cell chosen at each ancestor level j < count (pathX/pathY[j]);
    // entries for j >= count are never read (see ENC==0's ternary / ENC==1's
    // guarded accumulation) -- callers need not pre-zero pathX/pathY.
    static half[kEncodedDim] encodeCell(no_diff uint pathX[L], no_diff uint pathY[L], uint count) {
        half enc[kEncodedDim];
        if (ENC == 0u) {
            [ForceUnroll]
            for (uint j = 0; j < L - 1u; ++j) {
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
            for (uint j = 0; j < L - 1u; ++j) {
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

    // log p(x) = sum_i logHist_i(bin_i) + 2*L*log(G): the Jacobian of the
    // (bin_0, ..., bin_{L-1}) -> x map, nesting L GxG subdivisions per axis,
    // generalising HDF.slang's 2*log(kFinestRes) (kFinestRes = G^L here).
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
        return logp + 2.0f * float(L) * log(float(G));
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

        float2 jitter = float2(rand.nextFloat(), rand.nextFloat());
        float gx = 0.0f, gy = 0.0f, scale = 1.0f;
        [ForceUnroll]
        for (uint i = 0; i < L; ++i) {
            scale /= float(G);
            gx += float(pathX[i]) * scale;
            gy += float(pathY[i]) * scale;
        }
        return float2(gx, gy) + jitter * scale;
    }
}

typealias HDFG4L3  = HDFLevels<4u, 3u, 32u, 0u, 8u>; // 3 levels of 4x4 (16-bin heads), path one-hot -- iso-resolution (64) HDF comparison
typealias HDFG3L4  = HDFLevels<3u, 4u, 32u, 0u, 9u>; // 4 levels of 3x3 (9-bin heads), path one-hot -- finest res 81
typealias HDFG4L3B = HDFLevels<4u, 3u, 32u, 1u, 8u>; // same geometry as HDFG4L3, one-blob(8) conditioning instead of path one-hot
typealias HDFG3L4B = HDFLevels<3u, 4u, 32u, 1u, 8u>; // same geometry as HDFG3L4, one-blob(8) conditioning instead of path one-hot
typealias HDFG4L3S = HDFLevels<4u, 3u, 16u, 1u, 8u>; // HDFG4L3B with hidden width halved (16 instead of 32)

// Winners of a depth/width/geometry screening sweep over ~10 variants (see
// FINDING.md's HDF optimal-config entry for the full table and the runner-up
// configs that lost on speed, quality, or both). Key finding: MLP depth (3->2)
// is a far bigger lever than hidden width or grid geometry -- HDFFast is
// HDF's OWN 2-level 8x8/8x8 geometry with depth 2 instead of 3, and is
// strictly better than HDF on every axis (quality included).
typealias HDFFast    = HDFLevels<8u, 2u, 32u, 0u, 8u, 2u>; // HDF's own geometry, depth 2: better KL AND faster than HDF -- recommended default
typealias HDFFastest = HDFLevels<4u, 3u, 32u, 0u, 8u, 2u>; // HDFG4L3, depth 2: fastest, small (~6%) KL cost vs HDF
