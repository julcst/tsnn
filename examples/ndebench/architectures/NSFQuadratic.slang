import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;
import architectures.IArchitecture;

import TSNN.Encodings.Compose;
import TSNN.Encodings.OneBlob;
import TSNN.Flows.Bijection;
import TSNN.Flows.PiecewiseQuadratic;
import TSNN.Flows.Prior;

/// NSF-Quadratic (L=2 coupling layers, content.tex's "we limit all normalizing
/// flows to the minimum 2 layers"): a 2-layer coupling normalizing flow on
/// [-0.5,0.5]^2 with a uniform prior, using Muller et al. 2019's
/// piecewise-quadratic coupling transform (TSNN.Flows.PiecewiseQuadratic).
///
/// The coupling coordinate IS one-blob encoded (32 bins) -- see NSFLinear.slang's
/// doc comment for the full explanation (this file previously carried the same
/// incorrect claim NSFLinear.slang's did, now corrected the same way; see
/// FINDING.md).
///
/// See NSFLinear.slang for the shared layer-ordering rationale (mirrors
/// Source/RenderPasses/NISTracer/NIS.slang's evalLogQ/sampleFlow pairing, with
/// the conditioning coordinate always fed no_diff/detached).
struct NSFQuadratic : IArchitecture
{
    static constexpr const uint kNumBins = 16; // K
    static constexpr const float kDomain = 0.5f;
    static constexpr const uint kContextDim = 1u; // must match IArchitecture.Context's CoopVec width
    static constexpr const uint kSplineOut = 2u * kNumBins + 1u; // [W(K) | V(K+1)]
    static constexpr const uint kOneBlobBins = 32u; // matches NIS.slang's NIS_ONEBLOB_BINS default

    typealias CondMLP = MLP<kContextDim + kOneBlobBins, 32, 3, kSplineOut>;
    typealias Activation = LeakyReLU;

    static const UniformPrior kPrior = UniformPrior(kDomain);

    // Base indices into the host-supplied `Layout` buffer (see
    // IArchitecture.slang and TSNN.Utils.MLP's `__init(Layout, uint)`
    // overload's doc comment): each MLP contributes DEPTH+1 = 4 entries,
    // chained in declaration order, mirroring benchmark.py's own layout
    // builder for this architecture.
    static constexpr const uint kMLP0Base = 0u;
    static constexpr const uint kMLP1Base = kMLP0Base + 4u;

    static CondMLP makeMLP0(no_diff Layout layout) { return CondMLP(layout, kMLP0Base); }
    static CondMLP makeMLP1(no_diff Layout layout) { return CondMLP(layout, kMLP1Base); }

    static uint getParamCount(no_diff Layout layout) {
        // `end` is a byte offset in the fp16 parameter layout; both MLPs are
        // chained into one buffer, so the second MLP's end covers both blocks.
        return makeMLP1(layout).end / uint(sizeof(half));
    }

    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, no_diff Layout layout, uint tid, uint threadCount)
    {
        makeMLP0(layout).initParams<TMirror, TMaster>(params, master, tid, threadCount);
        makeMLP1(layout).initParams<TMirror, TMaster>(params, master, tid, threadCount);
    }

    // [ctx | oneBlob(coord)]: see NSFLinear.slang's condInput for the full
    // rationale; identical here.
    [Differentiable]
    static CoopVec<half, kContextDim + kOneBlobBins> condInput(Context ctx, no_diff float coord)
    {
        let u = coord / (2.0f * kDomain) + 0.5f;
        let encFloats = CoopVecToArray(no_diff encodeOneBlob<half, int(kOneBlobBins)>(u));
        half enc[kOneBlobBins];
        [ForceUnroll]
        for (uint i = 0; i < kOneBlobBins; ++i)
            enc[i] = half(encFloats[i]);
        return ComposeCoopVecPlus<kContextDim, kOneBlobBins>(ctx, enc);
    }

    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float2 x)
    {
        float z0 = x.x - 0.5f;
        float z1 = x.y - 0.5f;
        float ld = 0.0f;
        {
            let p = makeMLP0(layout).forward<Activation>(weights, condInput(ctx, z0));
            let r = pqForward<kNumBins>(z1, p, kDomain);
            z1 = r.value; ld += r.logAbsDet;
        }
        {
            let p = makeMLP1(layout).forward<Activation>(weights, condInput(ctx, z1));
            let r = pqForward<kNumBins>(z0, p, kDomain);
            z0 = r.value; ld += r.logAbsDet;
        }
        return kPrior.logProb(float2(z0, z1)) + ld;
    }

    static float2 sample(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, inout PCG32 rand)
    {
        float2 u01 = float2(rand.nextFloat(), rand.nextFloat());
        let z = kPrior.sample(u01);
        float x0 = z.x, x1 = z.y;
        {
            let p = makeMLP1(layout).forward<Activation>(weights, condInput(ctx, x1));
            let r = pqInverse<kNumBins>(x0, p, kDomain);
            x0 = r.value;
        }
        {
            let p = makeMLP0(layout).forward<Activation>(weights, condInput(ctx, x0));
            let r = pqInverse<kNumBins>(x1, p, kDomain);
            x1 = r.value;
        }
        return float2(x0, x1) + 0.5f;
    }
}
