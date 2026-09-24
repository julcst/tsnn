import TSNN.Utils.MLP;
import TSNN.LinearOps;
import TSNN.Activations;
import TSNN.CoopVec;
import TSNN.Utils.PCG32;
import TSNN.Encodings.TriangleWave;
import architectures.IArchitecture;

import TSNN.Encodings.Compose;

/// DF-L (Figueiredo, He, Kalantari, Kanyuk, EGSR 2025, arXiv:2506.00839, Eq. 7-8):
/// same X-then-Y factorization and triangle-wave conditioning as DFN, but each 1D
/// marginal/conditional is a piecewise-LINEAR density (M bin-center knots, v[i] =
/// M*softmax(logits)[i]) instead of a piecewise-constant histogram, giving a C0 PDF
/// sampled by inverse-CDF instead of a discrete bin draw.
///
/// Boundary treatment: v is looked up by clamping the interpolation index to
/// [0, M-1], which makes the two outermost half-width segments constant (= v[0]
/// resp. v[M-1]) automatically -- matching Eq. 7's clamped/nearest-neighbor
/// treatment. (The paper wraps ε1/φ periodically since it is a spherical azimuth;
/// this benchmark has no spherical semantics, so both axes use the same clamped
/// treatment for simplicity.)
///
/// Generic over per-axis bin count K, same rationale as DFN's own DFNImpl<K>: K is
/// directly comparable to NSFLinear's kNumBins=16 (DFL16 below matches it).
struct DFLImpl<let K : uint> : IArchitecture
{
    // Raw-conditioner-output clamp, matching TSNN.Mixtures.Histogram's
    // kHistMaxLogit / TSNN.Flows.{PiecewiseLinear,PiecewiseQuadratic,RQS}'s
    // own kXMaxRaw: DFL inlines its own softmax decode (see evalLogDensity1D's
    // doc comment for why) instead of routing through Histogram, so it never
    // picked up the clamp added there after DF-N's runaway-conditioner-MLP
    // divergence (FINDING.md). DFL shares the exact same XNet/YNet
    // conditioner structure that caused that divergence, and this file's own
    // `log(max(density, 1e-30f))` floor has the identical NaN-silently-
    // becomes-finite failure mode on this backend's `max` if a raw logit ever
    // reaches +-inf. Clamping here before any exp/softmax math closes that
    // gap the same way it was closed everywhere else.
    static const float kDFLMaxRaw = 15.0f;

    static constexpr const uint kX = K; // x resolution
    static constexpr const uint kY = K; // y resolution
    static constexpr const uint kEncodedDim = 12; // TriangleWave encoding
    static constexpr const uint kContextDim = 1u; // must match IArchitecture.Context's CoopVec width

    typealias XNet = MLP<kContextDim, kX, 3, kX>;
    typealias YNet = MLP<kContextDim + kEncodedDim, kY, 3, kY>;
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
        return CoopVecToArray(XNet.forward<Activation>(weights, layout, kXBase, ctx));
    }

    [Differentiable]
    static float[kY] forwardY(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float x) {
        // See DFN.forwardY: ComposeCoopVecPlus takes ctx as a CoopVec directly
        // and a no_diff half[] tail; the triangle-wave CoopVec is routed through
        // CoopVecToArray first (safe: x is no_diff data, not a learnable feature).
        let encFloats = CoopVecToArray(no_diff encodeTriangleWave<half, kEncodedDim>(x));
        half enc[kEncodedDim];
        [ForceUnroll]
        for (uint i = 0; i < kEncodedDim; ++i)
            enc[i] = half(encFloats[i]);

        let input = ComposeCoopVecPlus<kContextDim, kEncodedDim>(ctx, enc);
        let params = CoopVecToArray(YNet.forward<Activation>(weights, layout, kYBase, input));
        return params;
    }

    // Piecewise-linear log-density at eps in [0,1) given M unnormalised bin
    // logits.  v[i] = M*softmax(logits)[i] are the bin-center densities (Eq. 8);
    // sum(v)/M == 1 for both the interior trapezoids and the two clamped
    // boundary half-segments, so v can be read directly as a density -- no
    // separate Jacobian term is needed (unlike DFN's histogram, whose
    // log-probability sums to 1 over discrete bins rather than over the
    // continuous domain).
    //
    // Inlined (not factored into a helper returning v[M]) for the same reason
    // RQS/PiecewiseLinear/PiecewiseQuadratic inline their softmax decode: this
    // Slang autodiff backend has been observed to miscompile array/struct
    // results threaded back through a differentiable call boundary.
    [Differentiable]
    static float evalLogDensity1D<let M : uint>(float logits[M], no_diff float eps)
    {
        float clamped[M];
        [ForceUnroll] for (uint i = 0; i < M; ++i) clamped[i] = clamp(logits[i], -kDFLMaxRaw, kDFLMaxRaw);

        float maxL = -1.0#INF;
        [ForceUnroll] for (uint i = 0; i < M; ++i) maxL = max(maxL, clamped[i]);
        float sumExp = 0.0f;
        float p[M];
        [ForceUnroll] for (uint i = 0; i < M; ++i) { p[i] = exp(clamped[i] - maxL); sumExp += p[i]; }
        float scale = float(M) / sumExp;
        float v[M];
        [ForceUnroll] for (uint i = 0; i < M; ++i) v[i] = p[i] * scale;

        float m = eps * float(M) - 0.5f;
        float mf = no_diff floor(m);
        float frac = m - mf;
        int i0 = int(clamp(mf, 0.0f, float(M - 1u)));
        int i1 = int(clamp(mf + 1.0f, 0.0f, float(M - 1u)));
        float density = lerp(v[i0], v[i1], frac);
        return log(max(density, 1e-30f));
    }

    [Differentiable]
    static float evalLogPDF(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, no_diff float2 x) {
        let histXLogits = forwardX(weights, layout, ctx);
        var logPDF = evalLogDensity1D<kX>(histXLogits, x.x);

        let histYLogits = forwardY(weights, layout, ctx, x.x);
        logPDF += evalLogDensity1D<kY>(histYLogits, x.y);

        return logPDF;
    }

    // Non-differentiable v[M] = M*softmax(logits) computation shared by sample1D's
    // inverse-CDF; unlike evalLogDensity1D this is never threaded through
    // autodiff, so it is safe to factor out.
    static float[M] computeV<let M : uint>(float logits[M])
    {
        float clamped[M];
        for (uint i = 0; i < M; ++i) clamped[i] = clamp(logits[i], -kDFLMaxRaw, kDFLMaxRaw);

        float maxL = -1.0#INF;
        for (uint i = 0; i < M; ++i) maxL = max(maxL, clamped[i]);
        float sumExp = 0.0f;
        float p[M];
        for (uint i = 0; i < M; ++i) { p[i] = exp(clamped[i] - maxL); sumExp += p[i]; }
        float scale = float(M) / sumExp;
        float v[M];
        for (uint i = 0; i < M; ++i) v[i] = p[i] * scale;
        return v;
    }

    // Inverse-CDF sample of the piecewise-linear density defined by v[M]
    // (M-1 interior trapezoids of width 1/M between bin centers, plus two
    // half-width boundary segments of constant density v[0]/v[M-1]).  The
    // interior case solves the same monotone quadratic
    // 0.5*dv*t^2 + vLo*t - rhs == 0 as PiecewiseQuadratic.slang's pqInverse,
    // reindexed to bin-center knots instead of bin-edge knots.
    static float sample1D<let M : uint>(float v[M], float u)
    {
        float invM = 1.0f / float(M);
        float half_ = 0.5f * invM;

        // Boundary segment before c_0 = 0.5/M: x in [0, c_0], density v[0].
        float f0 = v[0] * half_;
        if (u < f0)
            return u / max(v[0], 1e-20f);

        float cum = f0;
        float x0 = half_; // c_0

        for (uint i = 0; i < M - 1u; ++i)
        {
            float vLo = v[i];
            float vHi = v[i + 1u];
            float dv  = vHi - vLo;
            float seg = (vLo + vHi) * half_;
            if (u < cum + seg)
            {
                float rhs = (u - cum) * float(M);
                float t;
                if (abs(dv) < 1e-6f)
                    t = rhs / max(vLo, 1e-20f);
                else
                {
                    float disc = max(vLo * vLo + 2.0f * dv * rhs, 0.0f);
                    t = (-vLo + sqrt(disc)) / dv;
                }
                t = clamp(t, 0.0f, 1.0f);
                return x0 + t * invM;
            }
            cum += seg;
            x0 += invM;
        }

        // Boundary segment after c_{M-1}: x in [c_{M-1}, 1], density v[M-1].
        // x0 == c_{M-1} here (half_ + (M-1)*invM).
        float rem = max(u - cum, 0.0f);
        return x0 + rem / max(v[M - 1u], 1e-20f);
    }

    static float2 sample(MatrixBiasBuffer weights, no_diff Layout layout, Context ctx, inout PCG32 rand) {
        let histXLogits = forwardX(weights, layout, ctx);
        let vX = computeV<kX>(histXLogits);
        let x = sample1D<kX>(vX, rand.nextFloat());

        let histYLogits = forwardY(weights, layout, ctx, x);
        let vY = computeV<kY>(histYLogits);
        let y = sample1D<kY>(vY, rand.nextFloat());

        return float2(x, y);
    }
}

typealias DFL = DFLImpl<32u>;
// Bin-matched to NSFLinear's kNumBins=16 -- see DFN16's own comment in DFN.slang.
typealias DFL16 = DFLImpl<16u>;
