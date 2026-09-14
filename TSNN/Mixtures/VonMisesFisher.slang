// TSNN/Mixtures/VonMisesFisher.slang
// Differentiable von Mises-Fisher (vMF) directional density: single-lobe
// primitives (normalizer, full-sphere pdf, own-mean-cap pdf, general-axis
// hemisphere mass), sampling, and an N-lobe mixture with the call shape of
// TSNN/Mixtures/TruncatedGMM.slang's TruncatedGMM<N> ([Differentiable] eval +
// non-differentiable sample).
//
// Generalizes the vMF math that previously lived only ad hoc in
// scripts/nisfit/NpmNetwork.slang (single-lobe normalizer, two branches) and
// scripts/nisfit/SampleUtils.slang (sampleVmf) into a reusable, tested TSNN
// module. Neither nisfit file is touched here -- nisfit's copies stay
// byte-identical so its published RESULTS.md numbers stay reproducible (see
// NpmNetwork.slang's own module comment); arms forked under
// scripts/nisbench/ import this module instead of carrying their own copy.
//
// Numerics reuse TSNN/Mixtures/GaussianHelpers.slang's log1mexp/log1p/expm1
// rather than re-deriving stable log-space identities, and
// TSNN/Mixtures/TruncatedGMM.slang's logSumExp/sampleSoftmax (same import
// DiscretizedGMM.slang already makes for the same reason).
//
// API:
//   vmfLogNormalizer(kappa)                              [Differentiable]
//   vmfLogPdf(mean, kappa, dir)                           [Differentiable]
//   vmfLogPdfCap(mean, kappa, dir, cosCap)                [Differentiable]
//   vmfHalfspaceLogMass(mean, kappa, axis)                 approximate, no grad
//   sampleVmf(mean, kappa, xi)
//   sampleVmfCap(mean, kappa, cosCap, xi)
//   VonMisesFisherMixture<N>{ eval<T>(params, dir) [Differentiable], sample<T>(params, rand) }
//
// `mean`/`dir`/`axis` are assumed unit vectors throughout; nothing here
// renormalizes them.

import TSNN.Mixtures.GaussianHelpers;
import TSNN.Mixtures.SoftmaxHelpers;
import TSNN.Utils.PCG32;

static const float kPi         = 3.141592653589793f;
static const float kTwoPi      = 6.283185307179586f;
static const float kLogTwoPi   = 1.8378770664093453f;
static const float kLogFourPi  = 2.5310242469692907f;

// Series-vs-exact crossover for both vmfLogNormalizer and vmfLogCapMass below.
// Below this, the exact formulas subtract two quantities that individually
// diverge to -inf as kappa -> 0 (see each function's comment); above it, the
// exact formula is used directly. See tests/vmf/test_vmf.py for the measured
// max relative error this threshold produces against a reference computed in
// double precision.
static const float kVmfSmallKappa = 1e-3f;

// ─── Single-lobe log-normalizer ──────────────────────────────────────────────
//
// C(kappa) = kappa / (4*pi*sinh(kappa)), the full-sphere vMF normalizing
// constant (kappa -> 0 reduces to the uniform-sphere density 1/(4*pi)).
//
// log(sinh(kappa)) = kappa - log(2) + log1mexp(-2*kappa) is exact and stable
// for all kappa >= 0: log1mexp (GaussianHelpers.slang) never forms the
// possibly-tiny 1-exp(-2*kappa) directly, and for large kappa exp(-2*kappa)
// underflows cleanly to 0 (log1mexp(very negative) -> 0), leaving
// log(sinh(kappa)) ~= kappa - log(2) with no overflow.
//
// log(kappa) - logSinh(kappa) itself loses precision to cancellation as
// kappa -> 0 (both terms -> -inf while their difference -> 0), so below
// kVmfSmallKappa this uses the series log(kappa/sinh(kappa))
//   = -kappa^2/6 + kappa^4/180 - O(kappa^6),
// truncated to its leading term (the kappa^4 term is ~5.6e-15 at
// kappa=kVmfSmallKappa, far below float32 precision). The two branches share
// the same first-order derivative in kappa (d/dkappa[log(kappa/sinh(kappa))]
// = 1/kappa - coth(kappa) = -kappa/3 + O(kappa^3), matching the series
// branch's -kappa/3 exactly to that order), so the crossover doesn't
// introduce a gradient discontinuity FD checks would see.
[Differentiable]
float vmfLogNormalizer(float kappa)
{
    if (kappa < kVmfSmallKappa)
        return -kLogFourPi - kappa * kappa / 6.0f;
    let logSinh = kappa - 0.6931471805599453f + log1mexp(-2.0f * kappa);
    return log(kappa) - kLogFourPi - logSinh;
}

// ─── Full-sphere log-pdf ──────────────────────────────────────────────────────

[Differentiable]
float vmfLogPdf(float3 mean, float kappa, no_diff float3 dir)
{
    return kappa * dot(mean, dir) + vmfLogNormalizer(kappa);
}

// ─── Cap-about-own-mean log-pdf ───────────────────────────────────────────────
//
// log P(dot(mean,dir) >= cosCap) for a vMF(mean, kappa), i.e. the mass in the
// cap centered on the distribution's OWN mean. Exact:
//   mass = (e^kappa - e^{kappa*cosCap}) / (e^kappa - e^{-kappa})
// Factor e^kappa out of both the numerator and denominator so the only
// exponentials formed are e^{kappa*(cosCap-1)} and e^{-2*kappa}, both in
// (0,1] for kappa>=0, cosCap<=1 -- never e^kappa itself (overflows float32
// past kappa~88):
//   log(mass) = log1mexp(kappa*(cosCap-1)) - log1mexp(-2*kappa)
// Both log1mexp arguments are individually <= 0, so this is the direct
// analogue of vmfLogNormalizer's logSinh identity above, and inherits the
// same kappa -> 0 cancellation problem (both terms -> -inf, in particular
// exactly -inf at kappa==0) -- handled the same way, with a first-order
// series so the branches agree to O(kappa) at the crossover:
//   mass ~= (1-cosCap)/2 * (1 + kappa*(1+cosCap)/2) + O(kappa^2)
//   log(mass) ~= log((1-cosCap)/2) + kappa*(1+cosCap)/2 + O(kappa^2)
// (the (1-cosCap)/2 term is exactly the uniform-sphere cap-area fraction, as
// expected in the kappa->0 limit).
[Differentiable]
float vmfLogCapMass(float kappa, no_diff float cosCap)
{
    if (kappa < kVmfSmallKappa)
        return log(max(0.5f * (1.0f - cosCap), 1e-30f)) + kappa * 0.5f * (1.0f + cosCap);
    return log1mexp(kappa * (cosCap - 1.0f)) - log1mexp(-2.0f * kappa);
}

// Large finite sentinel (never NaN/-inf) for directions outside the cap.
static const float kVmfCapSentinel = -1e30f;

[Differentiable]
float vmfLogPdfCap(float3 mean, float kappa, no_diff float3 dir, no_diff float cosCap)
{
    let w = dot(mean, dir);
    if (w < cosCap)
        return kVmfCapSentinel;
    return kappa * w + vmfLogNormalizer(kappa) - vmfLogCapMass(kappa, cosCap);
}

// ─── General-axis hemisphere mass (approximate) ──────────────────────────────
//
// log P(dot(axis,dir) >= 0) for a vMF(mean, kappa) where `axis` is NOT
// necessarily `mean`. Unlike the own-mean cap above, this has no elementary
// closed form (a non-central spherical-cap integral) -- reduced instead to a
// 1-D integral over the polar angle theta measured from `axis`, by
// integrating the azimuthal angle phi (also measured around `axis`)
// analytically: writing gamma = angle(mean,axis), the standard spherical law
// of cosines gives dot(mean,dir) = cos(gamma)*cos(theta) +
// sin(gamma)*sin(theta)*cos(phi), so
//   integral_0^{2pi} exp(kappa*sin(gamma)*sin(theta)*cos(phi)) dphi
//     = 2*pi*I0(kappa*sin(gamma)*sin(theta))
// where I0 is the order-0 modified Bessel function of the first kind. What
// remains,
//   mass = integral_0^{pi/2} 2*pi*C(kappa)*exp(kappa*cos(gamma)*cos(theta))
//              * I0(kappa*sin(gamma)*sin(theta)) * sin(theta) dtheta,
// is a smooth 1-D integral, evaluated with a fixed 16-point (8 symmetric
// pairs) Gauss-Legendre rule mapped from [-1,1] to [0,pi/2].
//
// APPROXIMATE: a fixed low-order rule resolves this fine at moderate kappa,
// but the integrand's boundary layer sharpens as kappa grows, and a fixed
// order eventually can't resolve an arbitrarily narrow peak. See
// tests/vmf/test_vmf.py for the measured max relative error against a dense
// numpy reference quadrature over the kappa range this benchmark uses. Not
// [Differentiable] -- it is a diagnostic/reference utility, not part of any
// trained model's forward path.
float besselI0e(float x)
{
    // Abramowitz & Stegun 9.8.1/9.8.2 rational approximation to I0, evaluated
    // in the exponentially-scaled form I0e(x) = exp(-x)*I0(x) so the result
    // stays bounded in (0,1] for arbitrarily large x (I0(x) itself overflows
    // float32 past x~88). Max relative error ~1.6e-7 (x<3.75) / ~1.9e-7
    // (x>=3.75) per A&S.
    if (x < 3.75f)
    {
        let t = x / 3.75f;
        let t2 = t * t;
        let i0 = 1.0f + t2 * (3.5156229f + t2 * (3.0899424f + t2 * (1.2067492f
            + t2 * (0.2659732f + t2 * (0.0360768f + t2 * 0.0045813f)))));
        return i0 * exp(-x);
    }
    let t = 3.75f / x;
    let poly = 0.39894228f + t * (0.01328592f + t * (0.00225319f + t * (-0.00157565f
        + t * (0.00916281f + t * (-0.02057706f + t * (0.02635537f + t * (-0.01647633f
        + t * 0.00392377f)))))));
    return poly / sqrt(x);
}

static const uint kGLOrder = 8u; // 8 symmetric pairs -> 16-point Gauss-Legendre
static const float kGLNode[kGLOrder] = {
    0.0950125098376374f, 0.2816035507792589f, 0.4580167776572274f, 0.6178762444026438f,
    0.7554044083550030f, 0.8656312023878318f, 0.9445750230732326f, 0.9894009349916499f,
};
static const float kGLWeight[kGLOrder] = {
    0.1894506104550685f, 0.1826034150449236f, 0.1691565193950025f, 0.1495959888165767f,
    0.1246289712555339f, 0.0951585116824928f, 0.0622535239386479f, 0.0271524594117541f,
};

float vmfHalfspaceLogMass(float3 mean, float kappa, float3 axis)
{
    if (kappa < kVmfSmallKappa)
        return log(0.5f); // uniform sphere: every hemisphere holds half the mass

    let cosGamma = clamp(dot(mean, axis), -1.0f, 1.0f);
    let sinGamma = sqrt(max(0.0f, 1.0f - cosGamma * cosGamma));
    let logC = vmfLogNormalizer(kappa); // log(kappa/(4*pi*sinh(kappa)))
    let quarterPi = kPi * 0.25f;

    float logTerms[2u * kGLOrder];
    [ForceUnroll]
    for (uint k = 0; k < kGLOrder; ++k)
    {
        [ForceUnroll]
        for (uint s = 0; s < 2u; ++s)
        {
            let x = (s == 0u) ? -kGLNode[k] : kGLNode[k];
            let theta = quarterPi * (1.0f + x); // map [-1,1] -> [0, pi/2]
            float sinT, cosT;
            sincos(theta, sinT, cosT);
            let radial = kappa * sinGamma * sinT;
            let axial  = kappa * cosGamma * cosT;
            // exp(axial)*I0(radial) == exp(axial+radial)*I0e(radial)
            let logIntegrand = logC + kLogTwoPi + axial + radial
                + log(max(besselI0e(radial), 1e-30f)) + log(max(sinT, 1e-30f));
            logTerms[2u * k + s] = logIntegrand + log(kGLWeight[k] * quarterPi);
        }
    }
    return logSumExp(logTerms);
}

// ─── Sampling ──────────────────────────────────────────────────────────────────
//
// Exact closed-form vMF sampling on the 3-sphere (Wood 1994 specialises to a
// direct inverse-CDF in 3D, no rejection needed): the marginal of
// w = dot(dir,mean) has pdf kappa/(2*sinh(kappa)) * exp(kappa*w) on [-1,1],
// whose CDF inverts in closed form. Ported from
// scripts/nisfit/SampleUtils.slang's sampleVmf (left untouched there so
// nisfit's published results stay reproducible -- see this file's header).
float3 sampleVmf(float3 mean, float kappa, float2 xi)
{
    float w;
    if (kappa < 1e-4f)
        w = 1.0f - 2.0f * xi.x;
    else
        w = 1.0f + log(xi.x + (1.0f - xi.x) * exp(-2.0f * kappa)) / kappa;
    w = clamp(w, -1.0f, 1.0f);
    return vmfBasisFromW(mean, w, xi.y);
}

// Own-mean-cap counterpart, consistent with vmfLogPdfCap: inverse-CDF over the
// truncated range w in [cosCap, 1] rather than rejection sampling (which
// would need matching acceptance-probability bookkeeping a closed-form
// inverse-CDF makes unnecessary). Derived the same way as sampleVmf above --
// dividing the cap CDF through by e^kappa so the only exponential formed is
// e^{kappa*(cosCap-1)} in (0,1], matching vmfLogCapMass's identity:
//   F(w) = (e^{kappa*w} - e^{kappa*cosCap}) / (e^kappa - e^{kappa*cosCap})
//        = (e^{kappa*(w-1)} - e^{kappa*(cosCap-1)}) / (1 - e^{kappa*(cosCap-1)})
// inverted for w given xi.x = F(w).
float3 sampleVmfCap(float3 mean, float kappa, float cosCap, float2 xi)
{
    float w;
    if (kappa < 1e-4f)
        w = cosCap + xi.x * (1.0f - cosCap); // uniform-in-w limit
    else
    {
        let e = exp(kappa * (cosCap - 1.0f)); // in (0,1]
        w = 1.0f + log(e * (1.0f - xi.x) + xi.x) / kappa;
    }
    w = clamp(w, cosCap, 1.0f);
    return vmfBasisFromW(mean, w, xi.y);
}

// Shared basis build: given w=dot(dir,mean) and a second uniform for the
// azimuth, complete the direction using Duff et al. 2017's branchless
// orthonormal basis around `mean`.
float3 vmfBasisFromW(float3 mean, float w, float xiAzimuth)
{
    let radial = sqrt(max(0.0f, 1.0f - w * w));
    let phi = kTwoPi * xiAzimuth;
    let local = float3(radial * cos(phi), radial * sin(phi), w);

    let sign = mean.z >= 0.0f ? 1.0f : -1.0f;
    let a = -1.0f / (sign + mean.z);
    let b = mean.x * mean.y * a;
    let t = float3(1.0f + sign * mean.x * mean.x * a, sign * b, -sign * mean.x);
    let bt = float3(b, sign + mean.y * mean.y * a, -mean.y);
    return t * local.x + bt * local.y + mean * local.z;
}

// ─── Per-lobe raw-parameter decode helpers ─────────────────────────────────────

// Exponential concentration parameterisation: zero raw output starts every
// lobe at kappa=1 with a non-vanishing gradient. Replicated from
// scripts/nisfit/NpmNetwork.slang's npmConcentration -- the original
// softplus-with-large-negative-shift parametrisation it replaced had
// d(kappa)/d(raw) ~= 3.4e-4 at initialisation and clamped every trained lobe
// to kappa <= 12; the upper clamp here keeps log-sinh and the fp16 backward
// path in range.
[Differentiable]
float vmfConcentration(float raw)
{
    return exp(clamp(raw, -12.0f, 7.0f));
}

[Differentiable]
float vmfSigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

// Non-differentiable: only ever applied to constants derived from a lobe
// index, to build the additive bias directionBias() returns.
float vmfLogit(float x)
{
    let p = clamp(x, 1e-4f, 1.0f - 1e-4f);
    return log(p) - log(1.0f - p);
}

// ─── N-lobe mixture ────────────────────────────────────────────────────────────
//
// STRIDE=4 packing [logWeight, kappaRaw, thetaRaw, phiRaw] per lobe, matching
// scripts/nisfit/NpmNetwork.slang's existing per-lobe layout so a forked arm's
// conditioner head can feed this struct unchanged. eval() returns the
// SOLID-ANGLE log-density (weighted logsumexp of per-lobe vmfLogPdf); unlike
// NpmNetwork.slang's forward() (which bakes in a "+log(4*pi)" UV-measure
// correction, NpmNetwork.slang:156-159), that term is deliberately left to
// the caller -- this module has no opinion on whether its result is consumed
// in solid-angle or uv measure.
struct VonMisesFisherMixture<let N : uint>
{
    static const uint STRIDE = 4u;
    typealias Output = float3;

    // Deterministic per-lobe direction bias: a Fibonacci sphere spread so
    // zero-initialised lobes don't all start pointing the same way.
    // Replicated (not imported -- see this file's header) from
    // scripts/nisfit/NpmNetwork.slang's npmDirectionBias/npmLogit.
    void directionBias(no_diff uint i, out float thetaBias, out float phiBias)
    {
        let y = (float(i) + 0.5f) / float(N);
        let theta01 = acos(1.0f - 2.0f * y) / kPi;
        let phi01 = frac(float(i) * 0.6180339887498948f + 0.5f / float(N));
        thetaBias = vmfLogit(theta01);
        phiBias = vmfLogit(phi01);
    }

    [Differentiable]
    float eval<T : IFloat>(T params[N * STRIDE], no_diff float3 dir)
    {
        float logits[N];
        float logWeights[N];
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
        {
            let base = i * STRIDE;
            let logWeight = params[base + 0].toFloat();
            let kappa = vmfConcentration(params[base + 1].toFloat());
            float thetaBias, phiBias;
            directionBias(i, thetaBias, phiBias);
            let theta = kPi * vmfSigmoid(params[base + 2].toFloat() + thetaBias);
            let phi   = kTwoPi * vmfSigmoid(params[base + 3].toFloat() + phiBias);
            let mean  = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            logits[i] = logWeight + vmfLogPdf(mean, kappa, dir);
            logWeights[i] = logWeight;
        }
        return logSumExp(logits) - logSumExp(logWeights);
    }

    Output sample<T : IFloat>(T params[N * STRIDE], inout PCG32 rand)
    {
        float logWeights[N];
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
            logWeights[i] = params[i * STRIDE].toFloat();
        let bin = sampleSoftmax(logWeights, rand.nextFloat());

        let base = bin * STRIDE;
        let kappa = vmfConcentration(params[base + 1].toFloat());
        float thetaBias, phiBias;
        directionBias(bin, thetaBias, phiBias);
        let theta = kPi * vmfSigmoid(params[base + 2].toFloat() + thetaBias);
        let phi   = kTwoPi * vmfSigmoid(params[base + 3].toFloat() + phiBias);
        let mean  = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        return sampleVmf(mean, kappa, float2(rand.nextFloat(), rand.nextFloat()));
    }
}
