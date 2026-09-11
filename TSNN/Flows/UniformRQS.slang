// TSNN/Flows/UniformRQS.slang
// Rational-quadratic spline CDF over FIXED, uniform-width bins.
//
// Not a coupling transform (RQS.slang/CircularRQS.slang bend a domain onto itself for a
// flow layer). This is a monotone 1-D CONDITIONAL DENSITY read directly: x lives on
// [-0.5, 0.5], split into K bins of fixed width 1/K each (no learned widths), and the
// spline's "y" axis is the CUMULATIVE PROBABILITY F(x) spanning [0, 1] exactly. Since X's
// total measure is 1 (width 1) and y is a genuine CDF, dy/dx IS the density in unit measure
// — so `logAbsDet` from the same per-bin formula RQS.slang uses is exactly log(pdf), with
// no extra normalising term.
//
// Parametrization (K bins, K+1 knots):
//   heights  p[i] = minH + (1 - minH*K) * softmax(h)[i]         -- DF's existing bin vector;
//            cumsum(p) is the CDF at each knot boundary. delta[i] = p[i]*K is DF-N's
//            piecewise-constant density value for bin i (average density over that bin).
//   knot derivatives d[0..K] (dy/dx at each knot), two modes:
//     derived — d[i] = 2*delta[i-1]*delta[i] / (delta[i-1]+delta[i]), the harmonic mean of
//               the two bins meeting at knot i (always positive since delta > 0 everywhere,
//               so RQS monotonicity holds and no extra parameters are needed).
//               Boundary: BOUNDED uses one-sided d[0]=delta[0], d[K]=delta[K-1] (NOT
//               RQS.slang's fixed identity slope, which would pin density to 1 at the poles
//               regardless of what the bins actually say); CIRCULAR wraps the index instead,
//               so d[0] is the harmonic mean across the seam (bin K-1, bin 0) and d[K]=d[0].
//     learned — d[i] = kMinDerivative + softplus(raw + kDerivativeUnnorm), reusing
//               CircularRQS.slang's zero-init offset so raw==0 decodes to slope 1. BOUNDED
//               reads K+1 raw values (every knot free); CIRCULAR reads only K (knot K
//               reuses knot 0, same convention as CircularRQS.slang).
//
// `d[K+1]` is always the parameter shape: BOUNDED reads every slot, CIRCULAR reads only
// [0, K) and ignores slot K. In `derived` mode the caller's `d` values are ignored entirely
// (the derivatives come from `h` alone) — pass anything, e.g. zeros.
//
// API:
//   urqsLogPdf<K>(eps, h[K], d[K+1], circular, learnedDeriv) -> log density   [Differentiable]
//   urqsInverse<K>(xi, h[K], d[K+1], circular, learnedDeriv) -> x             (sampling only)
//   urqsCdf<K>(eps, h[K], d[K+1], circular, learnedDeriv)    -> F(x)          [Differentiable, tests only]
//
// `circular`/`learnedDeriv` are `no_diff` compile-time-constant-folded flags (callers pass
// `static constexpr` bools) — branches inside one body, not a helper-function boundary, so
// they do not trip the inlining constraint below.
//
// Shared softmax and cumulative-sum array operations are provided by TSNN.CoopVec.
// The remaining decode work stays local to each entry point.

import TSNN.CoopVec;
import TSNN.Flows.Bijection;

static const float kURQSMinBinHeight  = 1e-3f;
static const float kURQSMinDerivative = 1e-3f;
// softplus_inv(1 - kURQSMinDerivative) ≈ 0.5413 -- same offset RQS.slang/CircularRQS.slang
// use so a zero-initialised raw derivative decodes to slope 1.
static const float kURQSDerivativeUnnorm = 0.5413f;

[Differentiable]
float urqsSoftplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

[Differentiable]
float urqsDerivative(float raw)
{
    return kURQSMinDerivative + urqsSoftplus(raw + kURQSDerivativeUnnorm);
}

[Differentiable]
float urqsHarmonicMean(float a, float b)
{
    // a, b are always > 0 (delta = p*K, p floored above 0 by kURQSMinBinHeight), so the sum
    // is always > 0: no epsilon guard needed against a genuine 0/0.
    return 2.0f * a * b / (a + b);
}

// ─── log-density (forward, [Differentiable]) ──────────────────────────────────────────────

[Differentiable]
float urqsLogPdf<let K : uint>(no_diff float eps, float h[K], float d[K + 1u],
                               no_diff bool circular, no_diff bool learnedDeriv)
{
    // heights: softmax + min-height floor. Sums to exactly 1 (K*minH + (1-K*minH)*1), so
    // cumsum(p) is a genuine CDF from 0 to 1.
    float p[K];
    softmax(h, p);
    [ForceUnroll] for (uint i = 0u; i < K; ++i)
        p[i] = kURQSMinBinHeight + (1.0f - kURQSMinBinHeight * float(K)) * p[i];

    float delta[K];
    [ForceUnroll] for (uint i = 0u; i < K; ++i) delta[i] = p[i] * float(K);

    float cumH[int(K) + 1];
    cumulativeSum(p, 0.0f, 1.0f, cumH);

    float dd[K + 1u];
    if (learnedDeriv)
    {
        if (circular)
        {
            [ForceUnroll] for (uint i = 0u; i < K; ++i) dd[i] = urqsDerivative(d[i]);
            dd[K] = dd[0];
        }
        else
        {
            [ForceUnroll] for (uint i = 0u; i < K + 1u; ++i) dd[i] = urqsDerivative(d[i]);
        }
    }
    else
    {
        if (circular)
        {
            [ForceUnroll] for (uint i = 0u; i < K; ++i)
                dd[i] = urqsHarmonicMean(delta[(i + K - 1u) % K], delta[i]);
            dd[K] = dd[0];
        }
        else
        {
            dd[0] = delta[0];
            dd[K] = delta[K - 1u];
            [ForceUnroll] for (uint i = 1u; i < K; ++i)
                dd[i] = urqsHarmonicMean(delta[i - 1u], delta[i]);
        }
    }

    let x = eps - 0.5f;
    let bw = 1.0f / float(K);
    let bin = min(uint(max(no_diff floor((x + 0.5f) * float(K)), 0.0f)), K - 1u);

    float bh = p[bin];
    float dk = dd[bin];
    float dk1 = dd[bin + 1u];
    float cw = -0.5f + float(bin) * bw;
    float delta_b = delta[bin];

    float theta = clamp((x - cw) / bw, 0.0f, 1.0f);
    float t1mt = theta * (1.0f - theta);
    float denom = delta_b + (dk + dk1 - 2.0f * delta_b) * t1mt;
    float safeDenom = max(denom, 1e-20f);

    float dnumer = delta_b * delta_b * (
        dk1 * theta * theta + 2.0f * delta_b * t1mt + dk * (1.0f - theta) * (1.0f - theta)
    );
    return log(max(dnumer, 1e-30f)) - 2.0f * log(safeDenom);
}

// ─── inverse-CDF sampling (not differentiable) ────────────────────────────────────────────

float urqsInverse<let K : uint>(float xi, float h[K], float d[K + 1u],
                                bool circular, bool learnedDeriv)
{
    float p[K];
    softmax(h, p);
    [ForceUnroll] for (uint i = 0u; i < K; ++i)
        p[i] = kURQSMinBinHeight + (1.0f - kURQSMinBinHeight * float(K)) * p[i];

    float delta[K];
    [ForceUnroll] for (uint i = 0u; i < K; ++i) delta[i] = p[i] * float(K);

    float cumH[int(K) + 1];
    cumulativeSum(p, 0.0f, 1.0f, cumH);
    cumH[K] = 1.0f;  // exact, guards fp drift from the running sum

    float dd[K + 1u];
    if (learnedDeriv)
    {
        if (circular)
        {
            [ForceUnroll] for (uint i = 0u; i < K; ++i) dd[i] = urqsDerivative(d[i]);
            dd[K] = dd[0];
        }
        else
        {
            [ForceUnroll] for (uint i = 0u; i < K + 1u; ++i) dd[i] = urqsDerivative(d[i]);
        }
    }
    else
    {
        if (circular)
        {
            [ForceUnroll] for (uint i = 0u; i < K; ++i)
                dd[i] = urqsHarmonicMean(delta[(i + K - 1u) % K], delta[i]);
            dd[K] = dd[0];
        }
        else
        {
            dd[0] = delta[0];
            dd[K] = delta[K - 1u];
            [ForceUnroll] for (uint i = 1u; i < K; ++i)
                dd[i] = urqsHarmonicMean(delta[i - 1u], delta[i]);
        }
    }

    let clampedXi = clamp(xi, 0.0f, 1.0f);
    int bin = 0;
    [ForceUnroll] for (uint i = 0u; i < K; ++i) bin = (clampedXi >= cumH[i + 1u]) ? int(i + 1u) : bin;
    bin = min(bin, int(K) - 1);

    let bw = 1.0f / float(K);
    float bh = p[bin];
    float dk = dd[bin];
    float dk1 = dd[bin + 1u];
    float cw = -0.5f + float(bin) * bw;
    float ch = cumH[bin];
    float delta_b = delta[bin];
    float s = clampedXi - ch;
    float A = dk + dk1 - 2.0f * delta_b;

    float a = s * A + bh * (delta_b - dk);
    float b = bh * dk - s * A;
    float c = -delta_b * s;
    float disc = max(b * b - 4.0f * a * c, 0.0f);
    float theta = (2.0f * c) / (-b - sqrt(disc));
    theta = clamp(theta, 0.0f, 1.0f);

    let x = theta * bw + cw;
    return x + 0.5f;  // back to eps-space [0, 1]
}

// ─── CDF value (forward, [Differentiable], tests only) ────────────────────────────────────

[Differentiable]
float urqsCdf<let K : uint>(no_diff float eps, float h[K], float d[K + 1u],
                            no_diff bool circular, no_diff bool learnedDeriv)
{
    float p[K];
    softmax(h, p);
    [ForceUnroll] for (uint i = 0u; i < K; ++i)
        p[i] = kURQSMinBinHeight + (1.0f - kURQSMinBinHeight * float(K)) * p[i];

    float delta[K];
    [ForceUnroll] for (uint i = 0u; i < K; ++i) delta[i] = p[i] * float(K);

    float cumH[int(K) + 1];
    cumulativeSum(p, 0.0f, 1.0f, cumH);

    float dd[K + 1u];
    if (learnedDeriv)
    {
        if (circular)
        {
            [ForceUnroll] for (uint i = 0u; i < K; ++i) dd[i] = urqsDerivative(d[i]);
            dd[K] = dd[0];
        }
        else
        {
            [ForceUnroll] for (uint i = 0u; i < K + 1u; ++i) dd[i] = urqsDerivative(d[i]);
        }
    }
    else
    {
        if (circular)
        {
            [ForceUnroll] for (uint i = 0u; i < K; ++i)
                dd[i] = urqsHarmonicMean(delta[(i + K - 1u) % K], delta[i]);
            dd[K] = dd[0];
        }
        else
        {
            dd[0] = delta[0];
            dd[K] = delta[K - 1u];
            [ForceUnroll] for (uint i = 1u; i < K; ++i)
                dd[i] = urqsHarmonicMean(delta[i - 1u], delta[i]);
        }
    }

    let x = eps - 0.5f;
    let bw = 1.0f / float(K);
    let bin = min(uint(max(no_diff floor((x + 0.5f) * float(K)), 0.0f)), K - 1u);

    float bh = p[bin];
    float dk = dd[bin];
    float dk1 = dd[bin + 1u];
    float cw = -0.5f + float(bin) * bw;
    float ch = cumH[bin];
    float delta_b = delta[bin];

    float theta = clamp((x - cw) / bw, 0.0f, 1.0f);
    float t1mt = theta * (1.0f - theta);
    float num = bh * (delta_b * theta * theta + dk * t1mt);
    float denom = delta_b + (dk + dk1 - 2.0f * delta_b) * t1mt;
    float safeDenom = max(denom, 1e-20f);
    return ch + num / safeDenom;
}
