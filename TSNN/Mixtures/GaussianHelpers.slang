

// SPDX-License-Identifier: BSD-2-Clause
//
// Slang/SPIR-V translation of Norbert Juffa's CUDA erff approximation.
// The translation changes target-language intrinsics but retains the
// original approximation and coefficients.
//
// Copyright (c) 2016-2021, Norbert Juffa
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// Based on Norbert Juffa's BSD-2-Clause erff approximation.
// Maximum error reported for the original CUDA implementation: 1.29532 ULP.
[ForceInline]
[BackwardDerivative(erf_bwd)]
float erf(float x)
{
    // WARNING: Overflow for |x| > 4
    float ax = abs(x);
    float x2 = x * x;
    float r;

    if (ax <= 1.0f)
    {
        r = -5.58853149e-4f;
        r = fma(r, x2,  4.90735564e-3f);
        r = fma(r, x2, -2.67030653e-2f);
        r = fma(r, x2,  1.12799220e-1f);
        r = fma(r, x2, -3.76123011e-1f);
        r = fma(r, x2,  1.28379121e-1f);

        return fma(r, x, x);
    }

    r = 2.58907676e-5f;
    r = fma(r, ax, -5.64874616e-4f);

    float u = 5.66795561e-3f;
    u = fma(u, ax, -3.51759829e-2f);

    r = fma(r, x2, u);
    r = fma(r, ax, 1.54329389e-1f);
    r = fma(r, ax, 9.15674746e-1f);
    r = fma(r, ax, 6.28459513e-1f);
    r = fma(r, ax, ax);

    r = 1.0f - exp2(-r);

    // Apply x's sign without a floating-point comparison or divergent branch.
    return asfloat(asuint(r) | (asuint(x) & 0x80000000u));
}

void erf_bwd(inout DifferentialPair<float> dpx, float dOut)
{
    float x = dpx.p;
    float dx = dOut * (2.0f / sqrt(float.getPi())) * exp(-x * x);
    dpx = diffPair(x, dx);
}

// ─── erfcx-based stable log-space normal CDF ──────────────────────────────────
//
// logTruncGaussianPDF's truncation-mass term used to go through erf() alone,
// which collapses to exactly 0 (so log(0) = -inf, then NaN gradients) once the
// standardized truncation interval sits entirely in one tail -- exactly the
// regime an unconstrained MLP head eventually wanders into. Fixing that at the
// source (a numerically stable logNormalCDF) is more robust than clamping
// mean/logSigma to keep erf() away from its blind spot, which trades the
// collapse for a wide dead (zero-gradient) region instead.
//
// erfcxPositive is Norbert Juffa's BSD-2-Clause "accuracy-optimized,
// performance-neutral" erfcxf, nonnegative-input branch only -- see
// https://forums.developer.nvidia.com/t/accuracy-optimized-performance-neutral-implementation-of-erfcxf/226946
// The two `rcp.approx.ftz.f32` PTX reciprocals in the original CUDA source
// have no equivalent outside CUDA, so this uses plain division instead; the
// Newton-style residual correction that follows each one in the original
// still applies (and costs the same few FMAs) whether the reciprocal it's
// correcting was exact or approximate, so accuracy can only improve.
// Coefficients and structure are otherwise unchanged. Negative-input handling
// is dropped per Juffa's note that it roughly doubles the cost -- every call
// site here reflects into the nonnegative half first.
// Maximum error reported for the original CUDA nonnegative-input path: 1.97521 ULP.
[ForceInline]
[BackwardDerivative(erfcxPositive_bwd)]
float erfcxPositive(float a)
{
    // q = (a-2)/(a+2), mapping [0, INF) -> [-1, 1]
    float p = a + 2.0f;
    float r = 1.0f / p;
    float q = fma(-4.0f, r, 1.0f);
    float t = fma(q + 1.0f, -2.0f, a);
    float e = fma(-a, q, t);
    q = fma(r, e, q);

    // Approximate (1+2a)*exp(a*a)*erfc(a) as p(q)+1 for q in [-1,1]
    p =            5.92619181e-5f;
    p = fma(p, q,  1.61231728e-4f);
    p = fma(p, q, -3.46499495e-4f);
    p = fma(p, q, -1.39683776e-3f);
    p = fma(p, q,  1.20587996e-3f);
    p = fma(p, q,  8.69013276e-3f);
    p = fma(p, q, -8.01389851e-3f);
    p = fma(p, q, -5.42122647e-2f);
    p = fma(p, q,  1.64048553e-1f);
    p = fma(p, q, -1.66031078e-1f);
    p = fma(p, q, -9.27637145e-2f);
    p = fma(p, q,  2.76978403e-1f);

    // Divide (1+p) by (1+2a) => exp(a*a)*erfc(a) == erfcx(a)
    float d = fma(0.25f, a, 0.125f);
    r = 1.0f / d;
    q = fma(p, r, r);
    e = fma(fma(-a, q, 4.0f), 0.25f, fma(-0.125f, q, p)); // residual correction
    r = 0.125f * fma(e, r, q);

    return isinf(a) ? 0.0f : r;
}

// d/dx erfcx(x) = 2x*erfcx(x) - 2/sqrt(pi) -- exact for all real x, cheap, and
// reuses the forward value instead of differentiating the degree-11
// polynomial above (the whole point of routing through this identity).
void erfcxPositive_bwd(inout DifferentialPair<float> dpx, float dOut)
{
    float x = dpx.p;
    float value = erfcxPositive(x);
    float dx = dOut * (2.0f * x * value - 2.0f / sqrt(float.getPi()));
    dpx = diffPair(x, dx);
}

// Stable log1p/expm1/log(1-exp(x)) for x<=0 -- Slang/HLSL has neither log1p
// nor expm1 built in. These favor the well-conditioned branch over the cheap
// one: for arguments this small, exp(x)-1 or log(1+y) lose most of their
// significant digits to cancellation, so a short Taylor series covers the
// region a direct call would get wrong.
//
// The threshold below (and the Taylor series' one extra term) replaces an
// earlier |arg| > 1e-4 cutoff that was far too tight for float32: direct
// exp(x)-1 (or log(1+y)) loses roughly log2(6e-8/|arg|) bits to cancellation,
// so 1e-4 -- covering only |arg| < 1e-4 with Taylor -- left the entire
// 1e-4..~1e-2 band on the direct branch, which is most of this file's actual
// arguments (measured up to ~4200 ULP error via TruncatedGMM.slang's calls;
// see tests/gaussianHelpers/). 0.05 plus a fourth Taylor term brings that
// down to ~25-55 ULP at the same instruction count -- a threshold/order fix,
// not a different algorithm. (A full range-reduced expm1f/log1pf, e.g.
// Norbert Juffa's forums.developer.nvidia.com/t/298202 and t/40575, gets to
// ~1 ULP but costs 2-2.7x more per call; not worth it for this file's narrow
// argument range.)
[Differentiable]
float log1p(float y)
{
    return (abs(y) > 0.05f) ? log(1.0f + y) : (y * fma(y, fma(y, fma(y, -1.0f / 4.0f, 1.0f / 3.0f), -0.5f), 1.0f));
}

[Differentiable]
float expm1(float x)
{
    return (abs(x) > 0.05f) ? (exp(x) - 1.0f) : (x * fma(x, fma(x, fma(x, 1.0f / 24.0f, 1.0f / 6.0f), 0.5f), 1.0f));
}

// log(1 - exp(x)) for x <= 0, per Machler (2012); avoids ever forming the
// (possibly tiny) 1-exp(x) directly.
[Differentiable]
float log1mexp(float x)
{
    return (x > -log(2.0f)) ? log(-expm1(x)) : log1p(-exp(x));
}

// log(Phi(z)) for z <= 0 only: Phi(z) = 0.5*erfc(-z/sqrt2), so
// log(Phi(z)) = -z^2/2 + log(erfcx(-z/sqrt2)) - log(2) -- never forms the
// (possibly tiny) linear-space Phi(z) directly. Split out of logNormalCDF
// below (rather than a self-call under the z<=0 branch) because Slang's
// compute-shader targets reject recursion outright, even the provably-single-
// level kind a sign-flip reflection produces here.
[Differentiable]
float logNormalCDFNonPositive(float z)
{
    let u = -z * (1.0f / sqrt(2.0f)); // -z/sqrt2, >= 0: erfcxPositive's domain
    return -0.5f * z * z + log(erfcxPositive(u)) - log(2.0f);
}

// Stable log(Phi(z)) for the standard normal CDF, all real z. For z>0,
// reflect: Phi(z) = 1 - Phi(-z), and -z<=0 is the stable branch above.
[Differentiable]
float logNormalCDF(float z)
{
    if (z <= 0.0f)
        return logNormalCDFNonPositive(z);
    return log1mexp(logNormalCDFNonPositive(-z));
}

[Differentiable]
float logNormalCDF(float x, float mean, float logSigma)
{
    return logNormalCDF((x - mean) * exp(-logSigma));
}

[ForceInline]
float normalCDF(float x, float mean, float logSigma)
{
    return exp(logNormalCDF(x, mean, logSigma));
}

// Inverse error function approximation.
// Slang/SPIR-V translation of Norbert Juffa's BSD-2-Clause compact erfinvf.
// https://forums.developer.nvidia.com/t/a-very-compact-implementation-of-the-inverse-error-function-erfinvf/61961
// Replaces an earlier port of Mike Giles's "Approximating the erfinv function"
// polynomial (https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf):
// measured max error over this file's argument range (see
// tests/gaussianHelpers/) was 5 ULP for Giles's version vs. 4 ULP for this
// one, at a small GPU-time win (~9%) rather than a cost -- reported max error
// for the original CUDA implementation: 2.446816 ULP. `lg2.approx.ftz.f32` ->
// log2().
//
// Copyright (c) 2018, Norbert Juffa
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
[ForceInline]
float erfinv(float x)
{
    float t = log2(fma(x, -x, 1.0f));
    float p, r;
    if (t < -8.8359375f)
    {
        p =              1.50166858e-11f;
        p = fma(p, t,  2.10561724e-09f);
        p = fma(p, t,  1.27291500e-07f);
        p = fma(p, t,  4.29223974e-06f);
        p = fma(p, t,  8.60039654e-05f);
        p = fma(p, t,  9.49774636e-04f);
        p = fma(p, t,  1.88959786e-03f);
        p = fma(p, t, -1.85239315e-01f);
        p = fma(p, t,  8.36782515e-01f);
        r = fma(p, x, 0.0f);
    }
    else
    {
        p =              2.00998329e-10f;
        p = fma(p, t,  7.63788321e-09f);
        p = fma(p, t,  9.44120231e-08f);
        p = fma(p, t,  1.27279494e-08f);
        p = fma(p, t, -8.98369126e-06f);
        p = fma(p, t, -3.40910883e-05f);
        p = fma(p, t,  7.70850747e-04f);
        p = fma(p, t,  5.54407062e-03f);
        p = fma(p, t, -1.60820886e-01f);
        p = fma(p, t,  8.86226892e-01f);
        r = x * p;
    }
    return r;
}

[ForceInline]
float normalCDFInverse(float p, float mean, float logSigma)
{
    float k = exp(logSigma) * sqrt(2.0f);
    return mean + k * erfinv(2.0f * p - 1.0f);
}

// Stable log P(0 <= X <= 1) for X ~ N(mean, sigma^2), sigma = exp(logSigma).
// This normalizer used to come from erf((1-mean)*k)+erf(mean*k) directly (see
// logTruncGaussianPDF below), which underflows to exactly 0 -- log(0) = -inf,
// then NaN gradients -- once the standardized interval [zLo, zHi] sits
// entirely in one tail. This keeps the interior case's cheap erf evaluation
// (both CDF values are O(1) there, no cancellation) but routes both tail
// cases through logNormalCDF and a logsubexp identity so the subtraction of
// two CDF values never happens in linear (collapsible) space: log(a - b) =
// log(a) + log1mexp(log(b) - log(a)) for a >= b > 0, where log(a) and log(b)
// individually stay finite (they cancel the common -z^2/2 term algebraically
// instead of forming exp(-z^2/2) and rounding it to zero first).
// Branchless-tail variant: keeps the interior erf fast path (it's a different,
// cheaper formula, not something a swap/negate can fold in) but collapses the
// two tail branches -- which were exact mirror images of each other -- into
// one, reflecting unconditionally via swap+negate instead of duplicating the
// log1mexp identity for each tail direction.
//
// [BackwardDerivative]: closed-form reverse-mode gradient instead of relying
// on Slang's autodiff through logNormalCDF/log1mexp/erf. d(logM)/d(mean) and
// d(logM)/d(logSigma) both reduce to phi(z)/M ratios (see logTruncMass_bwd);
// computing those ratios as exp(logPhi(z) - logM) keeps M itself -- which can
// collapse to 0 in exactly the tail regime this file exists to handle -- out
// of the computation entirely, same spirit as the forward pass above.
[BackwardDerivative(logTruncMass_bwd)]
float logTruncMass(float mean, float logSigma)
{
    let invSigma = exp(-logSigma);
    let zLo = -mean * invSigma;         // (0 - mean) / sigma
    let zHi = (1.0f - mean) * invSigma; // (1 - mean) / sigma; zHi > zLo always

    if (zLo < 0.0f && zHi > 0.0f)
    {
        // Interior: both CDF values are O(1) here, so the direct erf
        // difference is well-behaved and cheaper than routing through
        // logNormalCDF/log1mexp.
        let invSqrt2 = 1.0f / sqrt(2.0f);
        return log(0.5f * (erf(zHi * invSqrt2) - erf(zLo * invSqrt2)));
    }

    let reflect = zLo >= 0.0f;
    let lo = reflect ? -zHi : zLo;
    let hi = reflect ? -zLo : zHi;

    let logHi = logNormalCDF(hi);
    let logLo = logNormalCDF(lo);
    return logHi + log1mexp(logLo - logHi);
}

// d(logM)/d(mean) = -(phi(zHi) - phi(zLo)) / (sigma*M)
// d(logM)/d(logSigma) = (zLo*phi(zLo) - zHi*phi(zHi)) / M
// where M = exp(logM) is the truncation mass and phi is the standard normal
// density. Both are expressed as ratios phi(z)/M = exp(logPhi(z) - logM), so
// the (possibly astronomically small) M is never formed in linear space.
void logTruncMass_bwd(inout DifferentialPair<float> dpMean, inout DifferentialPair<float> dpLogSigma, float dOut)
{
    let mean = dpMean.p;
    let logSigma = dpLogSigma.p;
    let invSigma = exp(-logSigma);
    let zLo = -mean * invSigma;
    let zHi = (1.0f - mean) * invSigma;

    let logM = logTruncMass(mean, logSigma);
    let logNormConst = -0.5f * log(2.0f * float.getPi()); // log(1/sqrt(2*pi))
    let ratioLo = exp(logNormConst - 0.5f * zLo * zLo - logM); // phi(zLo) / M
    let ratioHi = exp(logNormConst - 0.5f * zHi * zHi - logM); // phi(zHi) / M

    let dMean = (ratioLo - ratioHi) * invSigma;
    let dLogSigma = zLo * ratioLo - zHi * ratioHi;

    dpMean = diffPair(mean, dOut * dMean);
    dpLogSigma = diffPair(logSigma, dOut * dLogSigma);
}

[Differentiable]
float logTruncGaussianPDF(float2 x, float2 mean, float2 logSigma)
{
    let inverseSigma = exp(-logSigma);
    let diff = (x - mean) * inverseSigma;
    let exponent = 0.5 * dot(diff, diff);
    float logPdf = -log(2.0f * float.getPi()) - logSigma.x - logSigma.y - exponent;

    [ForceUnroll]
    for (uint d = 0; d < 2; ++d)
        logPdf -= logTruncMass(mean[d], logSigma[d]);

    return logPdf;
}