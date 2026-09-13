// GPU accuracy/throughput probes for tests/gaussianHelpers/test_gaussian_helpers.py.
//
// Every "current" entry point calls straight into TSNN/Mixtures/GaussianHelpers.slang.
// Every "candidate" is a standalone alternative implementation kept here until it wins
// (per the accept rule in the optimization pass) -- nothing candidate lands in the
// library file until its accuracy/throughput numbers justify it.
import TSNN.Mixtures.GaussianHelpers;

// ─── Candidate: erf (Norbert Juffa, "accuracy-optimized-implementation-of-erff",
// forums.developer.nvidia.com/t/218114). BSD-2-Clause, Copyright (c) 2017-2022
// Norbert Juffa. Branch-free select instead of the library's `if`; 7 coefficients
// per branch (vs. the library's 6) for 0.96922/1.03872 ulp vs. the library's
// reported 1.29532 ulp. `ex2.approx.ftz.f32` -> exp2().
float erf_candidate(float a)
{
    const float switchover = 1.0f;
    float fa = abs(a);
    bool upper = fa >= switchover;
    float s = upper ? -fa : (a * a);
    float t = upper ? -fa : a;
    float c0 = upper ?  2.58684158e-5f :  8.05854797e-5f;
    float c1 = upper ?  5.64946968e-4f : -8.07280652e-4f;
    float c2 = upper ?  5.66854654e-3f :  5.19511057e-3f;
    float c3 = upper ?  3.51768062e-2f : -2.68568154e-2f;
    float c4 = upper ?  1.54329658e-1f :  1.12836286e-1f;
    float c5 = upper ? -9.15674567e-1f : -3.76126260e-1f;
    float c6 = upper ?  6.28459752e-1f :  1.28379166e-1f;
    float r = c0;
    r = fma(r, s, c1);
    r = fma(r, s, c2);
    r = fma(r, s, c3);
    r = fma(r, s, c4);
    r = fma(r, s, c5);
    r = fma(r, s, c6);
    r = fma(r, t, t);
    if (upper)
    {
        r = exp2(r);
        r = 1.0f - r;
        r = asfloat((asuint(a) & 0x80000000u) | asuint(r));
    }
    return r;
}

// ─── Candidate: expm1 (Norbert Juffa, "accuracy-optimized-implementation-of-
// expm1f-without-performance-penalty", forums.developer.nvidia.com/t/298202).
// BSD-2-Clause, Copyright (c) 2016-2023 Norbert Juffa. Full range-reduction
// expm1(a) = 2^i*(expm1(f)+1) - 1 instead of the library's branch-on-magnitude
// Taylor stub; 0.997458 ulp faithfully-rounded. `b` scales the "-1" (kept at 1.0
// here; this file never needs the fused expm1-scaled-by-b form). The library's
// `|a-1| > 88` overflow guard is dropped: unreachable on this codebase's live
// domain for expm1, which is `(-log(2), 0]` (only log1mexp's `x > -log(2)` branch
// calls it).
float expm1_candidate_scaled(float a, float b)
{
    float f, j, r, s, t, u, v, x, y;
    int i;

    j = fma(1.442695f, a, 12582912.f);
    i = asint(j);
    j = j - 12582912.f;
    f = fma(j, -6.93145752e-1f, a);
    f = fma(j, -1.42860677e-6f, f);

    s = f * f;
    if (a == 0.0f) s = a;
    r = fma(1.98662281e-4f, f, 1.39354519e-3f);
    t = fma(8.33332818e-3f, f, 4.16667648e-2f);
    r = fma(r, s, t);
    r = fma(r, f, 1.66666716e-1f);
    r = fma(r, f, 4.99999970e-1f);

    u = (j == 1.0f) ? (f + 0.5f) : f;
    v = fma(r, s, u);
    s = 0.5f * b;
    t = asfloat((i << 23) + asint(s));
    y = t - s;
    x = t - y;
    x = x - s;
    r = fma(v, t, x);
    r = r + y;
    r = r + r;
    if (j == 0.0f) r = v;
    if (j == 1.0f) r = v + v;
    return r;
}

float expm1_candidate(float a)
{
    float r = expm1_candidate_scaled(a, 1.0f);
    // Saturate to the correct asymptote (-1 or +inf) once the range-reduction's
    // exact-integer special cases (j==0, j==1) no longer cover the input;
    // `ex2.approx.ftz.f32` -> exp2(). Unreachable on this codebase's live
    // domain (-log(2), 0], but kept so this stays a safe general-purpose
    // expm1, not just a fit to that one call site.
    if (abs(a - 1.0f) > 88.0f)
    {
        r = exp2(a);
        r = fma(r, r, -1.0f);
    }
    return r;
}

// ─── Alternative: same branch-on-magnitude shape as the library's expm1, but
// with a wider threshold and a 4th-order Taylor term. The library's 1e-4
// threshold is far too tight for float32: direct exp(x)-1 loses ~log2(6e-8/|x|)
// bits to cancellation once |x| is small (measured: thousands of ulp for
// |x| in (1e-4, ~1e-2)), which is most of this codebase's actual (-log(2),0]
// domain. Same instruction count as the library version either way -- this is
// a threshold/order fix, not a different algorithm.
float expm1_widertaylor(float x)
{
    return (abs(x) > 0.05f) ? (exp(x) - 1.0f) : (x * fma(x, fma(x, fma(x, 1.0f / 24.0f, 1.0f / 6.0f), 0.5f), 1.0f));
}

float log1p_widertaylor(float y)
{
    return (abs(y) > 0.05f) ? log(1.0f + y) : (y * fma(y, fma(y, fma(y, -1.0f / 4.0f, 1.0f / 3.0f), -0.5f), 1.0f));
}

// ─── Candidate: log1p (Norbert Juffa, "faster-and-more-accurate-implementation-
// of-log1pf", forums.developer.nvidia.com/t/40575). BSD-2-Clause, Copyright (c)
// 2016 Norbert Juffa. Branch-free exponent-field decomposition instead of the
// library's branch-on-magnitude Taylor stub; 0.87168 ulp. The library's
// non-finite/zero fixup (`lg2.approx.ftz.f32` + `__fadd_rd`) is dropped:
// unreachable on this codebase's live domain for log1p, which is `[-0.5, 0)`
// (only log1mexp's `x <= -log(2)` branch calls it, as `log1p(-exp(x))`).
float log1p_candidate(float a)
{
    const float LOG_TWO = 0.693147182f;
    const float TWO_TO_M23 = 1.19209290e-7f;
    float m, r, s, t, u;
    int e;

    u = a + 1.0f;
    e = (asint(u) - asint(0.75f)) & 0xff800000;
    m = asfloat(asint(a) - e);
    s = asfloat(asint(4.0f) - e);
    m = m + fma(0.25f, s, -1.0f);
    s = m * m;
    r =             -4.54559326e-2f;
    t =              1.05529785e-1f;
    r = fma(r, s, -1.32279143e-1f);
    t = fma(t, s,  1.44911006e-1f);
    r = fma(r, s, -1.66416913e-1f);
    t = fma(t, s,  1.99886635e-1f);
    r = fma(r, s, -2.50001878e-1f);
    r = fma(t, m, r);
    r = fma(r, m,  3.33335280e-1f);
    r = fma(r, m, -5.00000000e-1f);
    r = fma(r, s, m);
    r = fma(float(e), LOG_TWO * TWO_TO_M23, r);
    return r;
}

// ─── Candidate: erfinv (Norbert Juffa, "a-very-compact-implementation-of-the-
// inverse-error-function-erfinvf", forums.developer.nvidia.com/t/61961).
// BSD-2-Clause, Copyright (c) 2018 Norbert Juffa. 2.446816 ulp.
// `lg2.approx.ftz.f32` -> log2().
float erfinv_candidate(float a)
{
    float t = fma(a, -a, 1.0f);
    t = log2(t);
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
        r = fma(p, a, 0.0f);
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
        r = a * p;
    }
    return r;
}

// ─── Alternative: log1mexp with the log(2.0f) call site replaced by the literal
// used elsewhere in this bench (guards against log(2.0f) not being constant-
// folded on this backend).
float log1mexp_litlog2(float x)
{
    const float LOG_TWO = 0.693147182f;
    return (x > -LOG_TWO) ? log(-expm1(x)) : log1p(-exp(x));
}

// Wraps the candidate expm1/log1p so log1mexp's own dispatch/branch-removal
// question can be re-measured once those primitives are settled.
float log1mexp_candidatePrimitives(float x)
{
    const float LOG_TWO = 0.693147182f;
    return (x > -LOG_TWO) ? log(-expm1_candidate(x)) : log1p_candidate(-exp(x));
}

// Same, but with the cheap widened-threshold Taylor fix instead of the full
// Juffa range-reduction -- the speed-preserving option.
float log1mexp_widertaylorPrimitives(float x)
{
    const float LOG_TWO = 0.693147182f;
    return (x > -LOG_TWO) ? log(-expm1_widertaylor(x)) : log1p_widertaylor(-exp(x));
}

// ═══ Entry points ═══════════════════════════════════════════════════════════

cbuffer CB
{
    uint gCount;
}
StructuredBuffer<float> gInputs;
RWStructuredBuffer<float> gOutputs;

// One evaluation per thread, for ULP measurement.
#define ACCURACY_KERNEL(name, fn) \
    [numthreads(64, 1, 1)] \
    void name(uint3 tid : SV_DispatchThreadID) \
    { \
        uint i = tid.x; \
        if (i >= gCount) return; \
        gOutputs[i] = fn(gInputs[i]); \
    }

ACCURACY_KERNEL(accuracyErfCurrentMain, erf)
ACCURACY_KERNEL(accuracyErfCandidateMain, erf_candidate)
ACCURACY_KERNEL(accuracyErfcxCurrentMain, erfcxPositive)
ACCURACY_KERNEL(accuracyExpm1CurrentMain, expm1)
ACCURACY_KERNEL(accuracyExpm1CandidateMain, expm1_candidate)
ACCURACY_KERNEL(accuracyExpm1WidertaylorMain, expm1_widertaylor)
ACCURACY_KERNEL(accuracyLog1pCurrentMain, log1p)
ACCURACY_KERNEL(accuracyLog1pCandidateMain, log1p_candidate)
ACCURACY_KERNEL(accuracyLog1pWidertaylorMain, log1p_widertaylor)
ACCURACY_KERNEL(accuracyLog1mexpCurrentMain, log1mexp)
ACCURACY_KERNEL(accuracyLog1mexpLitlog2Main, log1mexp_litlog2)
ACCURACY_KERNEL(accuracyLog1mexpCandidatePrimitivesMain, log1mexp_candidatePrimitives)
ACCURACY_KERNEL(accuracyLog1mexpWidertaylorPrimitivesMain, log1mexp_widertaylorPrimitives)
ACCURACY_KERNEL(accuracyErfinvCurrentMain, erfinv)
ACCURACY_KERNEL(accuracyErfinvCandidateMain, erfinv_candidate)

// Repeated evaluation per thread with a tiny per-iteration perturbation (so the
// compiler cannot hoist/CSE the loop into a single evaluation), for throughput
// measurement. Iteration count is a specialization constant so both current and
// candidate kernels share identical dispatch/loop overhead.
static const uint THROUGHPUT_ITERS = 1024;

#define THROUGHPUT_KERNEL(name, fn) \
    [numthreads(64, 1, 1)] \
    void name(uint3 tid : SV_DispatchThreadID) \
    { \
        uint i = tid.x; \
        if (i >= gCount) return; \
        float x = gInputs[i]; \
        float acc = 0.0f; \
        for (uint k = 0; k < THROUGHPUT_ITERS; ++k) \
        { \
            float xi = x + float(k) * 1e-7f; \
            acc += fn(xi); \
        } \
        gOutputs[i] = acc; \
    }

THROUGHPUT_KERNEL(throughputErfCurrentMain, erf)
THROUGHPUT_KERNEL(throughputErfCandidateMain, erf_candidate)
THROUGHPUT_KERNEL(throughputErfcxCurrentMain, erfcxPositive)
THROUGHPUT_KERNEL(throughputExpm1CurrentMain, expm1)
THROUGHPUT_KERNEL(throughputExpm1CandidateMain, expm1_candidate)
THROUGHPUT_KERNEL(throughputExpm1WidertaylorMain, expm1_widertaylor)
THROUGHPUT_KERNEL(throughputLog1pCurrentMain, log1p)
THROUGHPUT_KERNEL(throughputLog1pCandidateMain, log1p_candidate)
THROUGHPUT_KERNEL(throughputLog1pWidertaylorMain, log1p_widertaylor)
THROUGHPUT_KERNEL(throughputLog1mexpCurrentMain, log1mexp)
THROUGHPUT_KERNEL(throughputLog1mexpLitlog2Main, log1mexp_litlog2)
THROUGHPUT_KERNEL(throughputLog1mexpCandidatePrimitivesMain, log1mexp_candidatePrimitives)
THROUGHPUT_KERNEL(throughputLog1mexpWidertaylorPrimitivesMain, log1mexp_widertaylorPrimitives)
THROUGHPUT_KERNEL(throughputErfinvCurrentMain, erfinv)
THROUGHPUT_KERNEL(throughputErfinvCandidateMain, erfinv_candidate)
