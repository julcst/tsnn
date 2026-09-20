// TSNN/Flows/PiecewiseLinear.slang
// Piecewise-linear coupling transform — Müller et al., "Neural Importance
// Sampling" (2019), §4.1 (the original NIS transform, predating RQS).
//
// K equal-width bins on [-tail, tail]. The conditioner emits K unnormalised bin
// masses; softmax turns them into probabilities p[i] (Σ p = 1). The PDF is
// piecewise-CONSTANT (a learned histogram) so the CDF — hence the transform — is
// piecewise-LINEAR.  dy/dx = p[bin]·K (identity when p is uniform).
//
// API mirrors RQS.slang exactly (FlowStep{value, logAbsDet}, signed +fwd/−inv):
//   plForward<K>(x, q[K], tail)                 [Differentiable]
//   plInverse<K>(y, q[K], tail)
//   plForward<K>(x, CoopVec<half,K>, tail)      [Differentiable]
//   plInverse<K>(y, CoopVec<half,K>, tail)
//
// Packed parameter order is [q(K)].  Decode is inlined in each entry point for
// the same autodiff-miscompile reason documented in RQS.slang.
//
// Bin-probability floor: a bare softmax(q) can drive any p[i] arbitrarily close
// to 0 (a zero-initialised or adversarial conditioner will do this eventually).
// log(max(pb*K, 1e-30)) (below) then falls off the 1e-30 cliff instead of
// staying well-conditioned. Floored the same way RQS.slang floors its widths/
// heights/derivatives (kRQSMinBinWidth etc.): a convex combination of the
// uniform distribution and the softmax output,
//   p[i] = kPLMinBinProb + (1 - kPLMinBinProb*K) * p[i]*invSum,
// which keeps Sigma p == 1 exactly (both the floored and unfloored fields sum
// to 1, and the combination is linear) while lower-bounding every bin at
// kPLMinBinProb. That bounds log(p*K) below by log(kPLMinBinProb*K), demoting
// the 1e-30 guards below to sanity nets that should never actually trigger.
static const float kPLMinBinProb = 1e-3f;

import TSNN.CoopVec;
import TSNN.Flows.Bijection;

// ─── Array-form forward transform ────────────────────────────────────────────

[Differentiable]
FlowStep plForward<let K : uint>(float x, float q[K], no_diff float tail)
{
    if (x <= -tail || x >= tail) { FlowStep r = { x, 0.0f }; return r; }

    // bin probabilities via numerically-stable softmax
    float p[K];
    // Seed from -inf and route EVERY index (including 0) through max(): seeding
    // maxQ = q[0] directly and starting the loop at i=1 measurably drops the
    // backward-mode gradient path into q[0] under this Slang version's autodiff
    // (bwd_diff(...)(qPair,...) leaves qPair.d[0] at exactly 0 regardless of q,
    // confirmed against central differences -- every other index gradients
    // fine). Discovered while adding this file's own FD-vs-bwd_diff regression
    // test (tests/piecewise/test_piecewise.py); RQS.slang has the same pattern
    // and is presumed to share the defect -- see FINDING.md.
    float maxQ = -1.0#INF;
    [ForceUnroll] for (uint i = 0; i < K; i++) maxQ = max(maxQ, q[i]);
    float sum = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { p[i] = exp(q[i] - maxQ); sum += p[i]; }
    float invSum = 1.0f / sum;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        p[i] = kPLMinBinProb + (1.0f - kPLMinBinProb * float(K)) * p[i] * invSum;

    float span = 2.0f * tail;
    float w    = span / float(K);

    // locate bin and its cumulative probability at the left knot
    float u   = (x + tail) / span;      // ∈ [0,1)
    int   bin = min(int(u * float(K)), int(K) - 1);
    float cumP = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) cumP += (int(i) < bin) ? p[i] : 0.0f;

    float pb    = p[bin];
    float theta = clamp((x - (-tail + float(bin) * w)) / w, 0.0f, 1.0f);

    float y      = -tail + span * (cumP + pb * theta);
    float logDet = log(max(pb * float(K), 1e-30f));   // log(dy/dx)

    FlowStep r = { y, logDet };
    return r;
}

// ─── Array-form inverse transform ────────────────────────────────────────────

FlowStep plInverse<let K : uint>(float y, float q[K], float tail)
{
    if (y <= -tail || y >= tail) { FlowStep r = { y, 0.0f }; return r; }

    float p[K];
    // Seed from -inf and route EVERY index (including 0) through max(): seeding
    // maxQ = q[0] directly and starting the loop at i=1 measurably drops the
    // backward-mode gradient path into q[0] under this Slang version's autodiff
    // (bwd_diff(...)(qPair,...) leaves qPair.d[0] at exactly 0 regardless of q,
    // confirmed against central differences -- every other index gradients
    // fine). Discovered while adding this file's own FD-vs-bwd_diff regression
    // test (tests/piecewise/test_piecewise.py); RQS.slang has the same pattern
    // and is presumed to share the defect -- see FINDING.md.
    float maxQ = -1.0#INF;
    [ForceUnroll] for (uint i = 0; i < K; i++) maxQ = max(maxQ, q[i]);
    float sum = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { p[i] = exp(q[i] - maxQ); sum += p[i]; }
    float invSum = 1.0f / sum;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        p[i] = kPLMinBinProb + (1.0f - kPLMinBinProb * float(K)) * p[i] * invSum;

    float span = 2.0f * tail;
    float w    = span / float(K);

    float s = (y + tail) / span;        // CDF value ∈ [0,1)

    // locate bin: largest b with cumulative probability ≤ s
    float cum = 0.0f;
    int   bin = 0;
    [ForceUnroll] for (uint i = 0; i < K; i++)
    {
        bin = (s >= cum + p[i]) ? int(i + 1) : bin;
        cum += p[i];
    }
    bin = min(bin, int(K) - 1);

    // cumulative probability at the chosen bin's left knot
    float cumP = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) cumP += (int(i) < bin) ? p[i] : 0.0f;

    float pb    = p[bin];
    float theta = clamp((s - cumP) / max(pb, 1e-30f), 0.0f, 1.0f);

    float x      = -tail + (float(bin) + theta) * w;
    float logDet = log(max(pb * float(K), 1e-30f));

    FlowStep r = { x, -logDet };
    return r;
}

// ─── CoopVec-form overloads (unpack packed [q] internally) ────────────────────

// Raw-conditioner-output clamp, matching TSNN.Flows.RQS's kRQSMaxRaw: a
// runaway conditioner MLP can emit an unbounded (even +-inf) raw logit here,
// same as it did for TSNN.Mixtures.Histogram (see FINDING.md's DF-N/NSF-
// Quadratic writeup) -- clamping before any softmax math in plForward/
// plInverse prevents inf/NaN from ever entering this bijection's decode.
static const float kPLMaxRaw = 15.0f;

[Differentiable]
FlowStep plForward<let K : uint>(float x, CoopVec<half, K> cv, no_diff float tail)
{
    let arr = CoopVecToArray(cv);
    float q[K];
    [ForceUnroll] for (uint i = 0u; i < K; i++) q[i] = clamp(arr[i], -kPLMaxRaw, kPLMaxRaw);
    return plForward<K>(x, q, tail);
}

FlowStep plInverse<let K : uint>(float y, CoopVec<half, K> cv, float tail)
{
    let arr = CoopVecToArray(cv);
    float q[K];
    [ForceUnroll] for (uint i = 0u; i < K; i++) q[i] = clamp(arr[i], -kPLMaxRaw, kPLMaxRaw);
    return plInverse<K>(y, q, tail);
}
