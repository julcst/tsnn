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

import TSNN.CoopVec;
import TSNN.Flows.Bijection;

// ─── Array-form forward transform ────────────────────────────────────────────

[Differentiable]
FlowStep plForward<let K : uint>(float x, float q[K], no_diff float tail)
{
    if (x <= -tail || x >= tail) { FlowStep r = { x, 0.0f }; return r; }

    // bin probabilities via numerically-stable softmax
    float p[K];
    float maxQ = q[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxQ = max(maxQ, q[i]);
    float sum = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { p[i] = exp(q[i] - maxQ); sum += p[i]; }
    float invSum = 1.0f / sum;
    [ForceUnroll] for (uint i = 0; i < K; i++) p[i] = p[i] * invSum;

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
    float maxQ = q[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxQ = max(maxQ, q[i]);
    float sum = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { p[i] = exp(q[i] - maxQ); sum += p[i]; }
    float invSum = 1.0f / sum;
    [ForceUnroll] for (uint i = 0; i < K; i++) p[i] = p[i] * invSum;

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

[Differentiable]
FlowStep plForward<let K : uint>(float x, CoopVec<half, K> cv, no_diff float tail)
{
    let arr = CoopVecToArray(cv);
    float q[K];
    [ForceUnroll] for (uint i = 0u; i < K; i++) q[i] = arr[i];
    return plForward<K>(x, q, tail);
}

FlowStep plInverse<let K : uint>(float y, CoopVec<half, K> cv, float tail)
{
    let arr = CoopVecToArray(cv);
    float q[K];
    [ForceUnroll] for (uint i = 0u; i < K; i++) q[i] = arr[i];
    return plInverse<K>(y, q, tail);
}
