// TSNN/Flows/PiecewiseQuadratic.slang
// Piecewise-quadratic coupling transform — Müller et al., "Neural Importance
// Sampling" (2019), §4.2 (the original NIS transform, predating RQS).
//
// K bins on [-tail, tail].  The conditioner emits K unnormalised bin widths and
// K+1 unnormalised vertex heights.  softmax(W) → widths ω[i] (Σ ω = 1); the
// vertices are normalised so the piecewise-LINEAR PDF integrates to 1, which
// makes the CDF — hence the transform — piecewise-QUADRATIC.  The PDF is C0
// (continuous), one order smoother than the piecewise-linear variant's histogram.
//
// API mirrors RQS.slang exactly (FlowStep{value, logAbsDet}, signed +fwd/−inv):
//   pqForward<K>(x, W[K], V[K+1], tail)                 [Differentiable]
//   pqInverse<K>(y, W[K], V[K+1], tail)
//   pqForward<K>(x, CoopVec<half,2K+1>, tail)           [Differentiable]
//   pqInverse<K>(y, CoopVec<half,2K+1>, tail)
//
// Packed parameter order is [W(K) | V(K+1)].  Decode is inlined in each entry
// point for the same autodiff-miscompile reason documented in RQS.slang.
//
// Two floors, both for the same reason PiecewiseLinear.slang floors its bin
// probabilities: an unfloored softmax/normalisation can collapse a width or a
// vertex height arbitrarily close to 0, which makes `rhs / max(vb, 1e-30f)`
// (the linear-branch inverse) and the quadratic root's discriminant
// ill-conditioned right at the values a trained conditioner tends to wander
// into -- the recorded cause of the quadratic spline's training divergence.
//
//   kPQMinBinWidth: same convex-combination floor as RQS.slang's bin widths,
//   applied to om[i] -- Sigma om == 1 is preserved exactly (linear in om, and
//   the uniform field om==1/K satisfies it).
//
//   kPQMinVertex: v[i] = kPQMinVertex + (1-kPQMinVertex)*v[i], a convex
//   combination of the normalised vertex heights with the all-ones field. The
//   PDF-normalisation constraint Sigma om_i*(v_i+v_{i+1})/2 == 1 is LINEAR in
//   v, and v==1 satisfies it (Sigma om_i*1 == 1 since Sigma om == 1); a convex
//   combination of two solutions to a linear constraint is itself a solution,
//   so this floor preserves the constraint exactly with no second
//   re-normalisation pass needed.
static const float kPQMinBinWidth = 1e-3f;
static const float kPQMinVertex   = 1e-3f;

// Every running-max below is seeded from -inf and loops over every index
// (including 0), not from x[0] with the loop starting at i=1: the latter
// measurably drops the backward-mode gradient path into index 0 under this
// Slang version's autodiff -- see PiecewiseLinear.slang's header for the
// isolated repro and FINDING.md.

import TSNN.CoopVec;
import TSNN.Flows.Bijection;

// ─── Array-form forward transform ────────────────────────────────────────────

[Differentiable]
FlowStep pqForward<let K : uint>(float x, float W[K], float V[K + 1u], no_diff float tail)
{
    if (x <= -tail || x >= tail) { FlowStep r = { x, 0.0f }; return r; }

    // widths ω via softmax(W); Σ ω = 1
    float om[K];
    float maxW = -1.0#INF;
    [ForceUnroll] for (uint i = 0; i < K; i++) maxW = max(maxW, W[i]);
    float sumW = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { om[i] = exp(W[i] - maxW); sumW += om[i]; }
    float invW = 1.0f / sumW;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        om[i] = kPQMinBinWidth + (1.0f - kPQMinBinWidth * float(K)) * om[i] * invW;

    // vertices: exp(V), normalised so ∫ pdf = Σ ω·(v_i+v_{i+1})/2 = 1
    float ev[K + 1u];
    float maxV = -1.0#INF;
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) maxV = max(maxV, V[i]);
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) ev[i] = exp(V[i] - maxV);
    float denom = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) denom += 0.5f * (ev[i] + ev[i + 1]) * om[i];
    float invD = 1.0f / denom;
    float v[K + 1u];
    // Floor as a convex combination with the all-ones field -- exact under the
    // (now-floored-om) constraint above; see the header comment.
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++)
        v[i] = kPQMinVertex + (1.0f - kPQMinVertex) * (ev[i] * invD);

    // cumulative width knots Ω and cumulative area (CDF at knots) cumA
    float cumOm[K + 1u];
    float cumA[K + 1u];
    cumOm[0] = 0.0f;  cumA[0] = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++)
    {
        cumOm[i + 1] = cumOm[i] + om[i];
        cumA[i + 1]  = cumA[i]  + 0.5f * (v[i] + v[i + 1]) * om[i];
    }

    float span = 2.0f * tail;
    float u    = (x + tail) / span;     // ∈ [0,1)

    int bin = 0;
    [ForceUnroll] for (uint i = 0; i < K; i++) bin = (u >= cumOm[i + 1]) ? int(i + 1) : bin;
    bin = min(bin, int(K) - 1);

    float ob = om[bin];
    float vb = v[bin];
    float dv = v[bin + 1] - vb;
    float t  = clamp((u - cumOm[bin]) / ob, 0.0f, 1.0f);

    float s      = cumA[bin] + ob * (vb * t + 0.5f * dv * t * t);
    float y      = -tail + span * s;
    float pdf    = vb + dv * t;         // ds/du = dy/dx
    float logDet = log(max(pdf, 1e-30f));

    FlowStep r = { y, logDet };
    return r;
}

// ─── Array-form inverse transform ────────────────────────────────────────────

FlowStep pqInverse<let K : uint>(float y, float W[K], float V[K + 1u], float tail)
{
    if (y <= -tail || y >= tail) { FlowStep r = { y, 0.0f }; return r; }

    float om[K];
    float maxW = -1.0#INF;
    [ForceUnroll] for (uint i = 0; i < K; i++) maxW = max(maxW, W[i]);
    float sumW = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { om[i] = exp(W[i] - maxW); sumW += om[i]; }
    float invW = 1.0f / sumW;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        om[i] = kPQMinBinWidth + (1.0f - kPQMinBinWidth * float(K)) * om[i] * invW;

    float ev[K + 1u];
    float maxV = -1.0#INF;
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) maxV = max(maxV, V[i]);
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) ev[i] = exp(V[i] - maxV);
    float denom = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) denom += 0.5f * (ev[i] + ev[i + 1]) * om[i];
    float invD = 1.0f / denom;
    float v[K + 1u];
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++)
        v[i] = kPQMinVertex + (1.0f - kPQMinVertex) * (ev[i] * invD);

    float cumOm[K + 1u];
    float cumA[K + 1u];
    cumOm[0] = 0.0f;  cumA[0] = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++)
    {
        cumOm[i + 1] = cumOm[i] + om[i];
        cumA[i + 1]  = cumA[i]  + 0.5f * (v[i] + v[i + 1]) * om[i];
    }

    float span = 2.0f * tail;
    float s    = (y + tail) / span;     // CDF value ∈ [0,1)

    int bin = 0;
    [ForceUnroll] for (uint i = 0; i < K; i++) bin = (s >= cumA[i + 1]) ? int(i + 1) : bin;
    bin = min(bin, int(K) - 1);

    float ob = om[bin];
    float vb = v[bin];
    float dv = v[bin + 1] - vb;
    float rhs = (s - cumA[bin]) / ob;   // = vb·t + 0.5·dv·t²

    // solve 0.5·dv·t² + vb·t − rhs = 0 for the monotone root t ∈ [0,1]
    float t;
    if (abs(dv) < 1e-6f)
        t = rhs / max(vb, 1e-30f);
    else
    {
        float disc = max(vb * vb + 2.0f * dv * rhs, 0.0f);
        t = (-vb + sqrt(disc)) / dv;
    }
    t = clamp(t, 0.0f, 1.0f);

    float u      = cumOm[bin] + t * ob;
    float x      = -tail + span * u;
    float pdf    = vb + dv * t;
    float logDet = log(max(pdf, 1e-30f));

    FlowStep r = { x, -logDet };
    return r;
}

// ─── CoopVec-form overloads (unpack packed [W|V] internally) ──────────────────

// Raw-conditioner-output clamp, matching TSNN.Flows.RQS's kRQSMaxRaw: a
// runaway conditioner MLP can emit an unbounded (even +-inf) raw logit here,
// same as it did for TSNN.Mixtures.Histogram (see FINDING.md's DF-N/NSF-
// Quadratic writeup) -- clamping before any softmax/normalisation math in
// pqForward/pqInverse prevents inf/NaN from ever entering this bijection's
// decode (this file's own vertex normalisation divides by a sum of exp(V),
// which is more numerically fragile than PiecewiseLinear's plain softmax and
// is exactly what was observed to blow up in practice).
static const float kPQMaxRaw = 15.0f;

[Differentiable]
FlowStep pqForward<let K : uint>(float x, CoopVec<half, 2u * K + 1u> cv, no_diff float tail)
{
    let arr = CoopVecToArray(cv);
    float W[K], V[K + 1u];
    [ForceUnroll] for (uint i = 0u; i < K;      i++) W[i] = clamp(arr[i], -kPQMaxRaw, kPQMaxRaw);
    [ForceUnroll] for (uint i = 0u; i < K + 1u; i++) V[i] = clamp(arr[K + i], -kPQMaxRaw, kPQMaxRaw);
    return pqForward<K>(x, W, V, tail);
}

FlowStep pqInverse<let K : uint>(float y, CoopVec<half, 2u * K + 1u> cv, float tail)
{
    let arr = CoopVecToArray(cv);
    float W[K], V[K + 1u];
    [ForceUnroll] for (uint i = 0u; i < K;      i++) W[i] = clamp(arr[i], -kPQMaxRaw, kPQMaxRaw);
    [ForceUnroll] for (uint i = 0u; i < K + 1u; i++) V[i] = clamp(arr[K + i], -kPQMaxRaw, kPQMaxRaw);
    return pqInverse<K>(y, W, V, tail);
}
