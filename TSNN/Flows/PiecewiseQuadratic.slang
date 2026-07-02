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

import TSNN.CoopVec;
import TSNN.Flows.Bijection;

// ─── Array-form forward transform ────────────────────────────────────────────

[Differentiable]
FlowStep pqForward<let K : uint>(float x, float W[K], float V[K + 1u], no_diff float tail)
{
    if (x <= -tail || x >= tail) { FlowStep r = { x, 0.0f }; return r; }

    // widths ω via softmax(W); Σ ω = 1
    float om[K];
    float maxW = W[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxW = max(maxW, W[i]);
    float sumW = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { om[i] = exp(W[i] - maxW); sumW += om[i]; }
    float invW = 1.0f / sumW;
    [ForceUnroll] for (uint i = 0; i < K; i++) om[i] *= invW;

    // vertices: exp(V), normalised so ∫ pdf = Σ ω·(v_i+v_{i+1})/2 = 1
    float ev[K + 1u];
    float maxV = V[0];
    [ForceUnroll] for (uint i = 1; i < K + 1u; i++) maxV = max(maxV, V[i]);
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) ev[i] = exp(V[i] - maxV);
    float denom = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) denom += 0.5f * (ev[i] + ev[i + 1]) * om[i];
    float invD = 1.0f / denom;
    float v[K + 1u];
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) v[i] = ev[i] * invD;

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
    float maxW = W[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxW = max(maxW, W[i]);
    float sumW = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { om[i] = exp(W[i] - maxW); sumW += om[i]; }
    float invW = 1.0f / sumW;
    [ForceUnroll] for (uint i = 0; i < K; i++) om[i] *= invW;

    float ev[K + 1u];
    float maxV = V[0];
    [ForceUnroll] for (uint i = 1; i < K + 1u; i++) maxV = max(maxV, V[i]);
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) ev[i] = exp(V[i] - maxV);
    float denom = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) denom += 0.5f * (ev[i] + ev[i + 1]) * om[i];
    float invD = 1.0f / denom;
    float v[K + 1u];
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) v[i] = ev[i] * invD;

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

[Differentiable]
FlowStep pqForward<let K : uint>(float x, CoopVec<half, 2u * K + 1u> cv, no_diff float tail)
{
    let arr = CoopVecToArray(cv);
    float W[K], V[K + 1u];
    [ForceUnroll] for (uint i = 0u; i < K;      i++) W[i] = arr[i];
    [ForceUnroll] for (uint i = 0u; i < K + 1u; i++) V[i] = arr[K + i];
    return pqForward<K>(x, W, V, tail);
}

FlowStep pqInverse<let K : uint>(float y, CoopVec<half, 2u * K + 1u> cv, float tail)
{
    let arr = CoopVecToArray(cv);
    float W[K], V[K + 1u];
    [ForceUnroll] for (uint i = 0u; i < K;      i++) W[i] = arr[i];
    [ForceUnroll] for (uint i = 0u; i < K + 1u; i++) V[i] = arr[K + i];
    return pqInverse<K>(y, W, V, tail);
}
