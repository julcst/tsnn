// GPU probes for tests/piecewise/test_piecewise.py.
// Kept separate from the implementation so this file only exercises its public math.
//
// K is a compile-time generic parameter on TSNN.Flows.PiecewiseLinear/Quadratic, so
// each K the Python harness wants (8, 32) gets its own thin, non-generic entry-point
// wrapper around a shared generic helper -- same reason TruncatedGMMTest.slang fixes
// COMPONENTS at module scope, just done per-entry-point here since two K values are
// needed side by side in one module.
import TSNN.Flows.PiecewiseLinear;
import TSNN.Flows.PiecewiseQuadratic;
import TSNN.Flows.Bijection;

cbuffer CB
{
    uint gCount;
    float gTail;
}

// Row layout depends on the kernel:
//   *RoundTrip*: gInputs[row] = x (or y); gParams row = K floats (linear) or 2K+1
//     floats [W(K)|V(K+1)] (quadratic); gOutputs row = [fwd.value, fwd.logAbsDet,
//     inv.value, inv.logAbsDet].
//   *Grad*: same param row layout; gOutputs[row] = logAbsDet; gParamGrads row =
//     d(logAbsDet)/d(param), same length/order as the gParams row.
StructuredBuffer<float> gInputs;
StructuredBuffer<float> gParams;
RWStructuredBuffer<float> gOutputs;
RWStructuredBuffer<float> gParamGrads;

// ─── Piecewise-linear ──────────────────────────────────────────────────────────

void linearRoundTrip<let K : uint>(uint row, no_diff float tail)
{
    float x = gInputs[row];
    float q[K];
    [ForceUnroll] for (uint i = 0; i < K; i++) q[i] = gParams[row * K + i];
    let fwd = plForward<K>(x, q, tail);
    let inv = plInverse<K>(fwd.value, q, tail);
    uint base = row * 4u;
    gOutputs[base + 0] = fwd.value;
    gOutputs[base + 1] = fwd.logAbsDet;
    gOutputs[base + 2] = inv.value;
    gOutputs[base + 3] = inv.logAbsDet;
}

[Differentiable]
float linearLogAbsDet<let K : uint>(float x, float q[K], no_diff float tail)
{
    return plForward<K>(x, q, tail).logAbsDet;
}

void linearGrad<let K : uint>(uint row, no_diff float tail)
{
    float x = gInputs[row];
    float values[K], zeros[K];
    [ForceUnroll] for (uint i = 0; i < K; i++) { values[i] = gParams[row * K + i]; zeros[i] = 0.0f; }
    var qPair = diffPair(values, zeros);
    var xPair = diffPair(x, 0.0f);
    gOutputs[row] = linearLogAbsDet<K>(x, values, tail);
    bwd_diff(linearLogAbsDet<K>)(xPair, qPair, tail, 1.0f);
    [ForceUnroll] for (uint i = 0; i < K; i++) gParamGrads[row * K + i] = qPair.d[i];
}

[numthreads(64, 1, 1)]
void linearRoundTripK8Main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;
    linearRoundTrip<8u>(tid.x, gTail);
}

[numthreads(64, 1, 1)]
void linearRoundTripK32Main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;
    linearRoundTrip<32u>(tid.x, gTail);
}

[numthreads(64, 1, 1)]
void linearGradK8Main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;
    linearGrad<8u>(tid.x, gTail);
}

[numthreads(64, 1, 1)]
void linearGradK32Main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;
    linearGrad<32u>(tid.x, gTail);
}

// ─── Piecewise-quadratic ────────────────────────────────────────────────────────

void quadRoundTrip<let K : uint>(uint row, no_diff float tail)
{
    float x = gInputs[row];
    float W[K], V[K + 1u];
    uint stride = 2u * K + 1u;
    [ForceUnroll] for (uint i = 0; i < K;      i++) W[i] = gParams[row * stride + i];
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) V[i] = gParams[row * stride + K + i];
    let fwd = pqForward<K>(x, W, V, tail);
    let inv = pqInverse<K>(fwd.value, W, V, tail);
    uint base = row * 4u;
    gOutputs[base + 0] = fwd.value;
    gOutputs[base + 1] = fwd.logAbsDet;
    gOutputs[base + 2] = inv.value;
    gOutputs[base + 3] = inv.logAbsDet;
}

[Differentiable]
float quadLogAbsDet<let K : uint>(float x, float W[K], float V[K + 1u], no_diff float tail)
{
    return pqForward<K>(x, W, V, tail).logAbsDet;
}

void quadGrad<let K : uint>(uint row, no_diff float tail)
{
    float x = gInputs[row];
    uint stride = 2u * K + 1u;
    float Wv[K], Wz[K];
    float Vv[K + 1u], Vz[K + 1u];
    [ForceUnroll] for (uint i = 0; i < K;      i++) { Wv[i] = gParams[row * stride + i]; Wz[i] = 0.0f; }
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) { Vv[i] = gParams[row * stride + K + i]; Vz[i] = 0.0f; }
    var wPair = diffPair(Wv, Wz);
    var vPair = diffPair(Vv, Vz);
    var xPair = diffPair(x, 0.0f);
    gOutputs[row] = quadLogAbsDet<K>(x, Wv, Vv, tail);
    bwd_diff(quadLogAbsDet<K>)(xPair, wPair, vPair, tail, 1.0f);
    [ForceUnroll] for (uint i = 0; i < K;      i++) gParamGrads[row * stride + i] = wPair.d[i];
    [ForceUnroll] for (uint i = 0; i < K + 1u; i++) gParamGrads[row * stride + K + i] = vPair.d[i];
}

[numthreads(64, 1, 1)]
void quadRoundTripK8Main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;
    quadRoundTrip<8u>(tid.x, gTail);
}

[numthreads(64, 1, 1)]
void quadRoundTripK32Main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;
    quadRoundTrip<32u>(tid.x, gTail);
}

[numthreads(64, 1, 1)]
void quadGradK8Main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;
    quadGrad<8u>(tid.x, gTail);
}

[numthreads(64, 1, 1)]
void quadGradK32Main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;
    quadGrad<32u>(tid.x, gTail);
}
