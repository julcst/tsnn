// CoopVec autodiff support: IDifferentiable extension, element-wise ops,
// backward derivatives, and float↔CoopVec conversion utilities.
//
// Merges the three-file split used by RTXNS into a single coherent module.
// Import this module first; other TSNN modules depend on it.
//
// Layout (each section sorted alphabetically):
//   1. IDifferentiable extension
//   2. Element-wise math primitives (abs, copysign)
//   3. Activations — each forward co-located with its backward
//   4. Builtin backward (exp)
//   5. Float array / vector ↔ CoopVec conversions

// Extend the built-in CoopVec<T, K> type to participate in Slang autodiff.
extension<T : __BuiltinFloatingPointType, let K : int> CoopVec<T, K> : IDifferentiable
{
    typealias Differential = CoopVec<T, K>;
}

// ─── Element-wise math primitives ─────────────────────────────────────────────

CoopVec<T, K> abs<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
        v[i] = abs(v[i]);
    return v;
}

CoopVec<T, K> copysign<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> mag, CoopVec<T, K> sgn)
{
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
        mag[i] = copysign(mag[i], sgn[i]);
    return mag;
}

// ─── Activations ──────────────────────────────────────────────────────────────
//
// Each activation is element-wise and carries its own backward: the builtin
// CoopVec operators are not auto-differentiable here, so every activation needs
// an explicitly registered derivative ([BackwardDerivative] on the forward, or
// [BackwardDerivativeOf] for the simple ones).  The forward and the local
// derivative are evaluated in float for numerical headroom, then stored back in
// T.  The smooth activations (Swish/GELU/Mish) keep a non-zero gradient for x<0,
// which avoids the dead-unit stalls a hard ReLU/LeakyReLU can cause and tends to
// stabilise training of the conditioner MLPs.

// ELU:  f(x) = x>0 ? x : α·(exp(x)−1)
[BackwardDerivative(elu_bwd)]
CoopVec<T, K> elu<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v, no_diff float alpha = 1.0f)
{
    CoopVec<T, K> o;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        float x = v[i].toFloat();
        o[i] = T(x > 0.0f ? x : alpha * (exp(x) - 1.0f));
    }
    return o;
}

void elu_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p, float alpha, CoopVec<T, K> dResult)
{
    CoopVec<T, K> d;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        float x  = p.p[i].toFloat();
        float df = x > 0.0f ? 1.0f : alpha * exp(x);
        d[i] = T(df * dResult[i].toFloat());
    }
    p = diffPair(p.p, d);
}

// GELU (tanh approximation):  f(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
static const float kGeluC = 0.7978845608028654f; // √(2/π)

[BackwardDerivative(gelu_bwd)]
CoopVec<T, K> gelu<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    CoopVec<T, K> o;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        float x = v[i].toFloat();
        float g = kGeluC * (x + 0.044715f * x * x * x);
        o[i] = T(0.5f * x * (1.0f + tanh(g)));
    }
    return o;
}

void gelu_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p, CoopVec<T, K> dResult)
{
    CoopVec<T, K> d;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        float x     = p.p[i].toFloat();
        float g     = kGeluC * (x + 0.044715f * x * x * x);
        float th    = tanh(g);
        float dg    = kGeluC * (1.0f + 3.0f * 0.044715f * x * x);
        float df    = 0.5f * (1.0f + th) + 0.5f * x * (1.0f - th * th) * dg;
        d[i] = T(df * dResult[i].toFloat());
    }
    p = diffPair(p.p, d);
}

// Leaky ReLU:  f(x) = x>0 ? x : a·x
CoopVec<T, K> leakyReLU<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> p, no_diff T a = T(0.01))
{
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
        p[i] = p[i] < T(0.) ? a * p[i] : p[i];
    return p;
}

[BackwardDerivativeOf(leakyReLU)]
void leakyReLU_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p0,
    T a,
    CoopVec<T, K>.Differential dResult)
{
    CoopVec<T, K> d;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
        d[i] = p0.p[i] > T(0.) ? dResult[i] : a * dResult[i];
    p0 = diffPair(p0.p, d);
}

// Mish:  f(x) = x·tanh(softplus(x)),  softplus(x) = log(1 + exp(x))
[BackwardDerivative(mish_bwd)]
CoopVec<T, K> mish<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    CoopVec<T, K> o;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        float x  = v[i].toFloat();
        float sp = x > 20.0f ? x : log(1.0f + exp(x));
        o[i] = T(x * tanh(sp));
    }
    return o;
}

void mish_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p, CoopVec<T, K> dResult)
{
    CoopVec<T, K> d;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        float x   = p.p[i].toFloat();
        float sp  = x > 20.0f ? x : log(1.0f + exp(x));
        float th  = tanh(sp);
        float sig = 1.0f / (1.0f + exp(-x));
        float df  = th + x * (1.0f - th * th) * sig; // d/dx [x·tanh(sp)]
        d[i] = T(df * dResult[i].toFloat());
    }
    p = diffPair(p.p, d);
}

// ReLU:  f(x) = max(x, 0)
CoopVec<T, K> relu<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    return max(v, CoopVec<T, K>(T(0.)));
}

[BackwardDerivativeOf(relu)]
void relu_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p0,
    CoopVec<T, K>.Differential dResult)
{
    CoopVec<T, K> d;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
        d[i] = p0.p[i] > T(0.) ? dResult[i] : T(0.);
    p0 = diffPair(p0.p, d);
}

// Sigmoid:  f(x) = 1 / (1 + exp(−x))
CoopVec<T, K> sigmoid<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    let c1 = CoopVec<T, K>(T(1.));
    return c1 / (c1 + exp(CoopVec<T, K>(T(-1.)) * v));
}

[BackwardDerivativeOf(sigmoid)]
void sigmoid_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p0,
    CoopVec<T, K>.Differential dResult)
{
    let s = sigmoid(p0.p);
    p0 = diffPair(p0.p, dResult * s * (CoopVec<T, K>(T(1.)) - s));
}

// Swish / SiLU:  f(x) = x·σ(x)
[BackwardDerivative(swish_bwd)]
CoopVec<T, K> swish<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    CoopVec<T, K> o;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        float x = v[i].toFloat();
        float s = 1.0f / (1.0f + exp(-x));
        o[i] = T(x * s);
    }
    return o;
}

void swish_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p, CoopVec<T, K> dResult)
{
    CoopVec<T, K> d;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        float x  = p.p[i].toFloat();
        float s  = 1.0f / (1.0f + exp(-x));
        float df = s * (1.0f + x * (1.0f - s));   // σ + x·σ(1−σ)
        d[i] = T(df * dResult[i].toFloat());
    }
    p = diffPair(p.p, d);
}

// Tanh:  f(x) = tanh(x)
[BackwardDerivative(tanhAct_bwd)]
CoopVec<T, K> tanhAct<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    CoopVec<T, K> o;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
        o[i] = T(tanh(v[i].toFloat()));
    return o;
}

void tanhAct_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p, CoopVec<T, K> dResult)
{
    CoopVec<T, K> d;
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
    {
        float th = tanh(p.p[i].toFloat());
        d[i] = T((1.0f - th * th) * dResult[i].toFloat());
    }
    p = diffPair(p.p, d);
}

// ─── Builtin backward ─────────────────────────────────────────────────────────

// exp is a builtin; register its backward so autodiff can differentiate through it.
[BackwardDerivativeOf(exp)]
void exp_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p0,
    CoopVec<T, K>.Differential dResult)
{
    p0 = diffPair(p0.p, dResult * exp(p0.p));
}

// ─── Float array / vector ↔ CoopVec conversions ───────────────────────────────
//
// CoopVecToArray / CoopVecToArray_bwd:
//   Like VectorFromCoopVec but returns float[N] instead of vector<float,N>,
//   allowing N > 4.  The backward scatters the float-array gradient back into
//   the CoopVec differential.
//
// NOTE: convert the whole vector in one call and index the returned float[].  A
// per-element differentiable accessor (a custom-backward `at(v, i)` read) does
// NOT work here: reading several elements of the same CoopVec fans out to
// multiple backward calls whose one-hot differentials are not accumulated by
// this autodiff backend, so all but one element's gradient is silently lost and
// training diverges.  CoopVecToArray's single backward writes every slot at once
// and is the correct primitive.

[BackwardDerivative(CoopVecFromArray_bwd)]
CoopVec<T, N> CoopVecFromArray<T : __BuiltinFloatingPointType, let N : int>(float arr[N])
{
    var v = CoopVec<T, N>(T(0.));
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        v[i] = T(arr[i]);
    return v;
}

void CoopVecFromArray_bwd<T : __BuiltinFloatingPointType, let N : int>(
    inout DifferentialPair<float[N]> arr,
    CoopVec<T, N> grad)
{
    float dArr[N];
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        dArr[i] = grad[i].toFloat();
    arr = diffPair(arr.p, dArr);
}

[BackwardDerivative(CoopVecFromVector_bwd)]
CoopVec<T, N> CoopVecFromVector<T : __BuiltinFloatingPointType, let N : int>(vector<float, N> v)
{
    var out = CoopVec<T, N>(T(0.));
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        out[i] = T(v[i]);
    return out;
}

void CoopVecFromVector_bwd<T : __BuiltinFloatingPointType, let N : int>(
    inout DifferentialPair<vector<float, N>> v,
    CoopVec<T, N> grad)
{
    vector<float, N> dv;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        dv[i] = grad[i].toFloat();
    v = diffPair(v.p, dv);
}

[BackwardDerivative(CoopVecToArray_bwd)]
float[N] CoopVecToArray<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> v)
{
    float arr[N];
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        arr[i] = v[i].toFloat();
    return arr;
}

void CoopVecToArray_bwd<T : __BuiltinFloatingPointType, let N : int>(
    inout DifferentialPair<CoopVec<T, N>> v,
    float arr[N])
{
    CoopVec<T, N> dv;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        dv[i] = T(arr[i]);
    v = diffPair(v.p, dv);
}

[BackwardDerivative(VectorFromCoopVec_bwd)]
vector<float, N> VectorFromCoopVec<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> v)
{
    vector<float, N> out;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        out[i] = v[i].toFloat();
    return out;
}

void VectorFromCoopVec_bwd<T : __BuiltinFloatingPointType, let N : int>(
    inout DifferentialPair<CoopVec<T, N>> v,
    vector<float, N> grad)
{
    CoopVec<T, N> dv;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
        dv[i] = T(grad[i]);
    v = diffPair(v.p, dv);
}
