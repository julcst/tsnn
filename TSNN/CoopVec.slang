// CoopVec autodiff support: IDifferentiable extension, element-wise ops,
// backward derivatives, and float↔CoopVec conversion utilities.
//
// Merges the three-file split used by RTXNS into a single coherent module.
// Import this module first; other TSNN modules depend on it.

// Extend the built-in CoopVec<T, K> type to participate in Slang autodiff.
extension<T : __BuiltinFloatingPointType, let K : int> CoopVec<T, K> : IDifferentiable
{
    typealias Differential = CoopVec<T, K>;
}

// ─── Element-wise primitives ──────────────────────────────────────────────────

CoopVec<T, K> relu<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    return max(v, CoopVec<T, K>(T(0.)));
}

CoopVec<T, K> leakyReLU<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> p, no_diff T a = T(0.01))
{
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
        p[i] = p[i] < T(0.) ? a * p[i] : p[i];
    return p;
}

CoopVec<T, K> sigmoid<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    let c1 = CoopVec<T, K>(T(1.));
    return c1 / (c1 + exp(CoopVec<T, K>(T(-1.)) * v));
}

CoopVec<T, K> copysign<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> mag, CoopVec<T, K> sgn)
{
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
        mag[i] = copysign(mag[i], sgn[i]);
    return mag;
}

CoopVec<T, K> abs<T : __BuiltinFloatingPointType, let K : int>(CoopVec<T, K> v)
{
    [ForceUnroll]
    for (int i = 0; i < K; ++i)
        v[i] = abs(v[i]);
    return v;
}

// ─── Backward derivatives ─────────────────────────────────────────────────────

// exp is a builtin; register its backward so autodiff can differentiate through it.
[BackwardDerivativeOf(exp)]
void exp_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p0,
    CoopVec<T, K>.Differential dResult)
{
    p0 = diffPair(p0.p, dResult * exp(p0.p));
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

[BackwardDerivativeOf(sigmoid)]
void sigmoid_bwd<T : __BuiltinFloatingPointType, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> p0,
    CoopVec<T, K>.Differential dResult)
{
    let s = sigmoid(p0.p);
    p0 = diffPair(p0.p, dResult * s * (CoopVec<T, K>(T(1.)) - s));
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
