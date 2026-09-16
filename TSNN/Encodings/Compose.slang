/// Helpers for assembling an MLP input CoopVec from heterogeneous encodings.
/// All functions are differentiable with manually-registered backward passes.

import TSNN.CoopVec;

[BackwardDerivative(SetCoop_bwd)]
void SetCoop<let N : int, let M : int>(inout CoopVec<half, N> output, half[M] array, int offset)
{
    [ForceUnroll] 
    for(int i = 0; i < M; i++)
    {
        output[i + offset] = array[i];
    }
}

void SetCoop_bwd<let N : int, let M : int>(inout DifferentialPair<CoopVec<half, N>> output, inout DifferentialPair<half[M]> array, int offset)
{
    var dArray = array.d;
    [ForceUnroll] 
    for(int i = 0; i < M; i++)
    {
        dArray[i] += output.d[i + offset];
    }
    array = diffPair(array.p, dArray);
}

[BackwardDerivative(ComposeCoop_bwd)]
CoopVec<half, N1 + N2> ComposeCoop<let N1 : int, let N2 : int>(half[N1] a1, half[N2] a2)
{
    CoopVec<half, N1 + N2> output;
    [ForceUnroll] 
    for(int i = 0; i < N1; i++)
    {
        output[i] = a1[i];
    }
    [ForceUnroll] 
    for(int i = 0; i < N2; i++)
    {
        output[i + N1] = a2[i];
    }
    return output;
}

void ComposeCoop_bwd<let N1 : int, let N2 : int>(inout DifferentialPair<half[N1]> a1, inout DifferentialPair<half[N2]> a2, CoopVec<half, N1 + N2> dOutput)
{
    var dA1 = a1.d;
    [ForceUnroll] 
    for(int i = 0; i < N1; i++)
    {
        dA1[i] += dOutput[i];
    }
    a1 = diffPair(a1.p, dA1);

    var dA2 = a2.d;
    [ForceUnroll] 
    for(int i = 0; i < N2; i++)
    {
        dA2[i] += dOutput[i + N1];
    }
    a2 = diffPair(a2.p, dA2);
}


// Concatenate three half[] arrays into one CoopVec in a SINGLE construction (fresh
// output, not an inout accumulate).  Use this instead of chained SetCoop calls: the
// inout-CoopVec form leaves the output differential's untouched lanes uninitialised on
// this backend, and SetCoop_bwd reads them back as Inf/NaN, poisoning gradients that
// flow back into a learnable input (e.g. hash-grid features).  ComposeEnc/ComposeCoop
// build a fresh CoopVec whose backward writes every input differential at once.
[BackwardDerivative(ComposeCoop3_bwd)]
CoopVec<half, N1 + N2 + N3> ComposeCoop3<let N1 : int, let N2 : int, let N3 : int>(
    half[N1] a1, half[N2] a2, half[N3] a3)
{
    CoopVec<half, N1 + N2 + N3> output;
    [ForceUnroll] for (int i = 0; i < N1; i++) output[i] = a1[i];
    [ForceUnroll] for (int i = 0; i < N2; i++) output[N1 + i] = a2[i];
    [ForceUnroll] for (int i = 0; i < N3; i++) output[N1 + N2 + i] = a3[i];
    return output;
}

void ComposeCoop3_bwd<let N1 : int, let N2 : int, let N3 : int>(
    inout DifferentialPair<half[N1]> a1, inout DifferentialPair<half[N2]> a2,
    inout DifferentialPair<half[N3]> a3, CoopVec<half, N1 + N2 + N3> dOutput)
{
    var dA1 = a1.d;
    [ForceUnroll] for (int i = 0; i < N1; i++) dA1[i] += dOutput[i];
    a1 = diffPair(a1.p, dA1);
    var dA2 = a2.d;
    [ForceUnroll] for (int i = 0; i < N2; i++) dA2[i] += dOutput[N1 + i];
    a2 = diffPair(a2.p, dA2);
    var dA3 = a3.d;
    [ForceUnroll] for (int i = 0; i < N3; i++) dA3[i] += dOutput[N1 + N2 + i];
    a3 = diffPair(a3.p, dA3);
}

// TODO: Make procedural
[BackwardDerivative(ComposeEnc_bwd)]
CoopVec<half, N1 + N2 + N3 + N4 + N5 + N6> ComposeEnc<int N1, int N2, int N3, int N4, int N5, int N6>(
    Array<half, N1> a1,
    Array<half, N2> a2,
    Array<half, N3> a3,
    vector<half, N4> a4,
    vector<half, N5> a5,
    vector<half, N6> a6)
{
    CoopVec<half, N1 + N2 + N3 + N4 + N5 + N6> output;
    [ForceUnroll]
    for(int i = 0; i < N1; i++) output[i] = a1[i];
    [ForceUnroll]
    for(int i = 0; i < N2; i++) output[i + N1] = a2[i];
    [ForceUnroll]
    for(int i = 0; i < N3; i++) output[i + N1 + N2] = a3[i];
    [ForceUnroll]
    for(int i = 0; i < N4; i++) output[i + N1 + N2 + N3] = a4[i];
    [ForceUnroll]
    for(int i = 0; i < N5; i++) output[i + N1 + N2 + N3 + N4] = a5[i];
    [ForceUnroll]
    for(int i = 0; i < N6; i++) output[i + N1 + N2 + N3 + N4 + N5] = a6[i];
    return output;
}

// Backward for ComposeEnc (six-part)
void ComposeEnc_bwd<int N1, int N2, int N3, int N4, int N5, int N6>(
    inout DifferentialPair<Array<half, N1>> a1,
    inout DifferentialPair<Array<half, N2>> a2,
    inout DifferentialPair<Array<half, N3>> a3,
    inout DifferentialPair<vector<half, N4>> a4,
    inout DifferentialPair<vector<half, N5>> a5,
    inout DifferentialPair<vector<half, N6>> a6,
    CoopVec<half, N1 + N2 + N3 + N4 + N5 + N6> dOutput)
{
    var dA1 = a1.d;
    [ForceUnroll]
    for(int i = 0; i < N1; i++) dA1[i] += dOutput[i];
    a1 = diffPair(a1.p, dA1);

    var dA2 = a2.d;
    [ForceUnroll]
    for(int i = 0; i < N2; i++) dA2[i] += dOutput[i + N1];
    a2 = diffPair(a2.p, dA2);

    var dA3 = a3.d;
    [ForceUnroll]
    for(int i = 0; i < N3; i++) dA3[i] += dOutput[i + N1 + N2];
    a3 = diffPair(a3.p, dA3);

    var dA4 = a4.d;
    [ForceUnroll]
    for(int i = 0; i < N4; i++) dA4[i] += dOutput[i + N1 + N2 + N3];
    a4 = diffPair(a4.p, dA4);

    var dA5 = a5.d;
    [ForceUnroll]
    for(int i = 0; i < N5; i++) dA5[i] += dOutput[i + N1 + N2 + N3 + N4];
    a5 = diffPair(a5.p, dA5);

    var dA6 = a6.d;
    [ForceUnroll]
    for(int i = 0; i < N6; i++) dA6[i] += dOutput[i + N1 + N2 + N3 + N4 + N5];
    a6 = diffPair(a6.p, dA6);
}

// Concatenate a CoopVec (the differentiable head) with a no_diff half[] tail into a fresh
// CoopVec.  Every other primitive in this file takes half[] on BOTH sides -- this one exists
// for a conditioner input built from a CoopVec that already came out of an upstream
// differentiable computation (e.g. a trainable encoder's output), concatenated with a
// constant per-sample encoding (a one-hot cell, a triangle-wave coordinate, ...).
//
// A CoopVecToArray(head) + ComposeCoop(...) round trip is NOT equivalent and must not be used
// instead: the registered CoopVecFromArray/CoopVecToArray backward pair drops the incoming
// differential rather than threading it through on this Slang/Vulkan backend, so `head`'s
// upstream parameters (e.g. a hash-grid feature table) would silently stop receiving
// gradient while the loss curve still looked plausible -- see
// scripts/nisfit/NgrCompose.slang (this function's original, byte-identical home; duplicated
// here so scripts/nisbench's arm modules can import it from TSNN directly without depending
// on nisfit) and [[project_nis_hashgrid_grad_bugs]].
[BackwardDerivative(ComposeCoopVecPlus_bwd)]
CoopVec<half, N + M> ComposeCoopVecPlus<let N : int, let M : int>(
    CoopVec<half, N> head, no_diff half[M] tail)
{
    CoopVec<half, N + M> output;
    [ForceUnroll] for (int i = 0; i < N; ++i) output[i] = head[i];
    [ForceUnroll] for (int i = 0; i < M; ++i) output[N + i] = tail[i];
    return output;
}

// Accumulating backward, mirroring SetCoop_bwd/ComposeCoop_bwd/ComposeCoop3_bwd/ComposeEnc_bwd's
// pattern: read the existing accumulated differential, add this call's contribution, write back
// -- required so `head` (a CoopVec, unlike this file's other primitives whose primal parameters
// are already plain half[]/Array<half,N>) accumulates correctly when consumed at more than one
// call site inside a single [Differentiable] function body (see this function's own history:
// it used to overwrite `head`'s differential outright, discarding whatever another consumer of
// the SAME `head` value had already contributed -- see scripts/nisbench's FINDING.md).
//
// Routed through a PLAIN half[N] intermediate rather than any direct CoopVec-to-CoopVec
// per-lane copy: `head.d[i]`/`dOutput[i]` (CoopVec -> scalar) and `dHead[i] = dHeadArr[i]`
// (scalar/array -> CoopVec) are both the same primitive shape every other function in this file
// already uses successfully; a fused CoopVec[i] = CoopVec[i] assignment inside a plain (not
// [Differentiable]-transformed) function body was measured to silently produce garbage on this
// Slang/Vulkan backend -- the SAME textual pattern compiles correctly inside ComposeCoopVecPlus's
// own [Differentiable] forward above, but not here.
void ComposeCoopVecPlus_bwd<let N : int, let M : int>(
    inout DifferentialPair<CoopVec<half, N>> head, no_diff half[M] tail,
    CoopVec<half, N + M> dOutput)
{
    half dHeadArr[N];
    [ForceUnroll] for (int i = 0; i < N; ++i) dHeadArr[i] = head.d[i] + dOutput[i];
    CoopVec<half, N> dHead;
    [ForceUnroll] for (int i = 0; i < N; ++i) dHead[i] = dHeadArr[i];
    head = diffPair(head.p, dHead);
}
