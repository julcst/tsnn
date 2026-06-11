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
