/// Helpers for assembling an MLP input CoopVec from heterogeneous encodings.
/// All functions are differentiable with manually-registered backward passes.

import CoopVec;

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
