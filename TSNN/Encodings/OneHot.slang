/// One-hot encoding of an integer index into a float array of length N.
float[N] oneHot<let N : int>(uint index)
{
    float arr[N];
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
        arr[i] = (uint(i) == index) ? 1.0f : 0.0f;
    return arr;
}

/// Same encoding, directly in element type T (e.g. half) -- lets a call site skip a
/// separate float->T cast loop when it needs the one-hot in a non-float array already
/// (overloads on generic-argument count, so existing `oneHot<N>(...)` call sites are
/// unaffected).
T[N] oneHot<T : IFloat, let N : int>(uint index)
{
    T arr[N];
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
        arr[i] = T(uint(i) == index ? 1.0f : 0.0f);
    return arr;
}
