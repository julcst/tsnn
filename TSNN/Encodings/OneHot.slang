/// One-hot encoding of an integer index into a float array of length N.
float[N] oneHot<let N : int>(uint index)
{
    float arr[N];
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
        arr[i] = (uint(i) == index) ? 1.0f : 0.0f;
    return arr;
}
