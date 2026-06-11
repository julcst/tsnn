/// Frequency (positional / NeRF-style) encoding.
/// Expands each input component into NUM_SCALES pairs of (sin, cos) at
/// doubling frequencies starting at pi*x. Output size = NUM_SCALES * N * 2.

import TSNN.CoopVec;

[Differentiable]
CoopVec<T, NUM_SCALES * PARAMS_COUNT * 2> encodeFrequency<
    T            : __BuiltinFloatingPointType,
    let PARAMS_COUNT : int,
    let NUM_SCALES   : int>(CoopVec<T, PARAMS_COUNT> input)
{
    var v = VectorFromCoopVec(input);
    float arr[NUM_SCALES * PARAMS_COUNT * 2];

    [ForceUnroll]
    for (int i = 0; i < PARAMS_COUNT; ++i)
    {
        const int base = i * NUM_SCALES * 2;
        float sn, cn;
        sincos(v[i] * float.getPi(), sn, cn);
        arr[base + 0] = sn;
        arr[base + 1] = cn;
        [ForceUnroll]
        for (int j = 1; j < NUM_SCALES; ++j)
        {
            sn = 2.0f * sn * cn;
            cn = 2.0f * cn * cn - 1.0f;
            arr[base + j * 2 + 0] = sn;
            arr[base + j * 2 + 1] = cn;
        }
    }
    return CoopVecFromArray<T, NUM_SCALES * PARAMS_COUNT * 2>(arr);
}
