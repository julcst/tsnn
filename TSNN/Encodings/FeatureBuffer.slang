// Differentiable GPU buffer wrapper for learnable hash-grid feature tables.
// Provides loadVec (vector<T,N>) and loadCoop (CoopVec<T,N>) with correct
// backward derivatives for gradient accumulation.

import CoopVec;
import Utils.Math;

// Coalesced init for a learnable feature table, stored as `T` (e.g. half): tiny
// symmetric random values, the standard hash-grid scheme (instant-ngp uses
// U(±1e-4) so the initial encoding is near-zero and the MLP drives early
// learning).  Owns the whole strided loop; pass the reset dispatch's thread id /
// count and the table's element count.
void initFeatures<T : __BuiltinFloatingPointType>(
    RWByteAddressBuffer features, uint count, uint tid, uint threadCount, float scale = 1e-4f)
{
    for (uint i = tid; i < count; i += threadCount)
    {
        let u = 2.0f * uintToUnitFloat(pcg2d(uint2(i, 0xa511e9b3u)).x) - 1.0f;
        features.StoreAligned<T>(i * uint(sizeof(T)), T(u * scale));
    }
}

struct FeatureBufferGrad : IDifferentiablePtrType
{
    typealias Differential = FeatureBufferGrad;
    __init(RWByteAddressBuffer buf) { buffer = buf; }
    RWByteAddressBuffer buffer;
}

struct FeatureBuffer : IDifferentiablePtrType
{
    typealias Differential = FeatureBufferGrad;
    __init(ByteAddressBuffer buf) { buffer = buf; }
    ByteAddressBuffer buffer;
}

// Load an N-wide float vector; backward scatters gradients via F32 atomics.
// The gradient buffer must store float32 (not float16).
[BackwardDerivative(loadVec_bwd)]
vector<T, N> loadVec<T : __BuiltinFloatingPointType, int N>(FeatureBuffer b, int index)
{
    return b.buffer.LoadAligned<vector<T,N>>(index * N * sizeof(T));
}

void loadVec_bwd<T : __BuiltinFloatingPointType, int N>(DifferentialPtrPair<FeatureBuffer> b, int index, vector<T, N> dOut)
{
    float o;
    // NOTE: InterlockedAdd is only well supported for F32
    [ForceUnroll]
    for (int i = 0; i < N; ++i) b.d.buffer.InterlockedAddF32((index * N + i) * sizeof(float), dOut[i].toFloat(), o);
    // TODO: Use faster _NvInterlockedAddF16x2 when available
}

[BackwardDerivative(load_bwd)]
CoopVec<T, N> load<T : __BuiltinFloatingPointType, int N>(FeatureBuffer b, int index)
{
    let stride = N * sizeof(T); // NOTE: Documentation says aligned to 16 but it works without it
    return coopVecLoad<N, T>(b.buffer, index * stride);
}

void load_bwd<T : __BuiltinFloatingPointType, int N>(DifferentialPtrPair<FeatureBuffer> b, int index, CoopVec<T, N> dOut)
{
    let stride = N * sizeof(T); // NOTE: Documentation says aligned to 16 but it works without it
    coopVecReduceSumAccumulate(dOut, b.d.buffer, index * stride);
}
