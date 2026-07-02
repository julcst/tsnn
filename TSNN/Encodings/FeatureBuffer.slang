// Differentiable GPU buffer wrapper for learnable hash-grid feature tables.
// Provides loadVec (vector<T,N>) and loadCoop (CoopVec<T,N>) with correct
// backward derivatives for gradient accumulation.

import TSNN.CoopVec;
import TSNN.Utils.Math;

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

// CAS-based float atomic add — works without VK_EXT_shader_atomic_float, which
// InterlockedAddF32 silently relies on (and no-ops to zero where unsupported).
void atomicAddF32(RWByteAddressBuffer buf, uint byteAddr, float value)
{
    uint expected = buf.Load<uint>(byteAddr);
    for (;;)
    {
        float updated = asfloat(expected) + value;
        uint prev;
        buf.InterlockedCompareExchange(byteAddr, expected, asuint(updated), prev);
        if (prev == expected) break;
        expected = prev;
    }
}

void loadVec_bwd<T : __BuiltinFloatingPointType, int N>(DifferentialPtrPair<FeatureBuffer> b, int index, vector<T, N> dOut)
{
    // Scrub non-finite per-element BEFORE the atomic add: the score-function /
    // flow backward occasionally produces an Inf/NaN feature gradient for a high-
    // variance sample, and a single non-finite atomicAdd poisons that table entry
    // permanently (Inf+x = Inf).  Dropping the bad sample lets the finite ones
    // through, matching the optimiser's sanitize() philosophy.
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        let g = dOut[i].toFloat();
        if (isfinite(g))
            atomicAddF32(b.d.buffer, (index * N + i) * sizeof(float), g);
    }
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
