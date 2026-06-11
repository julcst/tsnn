// Multi-level hash-grid encodings with trilinear / bilinear interpolation.

import TSNN.Encodings.FeatureBuffer;
import TSNN.CoopVec;
import TSNN.Utils.Math;

uint hashGridIndex(uint3 cell, uint level, uint size)
{
    // Using morton code as hash function significantly improves coherence (+1.5ms) but has stripe artifacts
    // let h = morton3D(cell ^ uint3(level, level * 1664525u, level * 1013904223u));
    let h = pcg3d(cell^ uint3(level, level * 1664525u, level * 1013904223u) );
    return h.x & (size - 1u);
}

// NOTE: This uses vector<T, N> so only available for N <= 4 (but larger is generally not required for encoding)
[Differentiable]
vector<T, FEATURES> encodeHashGridSingle<T : __BuiltinFloatingPointType, uint FEATURES>(FeatureBuffer features, no_diff float3 position, no_diff uint level, no_diff float resolution, no_diff uint hashTableSize)
{
    static_assert(FEATURES <= 4, "encodeHashGridSingle only supports up to 4 features due to using vector<T, N> for loading");
    float3 scaled = position * float(resolution - 1u);

    // TODO: Reorder and make procedural
    let p000 = uint3(floor(scaled));
    let p100 = p000 + uint3(1, 0, 0);
    let p010 = p000 + uint3(0, 1, 0);
    let p110 = p000 + uint3(1, 1, 0);
    let p001 = p000 + uint3(0, 0, 1);
    let p101 = p000 + uint3(1, 0, 1);
    let p011 = p000 + uint3(0, 1, 1);
    let p111 = p000 + uint3(1, 1, 1);

    float3 fracCoord = frac(scaled);
    let wx0 = (1.0 - fracCoord.x);
    let wx1 = fracCoord.x;
    let wy0 = (1.0 - fracCoord.y);
    let wy1 = fracCoord.y;
    let wz0 = (1.0 - fracCoord.z);
    let wz1 = fracCoord.z;

    let w000 = wx0 * wy0 * wz0;
    let w100 = wx1 * wy0 * wz0;
    let w010 = wx0 * wy1 * wz0;
    let w110 = wx1 * wy1 * wz0;
    let w001 = wx0 * wy0 * wz1;
    let w101 = wx1 * wy0 * wz1;
    let w011 = wx0 * wy1 * wz1;
    let w111 = wx1 * wy1 * wz1;

    uint base = level * hashTableSize;
    uint i000 = hashGridIndex(p000, level, hashTableSize) + base;
    uint i100 = hashGridIndex(p100, level, hashTableSize) + base;
    uint i010 = hashGridIndex(p010, level, hashTableSize) + base;
    uint i110 = hashGridIndex(p110, level, hashTableSize) + base;
    uint i001 = hashGridIndex(p001, level, hashTableSize) + base;
    uint i101 = hashGridIndex(p101, level, hashTableSize) + base;
    uint i011 = hashGridIndex(p011, level, hashTableSize) + base;
    uint i111 = hashGridIndex(p111, level, hashTableSize) + base;

    // Accumulate weighted features in single pass to reduce intermediate allocations
    var result = loadVec<T, FEATURES>(features, i000) * T(w000);
    result += loadVec<T, FEATURES>(features, i100) * T(w100);
    result += loadVec<T, FEATURES>(features, i010) * T(w010);
    result += loadVec<T, FEATURES>(features, i110) * T(w110);
    result += loadVec<T, FEATURES>(features, i001) * T(w001);
    result += loadVec<T, FEATURES>(features, i101) * T(w101);
    result += loadVec<T, FEATURES>(features, i011) * T(w011);
    result += loadVec<T, FEATURES>(features, i111) * T(w111);

    return result;
}

// TODO: Extract in class
[Differentiable]
T[LEVELS * FEATURES] encodeHashGrid<T : __BuiltinFloatingPointType, uint LEVELS, uint FEATURES>(FeatureBuffer features, no_diff float3 position, no_diff uint hashTableSize, no_diff float baseResolution, no_diff float scale)
{
    T output[LEVELS * FEATURES];
    float levelResolution = baseResolution;
    [ForceUnroll]
    for (uint level = 0u; level < LEVELS; ++level)
    {
        let levelFeatures = encodeHashGridSingle<T, FEATURES>(features, position, level, levelResolution, hashTableSize);
        [ForceUnroll]
        for (uint i = 0u; i < FEATURES; ++i) output[level * FEATURES + i] = levelFeatures[i];
        levelResolution *= scale;
    }
    return output;
}
