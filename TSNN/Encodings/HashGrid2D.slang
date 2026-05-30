// Multi-level hash-grid encodings with trilinear / bilinear interpolation.

import FeatureBuffer;
import CoopVec;
import Utils.Math;

/// This is better for memory access pattern
uint mortonIndex(uint2 cell, uint level, uint size)
{
    uint morton = morton2D(cell);
    return morton & (size - 1u);
}

// NOTE: This uses vector<T, N> so only avaliable for N <= 4 (but larger is generally not required for encoding)
[Differentiable]
vector<T, FEATURES> encodeHashGridSingle<T : __BuiltinFloatingPointType, uint FEATURES>(FeatureBuffer features, no_diff float2 uv, no_diff uint level, no_diff float resolution, no_diff uint hashTableSize)
{
    static_assert(FEATURES <= 4, "encodeHashGridSingle only supports up to 4 features due to using vector<T, N> for loading");
    float2 scaled = uv * float(resolution - 1u);
    let p0 = uint2(floor(scaled));
    let p1 = p0 + uint2(1, 0);
    let p2 = p0 + uint2(0, 1);
    let p3 = p0 + uint2(1, 1);

    float2 fracCoord = frac(scaled);
    let w00 = (1.0 - fracCoord.x) * (1.0 - fracCoord.y);
    let w10 = fracCoord.x * (1.0 - fracCoord.y);
    let w01 = (1.0 - fracCoord.x) * fracCoord.y;
    let w11 = fracCoord.x * fracCoord.y;

    uint base = level * hashTableSize;
    uint i00 = mortonIndex(p0, level, hashTableSize) + base;
    uint i10 = mortonIndex(p1, level, hashTableSize) + base;
    uint i01 = mortonIndex(p2, level, hashTableSize) + base;
    uint i11 = mortonIndex(p3, level, hashTableSize) + base;

    let f00 = loadVec<T, FEATURES>(features, i00) * T(w00);
    let f10 = loadVec<T, FEATURES>(features, i10) * T(w10);
    let f01 = loadVec<T, FEATURES>(features, i01) * T(w01);
    let f11 = loadVec<T, FEATURES>(features, i11) * T(w11);

    return f00 + f01 + f10 + f11;
}

[Differentiable]
CoopVec<T, LEVELS * FEATURES> encodeHashGrid<T : __BuiltinFloatingPointType, uint LEVELS, uint FEATURES>(FeatureBuffer features, no_diff float2 uv, no_diff uint hashTableSize, no_diff float baseResolution, no_diff float scale)
{
    // TODO: Make this T
    float encoded[LEVELS * FEATURES];

    float levelResolution = baseResolution;
    [ForceUnroll]
    for (uint level = 0u; level < LEVELS; ++level)
    {
        let levelFeatures = encodeHashGridSingle<T, FEATURES>(features, uv, level, levelResolution, hashTableSize);
        [ForceUnroll]
        for (uint i = 0u; i < FEATURES; ++i) encoded[level * FEATURES + i] = levelFeatures[i].toFloat();
        levelResolution *= scale;
    }

    return CoopVecFromArray<T, LEVELS * FEATURES>(encoded);
}