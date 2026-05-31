// Config.slang - MLP layout and parameter-offset computation for the image-learn example.
// Mirror of the SlangMLP/Config.slang pattern, but importing TSNN instead of RTXNS.

import TSNN.Encodings.HashGrid2D;
import TSNN.Encodings.FeatureBuffer;
import TSNN.LinearOps;
import TSNN.CoopVec;
import TSNN.Utils.Math;
import TSNN.Utils.MLP;
import TSNN.Activations;

static constexpr const uint kHashLevels = 16u;
static constexpr const uint kHashFeaturesPerLevel = 2u;
static constexpr const uint kHashLog2TableSize = 19u;
static constexpr const uint kHashTableSize = 1u << kHashLog2TableSize;
static constexpr const uint kHashBaseResolution = 16u;
static constexpr const float kHashPerLevelScale = 1.5f;

static constexpr const uint kEncodingParamElementCount = kHashLevels * kHashTableSize * kHashFeaturesPerLevel;
static constexpr const MLP<kHashLevels * kHashFeaturesPerLevel, 64, 3, 3> kMLP = MLP<kHashLevels * kHashFeaturesPerLevel, 64, 3, 3>();
static constexpr const uint kParamElementCount = kMLP.end / sizeof(half);

[Differentiable]
CoopVec<half, 3> forward(FeatureBuffer encodingParams, MatrixBiasBuffer weights, no_diff float2 uv)
{
    let input = encodeHashGrid<half, kHashLevels, kHashFeaturesPerLevel>(
        encodingParams, uv, kHashTableSize, float(kHashBaseResolution), kHashPerLevelScale);

    return kMLP.forward<LeakyReLU>(weights, input);
}

CoopVec<half, 3> forward(ByteAddressBuffer enc, ByteAddressBuffer w, no_diff float2 uv)
{
    return forward(FeatureBuffer(enc), MatrixBiasBuffer(w), uv);
}
