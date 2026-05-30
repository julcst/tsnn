// Config.slang - MLP layout and parameter-offset computation for the image-learn example.
// Mirror of the SlangMLP/Config.slang pattern, but importing TSNN instead of RTXNS.

import TSNN.Encodings.HashGrid2D;
import TSNN.Encodings.FeatureBuffer;
import TSNN.LinearOps;
import TSNN.CoopVec;
import TSNN.Utils.Math;

static constexpr const uint kInputSize    = 32u;   // 16 hash levels × 2 features
static constexpr const uint kHiddenSize   = 64u;
static constexpr const uint kOutputSize   = 3u;
static constexpr const uint kHiddenLayers = 4u;
static constexpr const uint kTransitions  = kHiddenLayers + 1u;

static constexpr const uint kHashLevels          = 16u;
static constexpr const uint kHashFeaturesPerLevel = 2u;
static constexpr const uint kHashLog2TableSize    = 19u;
static constexpr const uint kHashTableSize        = 1u << kHashLog2TableSize;
static constexpr const uint kHashBaseResolution   = 16u;
static constexpr const float kHashPerLevelScale   = 1.5f;

static constexpr const float kOptimizerLossScale = 1.0f;
static constexpr const float kAdamBeta1  = 0.9f;
static constexpr const float kAdamBeta2  = 0.999f;
static constexpr const float kAdamEpsilon = 1e-8f;

constexpr uint layerInputSize(uint l)  { return (l == 0u) ? kInputSize : kHiddenSize; }
constexpr uint layerOutputSize(uint l) { return (l == kTransitions-1u) ? kOutputSize : kHiddenSize; }
constexpr uint layerWeightCount(uint l){ return layerInputSize(l) * layerOutputSize(l); }
constexpr uint layerBiasCount(uint l)  { return layerOutputSize(l); }

constexpr uint getParamCount()
{
    uint n = 0u;
    for (uint l = 0u; l < kTransitions; ++l)
        n += layerWeightCount(l) + layerBiasCount(l);
    return n;
}

static constexpr const uint kParamElementCount          = getParamCount();
static constexpr const uint kEncodingParamElementCount  = kHashLevels * kHashTableSize * kHashFeaturesPerLevel;

static constexpr uint kParamBufferSize         = align(kParamElementCount         * sizeof(float16_t), 4);
static constexpr uint kMomentsBufferSize        =        kParamElementCount         * sizeof(float);
static constexpr uint kEncodingParamBufferSize  = align(kEncodingParamElementCount * sizeof(float16_t), 4);
static constexpr uint kEncodingGradBufferSize   = align(kEncodingParamElementCount * sizeof(float), 4);
static constexpr uint kEncodingMomentsBufferSize= align(kEncodingParamElementCount * sizeof(float), 4);

constexpr uint2[kTransitions] getLayerOffsets()
{
    uint2 offsets[kTransitions];
    uint byteOff = 0;
    [ForceUnroll]
    for (uint l = 0u; l < kTransitions; ++l)
    {
        let weightOff = byteOff;
        byteOff = align(byteOff + 2u * layerWeightCount(l), 4);
        let biasOff = byteOff;
        byteOff = align(byteOff + 2u * layerBiasCount(l), 4);
        offsets[l] = uint2(weightOff, biasOff);
    }
    return offsets;
}

static constexpr const uint2 layerOffsets[kTransitions] = getLayerOffsets();
static constexpr const CoopVecMatrixLayout kMatrixLayout = CoopVecMatrixLayout::TrainingOptimal;
static constexpr const CoopVecComponentType kComponentType = CoopVecComponentType::Float16;
static constexpr const half kReluLeak = 0.01h;

[Differentiable]
CoopVec<half, kOutputSize> forward(
    FeatureBuffer encodingParams,
    MatrixBiasBuffer weights,
    no_diff float2 uv)
{
    let input = encodeHashGrid<half, kHashLevels, kHashFeaturesPerLevel>(
        encodingParams, uv, kHashTableSize,
        float(kHashBaseResolution), kHashPerLevelScale);

    var h = leakyReLU(LinearOp<half, kHiddenSize>(input, weights, layerOffsets[0], kMatrixLayout, kComponentType), kReluLeak);
    [ForceUnroll]
    for (int i = 1; i < kHiddenLayers; ++i)
        h = leakyReLU(LinearOp<half, kHiddenSize>(h, weights, layerOffsets[i], kMatrixLayout, kComponentType), kReluLeak);

    return LinearOp<half, kOutputSize>(h, weights, layerOffsets[kHiddenLayers], kMatrixLayout, kComponentType);
}

CoopVec<half, kOutputSize> forward(ByteAddressBuffer enc, ByteAddressBuffer w, no_diff float2 uv)
{
    return forward(FeatureBuffer(enc), MatrixBiasBuffer(w), uv);
}
