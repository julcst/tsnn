import Util;

struct MLPLayout<let DEPTH : uint>
{
    uint inputFeatures;
    uint hiddenFeatures;
    uint outputFeatures;

    __init(uint inputFeatures, uint hiddenFeatures, uint outputFeatures)
    {
        this.inputFeatures = inputFeatures;
        this.hiddenFeatures = hiddenFeatures;
        this.outputFeatures = outputFeatures;
    }

    uint layerInputSize(uint l)  { return (l == 0u) ? inputFeatures : hiddenFeatures; }
    uint layerOutputSize(uint l) { return (l == DEPTH) ? outputFeatures : hiddenFeatures; }
    uint layerWeightCount(uint l){ return layerInputSize(l) * layerOutputSize(l); }
    uint layerBiasCount(uint l)  { return layerOutputSize(l); }
    constexpr uint transitions() { return DEPTH + 1u; }

    uint getParamCount()
    {
        uint n = 0u;
        for (uint l = 0u; l < transitions(); ++l)
            n += layerWeightCount(l) + layerBiasCount(l);
        return n;
    }

    uint2[DEPTH + 1u] getLayerOffsets()
    {
        uint2 offsets[DEPTH + 1u];
        uint byteOff = 0;
        [ForceUnroll]
        for (uint l = 0u; l < DEPTH + 1u; ++l)
        {
            let weightOff = byteOff;
            byteOff = align(byteOff + 2u * layerWeightCount(l), 4);
            let biasOff = byteOff;
            byteOff = align(byteOff + 2u * layerBiasCount(l), 4);
            offsets[l] = uint2(weightOff, biasOff);
        }
        return offsets;
    }
};