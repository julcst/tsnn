import TSNN.LinearOps;
import TSNN.CoopVec;
import TSNN.Utils.Math;
import TSNN.Activations;

// ─── SOTA weight initialisation ───────────────────────────────────────────────
//
// Fan-in / fan-out scaled schemes (He/Kaiming, Xavier/Glorot, LeCun).  Biases
// initialise to zero.  The random draw is keyed on the global parameter index,
// so init is deterministic and independent of the launch configuration.

enum MLPInit
{
    HeNormal,       // N(0, 2/fan_in)             — ReLU / LeakyReLU / GELU / Swish (default)
    HeUniform,      // U(±√(6/fan_in))
    XavierNormal,   // N(0, 2/(fan_in+fan_out))   — tanh / sigmoid conditioners
    XavierUniform,  // U(±√(6/(fan_in+fan_out)))
    LeCunNormal,    // N(0, 1/fan_in)             — SELU
};

// Per-scheme weight magnitude (std for the normal schemes, half-width for uniform).
float initScale(MLPInit s, uint fanIn, uint fanOut)
{
    switch (s)
    {
    case MLPInit::HeUniform:     return sqrt(6.0f / float(fanIn));
    case MLPInit::XavierNormal:  return sqrt(2.0f / float(fanIn + fanOut));
    case MLPInit::XavierUniform: return sqrt(6.0f / float(fanIn + fanOut));
    case MLPInit::LeCunNormal:   return sqrt(1.0f / float(fanIn));
    default:                     return sqrt(2.0f / float(fanIn)); // HeNormal
    }
}

struct MLP<
    let INPUT : uint,
    let HIDDEN : uint,
    let DEPTH : uint,
    let OUTPUT : uint,
    let LAYOUT : CoopVecMatrixLayout = CoopVecMatrixLayout::TrainingOptimal,
    let COMPONENT_TYPE : CoopVecComponentType = CoopVecComponentType::Float16
> {
    uint end;
    uint2 offsets[DEPTH + 1u];

    __init(uint offset = 0)
    {
        uint byteOff = offset;
        uint2 offsets[DEPTH + 1u];
        [ForceUnroll]
        for (uint l = 0u; l < DEPTH + 1u; ++l)
        {
            let inSize = (l == 0u) ? INPUT : HIDDEN;
            let outSize = (l == DEPTH) ? OUTPUT : HIDDEN;
            let weightOff = byteOff;
            byteOff = align(byteOff + sizeof(half) * inSize * outSize, 4);
            let biasOff = byteOff;
            byteOff = align(byteOff + sizeof(half) * outSize, 4);
            offsets[l] = uint2(weightOff, biasOff);
        }

        this.offsets = offsets;
        this.end = byteOff;
    }

    [Differentiable]
    CoopVec<half, OUTPUT> forward<A : IActivation = LeakyReLU>(MatrixBiasBuffer weights, CoopVec<half, INPUT> input, A act = A())
    {
        var h = LinearOp<half, HIDDEN>(input, weights, offsets[0], LAYOUT, COMPONENT_TYPE);
        h = act.eval(h);
        [ForceUnroll]
        for (uint l = 1u; l < DEPTH; ++l)
        {
            h = LinearOp<half, HIDDEN>(h, weights, offsets[l], LAYOUT, COMPONENT_TYPE);
            h = act.eval(h);
        }
        return LinearOp<half, OUTPUT>(h, weights, offsets[DEPTH], LAYOUT, COMPONENT_TYPE);
    }

    // SOTA-initialised value for the half-element `localElem` within this block
    // (localElem indexes [offsets[0].x, end) in half slots).  Weights are scaled
    // by their layer's fan-in/fan-out; biases and alignment padding are zero.
    // `zeroLastLayer` zeros the final layer so the block starts as the
    // identity/zero map (used by the flow couplings).
    float initValue(uint localElem, MLPInit scheme, bool zeroLastLayer)
    {
        let byteOff = offsets[0].x + localElem * uint(sizeof(half));

        [ForceUnroll]
        for (uint l = 0u; l < DEPTH + 1u; ++l)
        {
            let inSize  = (l == 0u)    ? INPUT  : HIDDEN;
            let outSize = (l == DEPTH) ? OUTPUT : HIDDEN;
            let wStart  = offsets[l].x;
            let bStart  = offsets[l].y;

            if (byteOff >= bStart && byteOff < bStart + uint(sizeof(half)) * outSize)
                return 0.0f; // bias
            if (byteOff < wStart || byteOff >= wStart + uint(sizeof(half)) * inSize * outSize)
                continue;    // not this layer's weights → keep scanning

            if (zeroLastLayer && l == DEPTH)
                return 0.0f;

            let draw = (scheme == MLPInit::HeUniform || scheme == MLPInit::XavierUniform)
                ? 2.0f * uintToUnitFloat(pcg2d(uint2(byteOff, 0x85ebca6bu)).x) - 1.0f
                : randNormal(uint2(byteOff, 0x9e3779b9u));
            return initScale(scheme, inSize, outSize) * draw;
        }
        return 0.0f; // alignment padding
    }

    // Coalesced SOTA init of this block's parameters, stored as `T` (pass half for
    // the coopvec compute buffer, float for a full-precision one).  Owns the whole
    // strided loop (reset is not perf critical), so callers just hand over the
    // dispatch's thread id / count — natural to call once per block on a multi-MLP
    // layout, since each block fills only its own logical slice.  The layout
    // (offsets/end) is defined in half slots, which equal the logical parameter
    // indices; the storage stride is sizeof(T).
    void initParams<T : __BuiltinFloatingPointType>(
        RWByteAddressBuffer params, uint tid, uint threadCount,
        MLPInit scheme = MLPInit::HeNormal, bool zeroLastLayer = false)
    {
        let lo = offsets[0].x / uint(sizeof(half));
        let hi = end / uint(sizeof(half));
        for (uint e = lo + tid; e < hi; e += threadCount)
            params.StoreAligned<T>(e * uint(sizeof(T)), T(initValue(e - lo, scheme, zeroLastLayer)));
    }

    // Mixed-precision variant: writes a compute mirror (TMirror, e.g. half) and a
    // full-precision master (TMaster, e.g. float) from the same draw — Adam updates
    // the master, the mirror feeds the forward/backward compute.
    void initParams<TMirror : __BuiltinFloatingPointType, TMaster : __BuiltinFloatingPointType>(
        RWByteAddressBuffer params, RWByteAddressBuffer master, uint tid, uint threadCount,
        MLPInit scheme = MLPInit::HeNormal, bool zeroLastLayer = false)
    {
        let lo = offsets[0].x / uint(sizeof(half));
        let hi = end / uint(sizeof(half));
        for (uint e = lo + tid; e < hi; e += threadCount)
        {
            let v = initValue(e - lo, scheme, zeroLastLayer);
            params.StoreAligned<TMirror>(e * uint(sizeof(TMirror)), TMirror(v));
            master.StoreAligned<TMaster>(e * uint(sizeof(TMaster)), TMaster(v));
        }
    }
}
