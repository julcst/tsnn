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

// A fixed-shape MLP (INPUT -> HIDDEN [x(DEPTH-1)] -> OUTPUT, DEPTH+1 linear
// layers total) as a purely static "namespace" over its own generic shape
// parameters -- it holds no per-instance state, because there is none to
// hold: every layer's byte layout in the shared parameter buffer is
// host-supplied (a `layout` StructuredBuffer<uint2> of per-layer
// (weightOffset, biasOffset) pairs, plus this block's own `base` index into
// it), not computed here.
//
// That's a hard requirement, not a style choice: `coopVecOuterProductAccumulate`
// (the backward pass's gradient scatter, see TSNN.LinearOps) requires
// `TrainingOptimal` layout on current hardware (the Slang stdlib's own doc
// comment on it says so), and `TrainingOptimal`'s real per-matrix byte size
// is device-defined -- queried via Device::getCooperativeVectorMatrixSize
// (C++) / Device.get_coop_vec_matrix_size (SlangPy), with no shader-side
// equivalent -- and can be well above the naive `sizeof(half)*inSize*outSize`
// count a compile-time Slang `__init` could compute (confirmed on one GPU: a
// 32x32 fp16 matrix needs 3584 bytes `TrainingOptimal` vs. 2048 naive; a
// 32x1 needs 512 vs. 64). A struct that tried to compute its own offsets at
// Slang compile time would therefore silently under-allocate every layer
// for `TrainingOptimal`, aliasing/overflowing into neighbouring layers'
// storage. See FINDING.md's "TSNN.Utils.MLP: every `TrainingOptimal`-layout
// MLP was silently under-allocated..." entry for the investigation (an
// earlier version of this file computed offsets naively at compile time;
// its replacement by this host-layout scheme is that entry's actual fix).
//
// Every caller therefore needs its own small host-side "layout builder" that
// mirrors this file's own per-layer shape sequence (input, hidden, ...,
// hidden, output) and queries the real per-layer weight-matrix size for
// `LAYOUT`; see examples/ndebench/benchmark.py's `compute_layout()` for the
// SlangPy version and each render pass's own C++ equivalent.
struct MLP<
    let INPUT : uint,
    let HIDDEN : uint,
    let DEPTH : uint,
    let OUTPUT : uint,
    let LAYOUT : CoopVecMatrixLayout = CoopVecMatrixLayout::TrainingOptimal,
    let COMPONENT_TYPE : CoopVecComponentType = CoopVecComponentType::Float16
> {
    // layer `l`'s (inputWidth, outputWidth), l in [0, DEPTH]: layer 0 reads
    // INPUT, layer DEPTH writes OUTPUT, every layer in between is HIDDEN on
    // both sides. Shared by initValue's layer scan and by host-side layout
    // builders that want it (rather than re-deriving this from INPUT/HIDDEN/
    // DEPTH/OUTPUT independently) via a tiny per-architecture describe kernel.
    static uint2 layerShape(uint l)
    {
        return uint2((l == 0u) ? INPUT : HIDDEN, (l == DEPTH) ? OUTPUT : HIDDEN);
    }

    // Byte offset one past this block's last layer's bias, i.e. this block's
    // size in bytes when laid out starting at `layout[base].x`. Only needs
    // the last layer's own (compile-time-known) bias size: biases are plain
    // vectors, never `LAYOUT`-reordered/padded (only weight matrices are), so
    // `layout`'s own bias offsets already use their real, final size as-is.
    static uint end(no_diff StructuredBuffer<uint2> layout, no_diff uint base)
    {
        return align(layout[base + DEPTH].y + sizeof(half) * OUTPUT, 4);
    }

    static uint paramCount(no_diff StructuredBuffer<uint2> layout, no_diff uint base)
    {
        return end(layout, base) / uint(sizeof(half));
    }

    [Differentiable]
    static CoopVec<half, OUTPUT> forward<A : IActivation = LeakyReLU>(
        MatrixBiasBuffer weights, no_diff StructuredBuffer<uint2> layout, no_diff uint base,
        CoopVec<half, INPUT> input, A act = A())
    {
        var h = LinearOp<half, HIDDEN>(input, weights, layout[base], LAYOUT, COMPONENT_TYPE);
        h = act.eval(h);
        [ForceUnroll]
        for (uint l = 1u; l < DEPTH; ++l)
        {
            h = LinearOp<half, HIDDEN>(h, weights, layout[base + l], LAYOUT, COMPONENT_TYPE);
            h = act.eval(h);
        }
        return LinearOp<half, OUTPUT>(h, weights, layout[base + DEPTH], LAYOUT, COMPONENT_TYPE);
    }

    // SOTA-initialised value for the half-element `localElem` within this block
    // (localElem indexes [layout[base].x, end(layout,base)) in half slots).
    // Weights are scaled by their layer's fan-in/fan-out; biases and
    // alignment padding are zero. `zeroLastLayer` zeros the final layer so
    // the block starts as the identity/zero map (used by the flow couplings).
    static float initValue(
        no_diff StructuredBuffer<uint2> layout, no_diff uint base,
        uint localElem, MLPInit scheme, bool zeroLastLayer)
    {
        let byteOff = layout[base].x + localElem * uint(sizeof(half));

        [ForceUnroll]
        for (uint l = 0u; l < DEPTH + 1u; ++l)
        {
            let shape = layerShape(l);
            let inSize = shape.x;
            let outSize = shape.y;
            let wStart = layout[base + l].x;
            let bStart = layout[base + l].y;

            if (byteOff >= bStart && byteOff < bStart + uint(sizeof(half)) * outSize)
                return 0.0f; // bias
            if (byteOff < wStart || byteOff >= bStart)
                continue;    // not this layer's weights → keep scanning. Upper bound is
                             // bStart (this layer's own host-computed bias offset), not
                             // wStart + inSize*outSize*sizeof(half): LAYOUT's real
                             // per-layer weight region can be larger than the naive
                             // element count (see this struct's own top comment).

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
    static void initParams<T : IFloat>(
        RWByteAddressBuffer params, no_diff StructuredBuffer<uint2> layout, no_diff uint base,
        uint tid, uint threadCount,
        MLPInit scheme = MLPInit::HeNormal, bool zeroLastLayer = false)
    {
        let lo = layout[base].x / uint(sizeof(half));
        let hi = paramCount(layout, base);
        for (uint e = lo + tid; e < hi; e += threadCount)
            params.StoreAligned<T>(e * uint(sizeof(T)), T(initValue(layout, base, e - lo, scheme, zeroLastLayer)));
    }

    // Mixed-precision variant: writes a compute mirror (TMirror, e.g. half) and a
    // full-precision master (TMaster, e.g. float) from the same draw — Adam updates
    // the master, the mirror feeds the forward/backward compute.
    static void initParams<TMirror : IFloat, TMaster : IFloat>(
        RWByteAddressBuffer params, RWByteAddressBuffer master,
        no_diff StructuredBuffer<uint2> layout, no_diff uint base,
        uint tid, uint threadCount,
        MLPInit scheme = MLPInit::HeNormal, bool zeroLastLayer = false)
    {
        let lo = layout[base].x / uint(sizeof(half));
        let hi = paramCount(layout, base);
        for (uint e = lo + tid; e < hi; e += threadCount)
        {
            let v = initValue(layout, base, e - lo, scheme, zeroLastLayer);
            params.StoreAligned<TMirror>(e * uint(sizeof(TMirror)), TMirror(v));
            master.StoreAligned<TMaster>(e * uint(sizeof(TMaster)), TMaster(v));
        }
    }
}
