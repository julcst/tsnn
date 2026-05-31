import LinearOps;
import CoopVec;
import Math;
import Activations;

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

}
