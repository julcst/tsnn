// Differentiable linear layer: weight/bias buffer wrappers and LinearOp
// forward + backward using the Vulkan cooperative vector extension.

import CoopVec;

// GPU buffer holding a flat weight matrix followed by a bias vector.
// The Differential counterpart (MatrixBiasBufferGrad) collects gradients
// via atomic scatter (outer product accumulate / reduce-sum accumulate).

struct MatrixBiasBufferGrad : IDifferentiablePtrType
{
    typealias Differential = MatrixBiasBufferGrad;

    __init(RWByteAddressBuffer buf) { buffer = buf; }

    RWByteAddressBuffer buffer;
}

struct MatrixBiasBuffer : IDifferentiablePtrType
{
    typealias Differential = MatrixBiasBufferGrad;

    __init(ByteAddressBuffer buf) { buffer = buf; }

    ByteAddressBuffer buffer;
}

// Forward linear layer: output = W * input + bias.
// offsets.x = byte offset of weight matrix in buf.
// offsets.y = byte offset of bias vector in buf.
CoopVec<T, M> LinearOp<T : __BuiltinFloatingPointType, let M : int, let K : int>(
    CoopVec<T, K> ip,
    MatrixBiasBuffer buf,
    uint2 offsets,
    constexpr CoopVecMatrixLayout layout,
    constexpr CoopVecComponentType ctype)
{
    return coopVecMatMulAdd<T, M>(
        ip,          ctype,
        buf.buffer,  offsets[0], ctype,
        buf.buffer,  offsets[1], ctype,
        layout, false, 0
    );
}

// Backward: accumulates dW and db, returns dInput.
CoopVec<T, K> LinearOpBackward<T : __BuiltinFloatingPointType, let M : int, let K : int>(
    CoopVec<T, K>       ip,
    CoopVec<T, M>       grad,
    ByteAddressBuffer   buf,
    RWByteAddressBuffer dBuf,
    uint2               offsets,
    constexpr CoopVecMatrixLayout layout,
    constexpr CoopVecComponentType ctype)
{
    // dW += grad ⊗ ip   (outer product accumulate)
    coopVecOuterProductAccumulate(grad, ip, dBuf, offsets[0], 0, layout, ctype);
    // db += grad         (reduce-sum accumulate)
    coopVecReduceSumAccumulate(grad, dBuf, offsets[1]);
    // dInput = W^T * grad
    return coopVecMatMul<T, K>(grad, ctype, buf, offsets[0], ctype, layout, true, 0);
}

[BackwardDerivativeOf(LinearOp)]
void LinearOp_bwd<T : __BuiltinFloatingPointType, let M : int, let K : int>(
    inout DifferentialPair<CoopVec<T, K>> ip,
    DifferentialPtrPair<MatrixBiasBuffer> buf,
    uint2 offsets,
    constexpr CoopVecMatrixLayout layout,
    constexpr CoopVecComponentType ctype,
    CoopVec<T, M>.Differential grad)
{
    ip = diffPair(ip.p,
        LinearOpBackward<T, M, K>(
            ip.p, grad,
            buf.p.buffer, buf.d.buffer,
            offsets, layout, ctype
        )
    );
}
