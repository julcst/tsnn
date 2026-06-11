import TSNN.CoopVec;

interface IActivation
{
    [Differentiable]
    CoopVec<T, N> eval<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input);
}

struct ReLU : IActivation
{
    [Differentiable]
    CoopVec<T, N> eval<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)
    {
        return relu(input);
    }
}

struct LeakyReLU : IActivation
{
    [Differentiable]
    CoopVec<T, N> eval<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)
    {
        return leakyReLU(input);
    }
}

// ─── Smooth SOTA activations (see CoopVec.slang for the math + backwards) ──────

struct Swish : IActivation
{
    [Differentiable]
    CoopVec<T, N> eval<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)
    {
        return swish(input);
    }
}

// SiLU is the same function as Swish (β = 1); alias for convenience.
typealias SiLU = Swish;

struct GELU : IActivation
{
    [Differentiable]
    CoopVec<T, N> eval<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)
    {
        return gelu(input);
    }
}

struct ELU : IActivation
{
    [Differentiable]
    CoopVec<T, N> eval<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)
    {
        return elu(input);
    }
}

struct Tanh : IActivation
{
    [Differentiable]
    CoopVec<T, N> eval<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)
    {
        return tanhAct(input);
    }
}

struct Mish : IActivation
{
    [Differentiable]
    CoopVec<T, N> eval<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)
    {
        return mish(input);
    }
}