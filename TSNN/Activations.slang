import CoopVec;

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