public interface IVector<T> : IDifferentiable
    where T : IFloat
{
    /// The compile-time size of the vector.
    public static const int Size;

    /// Ensures the differential is also a vector with the same structure.
    // __constraint Differential : IVector<T>;

    public __subscript(int index) -> T
    {
        [Differentiable] get;
        [Differentiable] set;
    }
}

public extension<T, int N> Array<T, N> : IVector<T>
    where T : IFloat
{
    public static const int Size = N;

    __subscript(int index) -> T
    {
        [Differentiable]
        get { return this[index]; }
        [Differentiable]
        set { this[index] = newValue; }
    }
}

public struct Slice<int N, T, V : IVector<T>> : IVector<T>
    where T : IFloat
{
    public static const int Size = N;
    V base;
    int offset;
    int stride;

    __init(V base, int offset, int stride = 1)
    {
        this.base = base;
        this.offset = offset;
        this.stride = stride;
    }

    __subscript(int index) -> T
    {
        [Differentiable]
        get { return base[offset + index * stride]; }
        [Differentiable]
        set { base[offset + index * stride] = newValue; }
    }
}
