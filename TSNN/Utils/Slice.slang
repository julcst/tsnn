struct Slice<T, A : IArray<T>> : IArray<T>
{
    A base;
    int offset;
    int count;
    int stride;

    __init(A base, int offset, int count, int stride = 1)
    {
        this.base = base;
        this.offset = offset;
        this.count = count;
        this.stride = stride;
    }
    
    int getCount() { return count; }

    __subscript(int index) -> T { get { return base[offset + index * stride]; } }
}
