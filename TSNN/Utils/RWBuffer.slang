void StoreFloat(RWByteAddressBuffer buffer, uint elementIndex, float value)
{
    buffer.StoreAligned<float>(elementIndex * sizeof(float), value);
}

float LoadFloat(RWByteAddressBuffer buffer, uint elementIndex)
{
    return buffer.LoadAligned<float>(elementIndex * sizeof(float));
}

void StoreHalf(RWByteAddressBuffer buffer, uint elementIndex, float value)
{
    buffer.StoreAligned<half>(elementIndex * sizeof(half), half(value));
}

float LoadHalf(RWByteAddressBuffer buffer, uint elementIndex)
{
    return float(buffer.LoadAligned<half>(elementIndex * sizeof(half)));
}

// Vectorized helpers
float4 LoadFloat4(RWByteAddressBuffer buffer, uint elementIndex)
{
    return buffer.LoadAligned<float4>(elementIndex * sizeof(float));
}

void StoreFloat4(RWByteAddressBuffer buffer, uint elementIndex, float4 v)
{
    buffer.StoreAligned<float4>(elementIndex * sizeof(float), v);
}

float4 LoadHalf4(RWByteAddressBuffer buffer, uint elementIndex)
{
    return float4(buffer.LoadAligned<half4>(elementIndex * sizeof(half)));
}

void StoreHalf4(RWByteAddressBuffer buffer, uint elementIndex, float4 v)
{
    buffer.StoreAligned<half4>(elementIndex * sizeof(half), half4(v));
}