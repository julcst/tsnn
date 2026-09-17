import TSNN.Utils.PCG32;

// Architecture-independent, like `optimize` in Optimize.slang -- no macro,
// no REGISTRY entry.

StructuredBuffer<float> gMarginalCDF;    // gHeight entries, ends at 1.0
StructuredBuffer<float> gConditionalCDF; // gHeight * gWidth, row-major, each row ends at 1.0
RWStructuredBuffer<float2> gSamples;

// All scalars go through a named cbuffer, never bare `uniform` entry-point
// parameters -- see Optimize.slang's `InitCB` comment for the GPU-hang writeup.
cbuffer ImageSampleCB
{
    uint gWidth;
    uint gHeight;
    uint gCount;
    uint gStreamOffset;
    uint gSeed;
}

// First index with cdf[i] > u, i.e. np.searchsorted(..., side="right"),
// matching the CPU sampler this replaces.  Clamped to count - 1.
uint upperBound(StructuredBuffer<float> cdf, uint begin, uint count, float u)
{
    uint lo = 0;
    uint hi = count;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2;
        if (cdf[begin + mid] > u)
            hi = mid;
        else
            lo = mid + 1;
    }
    return min(lo, count - 1);
}

[numthreads(256, 1, 1)]
void sampleImage(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;

    var rand = PCG32(gSeed, gStreamOffset + tid.x);
    uint row = upperBound(gMarginalCDF, 0, gHeight, rand.nextFloat());
    uint col = upperBound(gConditionalCDF, row * gWidth, gWidth, rand.nextFloat());
    float jx = rand.nextFloat();
    float jy = rand.nextFloat();

    // One ulp below 1.0f -- preserves the np.nextafter clamp in benchmark.py's
    // former sample_stream() and mirrors the cell clamp in
    // examples/hggrid/Network.slang's sampleModel.
    let kMaxCoord = 0.99999994f;
    gSamples[tid.x] = float2(
        min((col + jx) / float(gWidth), kMaxCoord),
        min((row + jy) / float(gHeight), kMaxCoord));
}
