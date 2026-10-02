import architectures.IArchitecture;
#include "Configurations.slangh"

RWStructuredBuffer<uint> gMetadata;
StructuredBuffer<uint2> gLayout;

void metadata<T : IArchitecture>(uint3 tid : SV_DispatchThreadID)
{
    gMetadata[0] = T.getParamCount(gLayout);
}

#define NDEBENCH_METADATA_KERNEL(ARCHITECTURE) \
    [numthreads(1, 1, 1)] \
    void metadata_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        metadata<ARCHITECTURE>(tid); \
    }


#define NDEBENCH_MODEL(NAME) NDEBENCH_METADATA_KERNEL(NAME)
#define NDEBENCH_GRID(NAME, G, L, H, D, K) NDEBENCH_MODEL(NAME)
#include "Models.slangh"
#undef NDEBENCH_GRID
#undef NDEBENCH_MODEL
