import architectures.IArchitecture;
import architectures.TMM;
import architectures.HGGrid;

RWStructuredBuffer<uint> gMetadata;

void metadata<T : IArchitecture>(uint3 tid : SV_DispatchThreadID)
{
    gMetadata[0] = T.getParamCount();
}

#define NDEBENCH_METADATA_KERNEL(ARCHITECTURE) \
    [numthreads(1, 1, 1)] \
    void metadata_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        metadata<ARCHITECTURE>(tid); \
    }

NDEBENCH_METADATA_KERNEL(TMM)
NDEBENCH_METADATA_KERNEL(HGGrid)
