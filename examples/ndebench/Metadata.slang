import architectures.TMM;

RWStructuredBuffer<uint> gMetadata;

#define NDEBENCH_METADATA_KERNEL(ARCHITECTURE) \
    [numthreads(1, 1, 1)] \
    void metadata_ ## ARCHITECTURE(uint3 tid : SV_DispatchThreadID) \
    { \
        gMetadata[0] = ARCHITECTURE.getParamCount(); \
    }

NDEBENCH_METADATA_KERNEL(TMM)
