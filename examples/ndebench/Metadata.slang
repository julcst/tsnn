import architectures.IArchitecture;
import architectures.TMM;
import architectures.HGGrid;
import architectures.DFN;
import architectures.DFL;
import architectures.NSFLinear;
import architectures.NSFQuadratic;
import architectures.NSFRQS;
import architectures.HDF;

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
NDEBENCH_METADATA_KERNEL(DFN)
NDEBENCH_METADATA_KERNEL(DFL)
NDEBENCH_METADATA_KERNEL(NSFLinear)
NDEBENCH_METADATA_KERNEL(NSFQuadratic)
NDEBENCH_METADATA_KERNEL(NSFRQS)
NDEBENCH_METADATA_KERNEL(HDF)
