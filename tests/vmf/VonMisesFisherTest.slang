// GPU probes for tests/vmf/test_vmf.py.
// Kept separate from the implementation so this file only exercises its public math.
import TSNN.Mixtures.VonMisesFisher;
import TSNN.Utils.PCG32;

cbuffer CB
{
    uint gCount;
}

StructuredBuffer<float> gInputs;
StructuredBuffer<float> gParams;
RWStructuredBuffer<float> gOutputs;

float3 loadVec3(StructuredBuffer<float> buf, uint base) { return float3(buf[base], buf[base + 1], buf[base + 2]); }
void storeVec3(RWStructuredBuffer<float> buf, uint base, float3 v) { buf[base] = v.x; buf[base + 1] = v.y; buf[base + 2] = v.z; }

// ─── Forward-only probes ───────────────────────────────────────────────────────

// row: [kappa] -> out: [logNormalizer]
[numthreads(64, 1, 1)]
void logNormalizerMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCount) return;
    gOutputs[i] = vmfLogNormalizer(gInputs[i]);
}

// row: [mean.xyz, kappa, dir.xyz] (stride 7) -> out: [logPdf]
[numthreads(64, 1, 1)]
void logPdfMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCount) return;
    uint base = i * 7u;
    let mean = loadVec3(gInputs, base);
    let kappa = gInputs[base + 3u];
    let dir = loadVec3(gInputs, base + 4u);
    gOutputs[i] = vmfLogPdf(mean, kappa, dir);
}

// row: [mean.xyz, kappa, dir.xyz, cosCap] (stride 8) -> out: [logPdf]
[numthreads(64, 1, 1)]
void logPdfCapMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCount) return;
    uint base = i * 8u;
    let mean = loadVec3(gInputs, base);
    let kappa = gInputs[base + 3u];
    let dir = loadVec3(gInputs, base + 4u);
    let cosCap = gInputs[base + 7u];
    gOutputs[i] = vmfLogPdfCap(mean, kappa, dir, cosCap);
}

// row: [mean.xyz, kappa, axis.xyz] (stride 7) -> out: [logMass]
[numthreads(64, 1, 1)]
void halfspaceLogMassMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCount) return;
    uint base = i * 7u;
    let mean = loadVec3(gInputs, base);
    let kappa = gInputs[base + 3u];
    let axis = loadVec3(gInputs, base + 4u);
    gOutputs[i] = vmfHalfspaceLogMass(mean, kappa, axis);
}

// row: [mean.xyz, kappa, xi.xy] (stride 6) -> out: [dir.xyz] (stride 3)
[numthreads(64, 1, 1)]
void sampleMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCount) return;
    uint base = i * 6u;
    let mean = loadVec3(gInputs, base);
    let kappa = gInputs[base + 3u];
    let xi = float2(gInputs[base + 4u], gInputs[base + 5u]);
    storeVec3(gOutputs, i * 3u, sampleVmf(mean, kappa, xi));
}

// row: [mean.xyz, kappa, cosCap, xi.xy] (stride 7) -> out: [dir.xyz] (stride 3)
[numthreads(64, 1, 1)]
void sampleCapMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCount) return;
    uint base = i * 7u;
    let mean = loadVec3(gInputs, base);
    let kappa = gInputs[base + 3u];
    let cosCap = gInputs[base + 4u];
    let xi = float2(gInputs[base + 5u], gInputs[base + 6u]);
    storeVec3(gOutputs, i * 3u, sampleVmfCap(mean, kappa, cosCap, xi));
}

// ─── Gradient probes: bwd_diff(mean, kappa), dir/cosCap held no_diff ─────────────

// row: [mean.xyz, kappa, dir.xyz] (stride 7) -> out: [value, dmean.xyz, dkappa] (stride 5)
[numthreads(64, 1, 1)]
void logPdfGradMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCount) return;
    uint base = i * 7u;
    let mean = loadVec3(gInputs, base);
    let kappa = gInputs[base + 3u];
    let dir = loadVec3(gInputs, base + 4u);

    var meanPair = diffPair(mean, float3(0.0f));
    var kappaPair = diffPair(kappa, 0.0f);
    let value = vmfLogPdf(mean, kappa, dir);
    bwd_diff(vmfLogPdf)(meanPair, kappaPair, dir, 1.0f);

    uint outBase = i * 5u;
    gOutputs[outBase + 0] = value;
    storeVec3(gOutputs, outBase + 1, meanPair.d);
    gOutputs[outBase + 4] = kappaPair.d;
}

// row: [mean.xyz, kappa, dir.xyz, cosCap] (stride 8) -> out: [value, dmean.xyz, dkappa] (stride 5)
[numthreads(64, 1, 1)]
void logPdfCapGradMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCount) return;
    uint base = i * 8u;
    let mean = loadVec3(gInputs, base);
    let kappa = gInputs[base + 3u];
    let dir = loadVec3(gInputs, base + 4u);
    let cosCap = gInputs[base + 7u];

    var meanPair = diffPair(mean, float3(0.0f));
    var kappaPair = diffPair(kappa, 0.0f);
    let value = vmfLogPdfCap(mean, kappa, dir, cosCap);
    bwd_diff(vmfLogPdfCap)(meanPair, kappaPair, dir, cosCap, 1.0f);

    uint outBase = i * 5u;
    gOutputs[outBase + 0] = value;
    storeVec3(gOutputs, outBase + 1, meanPair.d);
    gOutputs[outBase + 4] = kappaPair.d;
}

// ─── N=4 lobe mixture ─────────────────────────────────────────────────────────

static const uint kMixtureLobes = 4u;
typealias TestMixture = VonMisesFisherMixture<kMixtureLobes>;

// gParams row: kMixtureLobes*STRIDE floats; gInputs row: dir.xyz (stride 3) -> out: [logDensity]
[numthreads(64, 1, 1)]
void mixtureEvalMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCount) return;
    float params[kMixtureLobes * TestMixture::STRIDE];
    [ForceUnroll]
    for (uint j = 0; j < kMixtureLobes * TestMixture::STRIDE; ++j)
        params[j] = gParams[i * (kMixtureLobes * TestMixture::STRIDE) + j];
    let dir = loadVec3(gInputs, i * 3u);
    TestMixture mixture;
    gOutputs[i] = mixture.eval(params, dir);
}

// gParams row: kMixtureLobes*STRIDE floats (one row, index 0, shared by every thread)
// -> out: [dir.xyz] (stride 3), one PCG32-seeded draw per thread.
[numthreads(64, 1, 1)]
void mixtureSampleMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCount) return;
    float params[kMixtureLobes * TestMixture::STRIDE];
    [ForceUnroll]
    for (uint j = 0; j < kMixtureLobes * TestMixture::STRIDE; ++j)
        params[j] = gParams[j];
    TestMixture mixture;
    PCG32 rng = PCG32(uint64_t(i + 1), uint64_t(17));
    storeVec3(gOutputs, i * 3u, mixture.sample(params, rng));
}
