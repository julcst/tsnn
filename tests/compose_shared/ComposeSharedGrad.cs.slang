// external/tsnn/tests/compose_shared/ComposeSharedGrad.cs.slang — bwd_diff probes for
// ComposeShared.slang's three loss variants (see that file's header for what each isolates),
// plus a forward-only loss for central-difference checks driven from Python by perturbing
// gCtxIn between dispatches.

import TSNN.CoopVec;
import ComposeShared;

StructuredBuffer<float> gCtxIn;        // gCount * kCtxDim
StructuredBuffer<float> gTail;         // kTailDim -- one fixed tail shared by every row
StructuredBuffer<float> gW1;           // kCtxDim -- rawOnlyLoss's weights
StructuredBuffer<float> gW2;           // kCtxDim + kTailDim -- composeOnlyLoss's weights
RWStructuredBuffer<float> gLoss;       // gCount * 3 -- [shared, rawOnly, composeOnly]
RWStructuredBuffer<float> gCtxGrad;    // gCount * 3 * kCtxDim -- same order

cbuffer CB
{
    uint gCount;
}

[numthreads(64, 1, 1)]
void gradMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;
    uint row = tid.x;

    float ctxArr[kCtxDim], zeros[kCtxDim];
    [ForceUnroll] for (uint i = 0u; i < kCtxDim; ++i) { ctxArr[i] = gCtxIn[row * kCtxDim + i]; zeros[i] = 0.0f; }
    half tail[kTailDim];
    [ForceUnroll] for (uint i = 0u; i < kTailDim; ++i) tail[i] = half(gTail[i]);
    float w1[kCtxDim];
    [ForceUnroll] for (uint i = 0u; i < kCtxDim; ++i) w1[i] = gW1[i];
    float w2[kCtxDim + kTailDim];
    [ForceUnroll] for (uint i = 0u; i < kCtxDim + kTailDim; ++i) w2[i] = gW2[i];

    // shared (both consumers, one ctx) -- the shape under test
    var ctxPairShared = diffPair(ctxArr, zeros);
    gLoss[row * 3u + 0u] = sharedLoss(ctxArr, tail, w1, w2);
    bwd_diff(sharedLoss)(ctxPairShared, tail, w1, w2, 1.0f);

    // raw-only (single consumer -- ground-truth component A)
    var ctxPairRaw = diffPair(ctxArr, zeros);
    gLoss[row * 3u + 1u] = rawOnlyLoss(ctxArr, w1);
    bwd_diff(rawOnlyLoss)(ctxPairRaw, w1, 1.0f);

    // compose-only (single consumer -- ground-truth component B)
    var ctxPairCompose = diffPair(ctxArr, zeros);
    gLoss[row * 3u + 2u] = composeOnlyLoss(ctxArr, tail, w2);
    bwd_diff(composeOnlyLoss)(ctxPairCompose, tail, w2, 1.0f);

    [ForceUnroll]
    for (uint i = 0u; i < kCtxDim; ++i)
    {
        gCtxGrad[(row * 3u + 0u) * kCtxDim + i] = ctxPairShared.d[i];
        gCtxGrad[(row * 3u + 1u) * kCtxDim + i] = ctxPairRaw.d[i];
        gCtxGrad[(row * 3u + 2u) * kCtxDim + i] = ctxPairCompose.d[i];
    }
}

// Forward-only evaluation of the shared (both-consumers) loss, for central-difference checks.
[numthreads(64, 1, 1)]
void lossOnlyMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;
    uint row = tid.x;
    float ctxArr[kCtxDim];
    [ForceUnroll] for (uint i = 0u; i < kCtxDim; ++i) ctxArr[i] = gCtxIn[row * kCtxDim + i];
    half tail[kTailDim];
    [ForceUnroll] for (uint i = 0u; i < kTailDim; ++i) tail[i] = half(gTail[i]);
    float w1[kCtxDim];
    [ForceUnroll] for (uint i = 0u; i < kCtxDim; ++i) w1[i] = gW1[i];
    float w2[kCtxDim + kTailDim];
    [ForceUnroll] for (uint i = 0u; i < kCtxDim + kTailDim; ++i) w2[i] = gW2[i];
    gLoss[row] = sharedLoss(ctxArr, tail, w1, w2);
}
