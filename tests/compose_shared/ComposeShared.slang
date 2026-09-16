// external/tsnn/tests/compose_shared/ComposeShared.slang — isolated repro for the
// ComposeCoopVecPlus_bwd accumulation bug (see scripts/nisbench's FINDING.md).
//
// Mirrors the SHAPE every one of nisbench's three arm modules uses (HGGrid.slang,
// BenchDf.slang, BenchFlow.slang): one differentiable CoopVec `ctx` consumed RAW by a first
// differentiable transform, and consumed a second time through `ComposeCoopVecPlus(ctx, tail)`
// feeding a second differentiable transform -- except here `ctx` is fetched ONCE and shared,
// which is exactly the pattern those three files' headers say this Slang/Vulkan backend has
// "repeatedly shown" to drop one contribution's gradient.
//
// Deliberately uses a PLAIN weighted-sum reduction (CoopVecToArray + array dot product) as the
// "consumer" instead of a real TSNN.Utils.MLP/TSNN.LinearOps MLP.forward call: importing
// TSNN.LinearOps/TSNN.Utils.MLP ALONGSIDE TSNN.Encodings.Compose in one translation unit was
// found, while building this file, to trigger a SEPARATE, more severe corruption of
// ComposeCoopVecPlus_bwd's result (wrong by many orders of magnitude, not just "one consumer's
// contribution missing") -- reproduced independently of everything below and NOT specific to
// the accumulate-vs-overwrite bug this file targets (renaming ComposeCoopVecPlus's `M` generic
// parameter changed the corruption's shape but did not clear it; scoped out of this repro to
// keep the two findings separable -- see FINDING.md's "ComposeCoopVecPlus_bwd" entry for both).
// A real arm's forward() always imports LinearOps/MLP alongside Compose, so that second bug is
// the reason Step 3 (sharing context across a real MLP-consuming call site) was NOT applied to
// any arm's training/gradient path in this change -- see HGGrid.slang/BenchDf.slang/
// BenchFlow.slang's headers for the current, still-double-fetching state.
//
// Three loss variants isolate the accumulate claim (see ComposeSharedGrad.cs.slang's gradMain):
//   sharedLoss      -- ctx used at BOTH call sites (the shape under test).
//   rawOnlyLoss     -- ctx used ONLY at the raw call site (single consumer; known-safe).
//   composeOnlyLoss -- ctx used ONLY via ComposeCoopVecPlus (single consumer; known-safe).
// rawOnlyLoss's and composeOnlyLoss's own bwd_diff results are each single-use, so they are
// unaffected by any cross-call-site accumulation bug and serve as the ground truth: their SUM
// must equal sharedLoss's bwd_diff result. Before the ComposeCoopVecPlus_bwd fix, it does not
// (composeOnlyLoss's contribution is dropped when combined with rawOnlyLoss's in one call);
// after the fix, it does.

import TSNN.CoopVec;
import TSNN.Encodings.Compose;

static constexpr const uint kCtxDim  = 8u;
static constexpr const uint kTailDim = 4u;

[Differentiable]
float rawOnlyLoss(float ctxArr[kCtxDim], no_diff float w[kCtxDim])
{
    let ctx = CoopVecFromArray<half, kCtxDim>(ctxArr);
    let arr = CoopVecToArray(ctx);
    float sum = 0.0f;
    [ForceUnroll] for (uint i = 0u; i < kCtxDim; ++i) sum += arr[i] * w[i];
    return sum;
}

[Differentiable]
float composeOnlyLoss(float ctxArr[kCtxDim], no_diff half tail[kTailDim], no_diff float w[kCtxDim + kTailDim])
{
    let ctx = CoopVecFromArray<half, kCtxDim>(ctxArr);
    let composed = ComposeCoopVecPlus<int(kCtxDim), int(kTailDim)>(ctx, tail);
    let arr = CoopVecToArray(composed);
    float sum = 0.0f;
    [ForceUnroll] for (uint i = 0u; i < kCtxDim + kTailDim; ++i) sum += arr[i] * w[i];
    return sum;
}

// The shape under test: ctx fetched once, consumed at BOTH call sites.
[Differentiable]
float sharedLoss(float ctxArr[kCtxDim], no_diff half tail[kTailDim],
    no_diff float w1[kCtxDim], no_diff float w2[kCtxDim + kTailDim])
{
    let ctx = CoopVecFromArray<half, kCtxDim>(ctxArr);
    let rawArr = CoopVecToArray(ctx);
    float a = 0.0f;
    [ForceUnroll] for (uint i = 0u; i < kCtxDim; ++i) a += rawArr[i] * w1[i];

    let composed = ComposeCoopVecPlus<int(kCtxDim), int(kTailDim)>(ctx, tail);
    let composedArr = CoopVecToArray(composed);
    float b = 0.0f;
    [ForceUnroll] for (uint i = 0u; i < kCtxDim + kTailDim; ++i) b += composedArr[i] * w2[i];

    return a + b;
}
