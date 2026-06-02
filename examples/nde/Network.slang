// examples/nde/Network.slang
// Normalizing-flow network for 2-D neural density estimation.
//
// Architecture: 4 alternating coupling layers, each with a small
// 1 → HIDDEN → HIDDEN → (3·K−1) conditioner MLP that produces RQS
// parameters for the transformed dimension.
//
//   Layer 0 – condition on x, transform y  (cond_dim = 0)
//   Layer 1 – condition on y, transform x  (cond_dim = 1)
//   Layer 2 – condition on x, transform y  (cond_dim = 0)
//   Layer 3 – condition on y, transform x  (cond_dim = 1)
//
// All four conditioner MLPs are packed sequentially into a single
// MatrixBiasBuffer (weights only; one shared buffer for all layers).
// The offset of each MLP is the `end` field of the previous one —
// the standard TSNN chaining pattern.
//
// forward()  – [Differentiable], returns log p(x) for training
// sampleFlow() – non-differentiable inverse flow for generating samples

import TSNN.LinearOps;
import TSNN.CoopVec;
import TSNN.Utils.MLP;
import TSNN.Activations;
import TSNN.Flows.RQS;
import TSNN.Flows.CouplingLayer;
import TSNN.Utils.Math;

// ─── Architecture constants ───────────────────────────────────────────────────

static constexpr const uint kNumBins       = 8u;   // RQS bins per spline
static constexpr const uint kHidden        = 32u;  // conditioner MLP hidden width
static constexpr const uint kCondDepth     = 2u;   // hidden layers in conditioner
static constexpr const uint kNumFlowLayers = 4u;   // coupling layers
static constexpr const float kTailBound    = 3.0f; // linear-tail domain bound

// Conditioner MLP output size: K widths + K heights + (K-1) interior derivatives
static constexpr const uint kParamsPerSpline = 3u * kNumBins - 1u; // 23

// ─── Four conditioner MLPs, chained by byte offset ───────────────────────────
//
// Each kMLPi is a compile-time constexpr holding the byte offsets for every
// weight/bias tensor of that MLP.  kMLPi.end is the first byte NOT belonging
// to MLP i, so it doubles as the start offset of MLP i+1.

static constexpr const MLP<1u, kHidden, kCondDepth, kParamsPerSpline> kMLP0 = MLP<1u, kHidden, kCondDepth, kParamsPerSpline>(0u);
static constexpr const MLP<1u, kHidden, kCondDepth, kParamsPerSpline> kMLP1 = MLP<1u, kHidden, kCondDepth, kParamsPerSpline>(kMLP0.end);
static constexpr const MLP<1u, kHidden, kCondDepth, kParamsPerSpline> kMLP2 = MLP<1u, kHidden, kCondDepth, kParamsPerSpline>(kMLP1.end);
static constexpr const MLP<1u, kHidden, kCondDepth, kParamsPerSpline> kMLP3 = MLP<1u, kHidden, kCondDepth, kParamsPerSpline>(kMLP2.end);

// Total parameter count (float16 elements) and byte size of the whole buffer
static constexpr const uint kParamByteCount    = kMLP3.end;
static constexpr const uint kParamElementCount = kParamByteCount / sizeof(half);

// ─── Initialisation helpers (used by Optimize.cs.slang resetMain) ────────────
//
// The output layer of each conditioner is zero-initialised so the initial flow
// is close to identity.  The derivative biases alone are set to kRQSBoundaryUnnorm
// (≈ 0.5413) so that softplus(·) + kRQSMinDerivative ≈ 1.0 (identity slope).

// Byte range of the output-layer weight matrix within a single MLP block
static constexpr const uint kOutputWeightStart = kMLP0.offsets[kCondDepth].x;
static constexpr const uint kOutputWeightEnd   = kMLP0.offsets[kCondDepth].y;

// Byte range of the derivative portion of the output-layer bias vector
// (Widths occupy bytes [0, 2·K), heights [2·K, 4·K), derivs [4·K, 4·K + 2·(K-1)))
static constexpr const uint kOutputBiasStart   = kMLP0.offsets[kCondDepth].y;
static constexpr const uint kDerivBiasStart    = kOutputBiasStart + sizeof(half) * 2u * kNumBins;
static constexpr const uint kDerivBiasEnd      = kOutputBiasStart + sizeof(half) * (3u * kNumBins - 1u);
static constexpr const uint kMLPBlockBytes     = kMLP0.end; // bytes per MLP block

// ─── Helper: apply one coupling step from CoopVec MLP params ─────────────────
//
// Unpacks the raw CoopVec MLP output into (w, h, d) arrays and applies the RQS
// forward transform.  Keeping this in a single [Differentiable] function lets
// Slang thread the CoopVec gradient back through VectorFromCoopVec cleanly,
// without needing out-array parameters or a custom differential struct.
//
// Returns float2(y, log|dy/dz_mod|).

[Differentiable]
float2 applyCouplingRQS(
    no_diff float                        z_mod,
    CoopVec<half, kParamsPerSpline>      cv_params
)
{
    // CoopVecToArray (not VectorFromCoopVec) since N=23 exceeds the HLSL
    // vector<float,N> limit of 4.
    let arr = CoopVecToArray(cv_params);

    float w[kNumBins];
    float h[kNumBins];
    float d[kNumBins - 1u];

    [ForceUnroll] for (uint i = 0u; i < kNumBins;        i++) w[i] = arr[i];
    [ForceUnroll] for (uint i = 0u; i < kNumBins;        i++) h[i] = arr[kNumBins + i];
    [ForceUnroll] for (uint i = 0u; i < kNumBins - 1u;   i++) d[i] = arr[2u * kNumBins + i];

    return rqsForward<kNumBins>(z_mod, w, h, d, kTailBound);
}

// ─── Differentiable forward pass (training) ───────────────────────────────────
//
// Returns log p_θ(x) = log p_z(f(x)) + Σ log|∂f_i/∂x_i|
// where p_z is the standard bivariate Gaussian prior.
// x is treated as no_diff: gradients flow only through the MLP weights.

[Differentiable]
float forward(MatrixBiasBuffer weights, no_diff float2 x)
{
    float z0 = x.x;
    float z1 = x.y;
    float logDet = 0.0f;

    // ── Layer 0: condition on z0, transform z1 ────────────────────────────────
    {
        let cv_in = CoopVecFromVector<half>(vector<float, 1>(z0));
        let r = applyCouplingRQS(z1, kMLP0.forward<LeakyReLU>(weights, cv_in));
        z1 = r.x;   logDet += r.y;
    }

    // ── Layer 1: condition on z1, transform z0 ────────────────────────────────
    {
        let cv_in = CoopVecFromVector<half>(vector<float, 1>(z1));
        let r = applyCouplingRQS(z0, kMLP1.forward<LeakyReLU>(weights, cv_in));
        z0 = r.x;   logDet += r.y;
    }

    // ── Layer 2: condition on z0, transform z1 ────────────────────────────────
    {
        let cv_in = CoopVecFromVector<half>(vector<float, 1>(z0));
        let r = applyCouplingRQS(z1, kMLP2.forward<LeakyReLU>(weights, cv_in));
        z1 = r.x;   logDet += r.y;
    }

    // ── Layer 3: condition on z1, transform z0 ────────────────────────────────
    {
        let cv_in = CoopVecFromVector<half>(vector<float, 1>(z1));
        let r = applyCouplingRQS(z0, kMLP3.forward<LeakyReLU>(weights, cv_in));
        z0 = r.x;   logDet += r.y;
    }

    // ── Gaussian prior: log p_z(z0, z1) ──────────────────────────────────────
    float logPrior = -0.5f * (z0 * z0 + z1 * z1) - log(2.0f * float.getPi());
    return logPrior + logDet;
}

// Convenience overload: accepts raw ByteAddressBuffer (no autodiff needed)
float forward(ByteAddressBuffer weights, no_diff float2 x)
{
    return forward(MatrixBiasBuffer(weights), x);
}

// ─── Non-differentiable inverse flow (sampling) ───────────────────────────────
//
// Shared unpack helper — not [Differentiable], used only for the inverse path.
float applyCouplingRQSInv(float y_mod, CoopVec<half, kParamsPerSpline> cv_params)
{
    let arr = CoopVecToArray(cv_params);
    float w[kNumBins], h[kNumBins], d[kNumBins - 1u];
    [ForceUnroll] for (uint i = 0u; i < kNumBins;        i++) w[i] = arr[i];
    [ForceUnroll] for (uint i = 0u; i < kNumBins;        i++) h[i] = arr[kNumBins + i];
    [ForceUnroll] for (uint i = 0u; i < kNumBins - 1u;   i++) d[i] = arr[2u * kNumBins + i];
    return rqsInverse<kNumBins>(y_mod, w, h, d, kTailBound).x;
}

// Runs the coupling layers in reverse (3 → 0) to map z ~ N(0,I) → x ~ p_θ.
float2 sampleFlow(ByteAddressBuffer weights, float2 z)
{
    let buf = MatrixBiasBuffer(weights);
    float x0 = z.x;
    float x1 = z.y;

    x0 = applyCouplingRQSInv(x0, kMLP3.forward<LeakyReLU>(buf, CoopVecFromVector<half>(vector<float, 1>(x1))));
    x1 = applyCouplingRQSInv(x1, kMLP2.forward<LeakyReLU>(buf, CoopVecFromVector<half>(vector<float, 1>(x0))));
    x0 = applyCouplingRQSInv(x0, kMLP1.forward<LeakyReLU>(buf, CoopVecFromVector<half>(vector<float, 1>(x1))));
    x1 = applyCouplingRQSInv(x1, kMLP0.forward<LeakyReLU>(buf, CoopVecFromVector<half>(vector<float, 1>(x0))));

    return float2(x0, x1);
}
