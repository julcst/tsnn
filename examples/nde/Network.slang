// examples/nde/Network.slang
// Normalizing-flow network for 2-D neural density estimation.
//
// 8 alternating coupling layers on a fixed normalized domain [-kDomain, kDomain].
// The caller is expected to scale its data into this domain; there is no
// per-dataset tail-bound to thread through the code.
//
// Layers alternate which coordinate they condition on / transform:
//   even layers – condition on z0 (x), transform z1 (y)
//   odd  layers – condition on z1 (y), transform z0 (x)
//
// The coupling transform is selectable at compile time via USE_RQS:
//   USE_RQS 0 → affine  (s,t),                conditioner output dim 2
//   USE_RQS 1 → rational-quadratic spline,    conditioner output dim 3K-1
// Flip USE_RQS here AND set USE_RQS in main.py to match (the parameter count
// differs).  RQS is more expressive (lower JSD) but, being a piecewise spline,
// is only C1: its second derivative jumps at knots, so the density shows faint
// grid "creases" at the kNumBins knot lines (amplified by exp() in the density
// view).  This is inherent to rational-quadratic splines, not a bug; raising
// kNumBins makes the creases finer.  Affine is smoother but less expressive.
//
// forward()    – [Differentiable], returns log p_θ(x) for NLL training
// sampleFlow() – inverse flow, maps Gaussian sample z → x ~ p_θ

#define USE_RQS 1

import TSNN.LinearOps;
import TSNN.CoopVec;
import TSNN.Utils.MLP;
import TSNN.Activations;
import TSNN.Flows.AffineTransform;
import TSNN.Flows.RQS;
import TSNN.Utils.Math;

// ─── Architecture constants ───────────────────────────────────────────────────

static constexpr const uint  kHidden    = 64u;
static constexpr const uint  kCondDepth = 2u;
static constexpr const uint  kNumBins   = 16u;  // RQS bins (USE_RQS only)
static constexpr const float kDomain    = 1.0f; // data/latent domain [-1, 1]

#if USE_RQS
static constexpr const uint kParamsPerLayer = 3u * kNumBins - 1u;
#else
static constexpr const uint kParamsPerLayer = 2u;
#endif

typealias CondMLP = MLP<1u, kHidden, kCondDepth, kParamsPerLayer>;

// ─── Eight conditioner MLPs chained by byte offset ───────────────────────────

static constexpr const CondMLP kMLP0 = CondMLP(0u);
static constexpr const CondMLP kMLP1 = CondMLP(kMLP0.end);
static constexpr const CondMLP kMLP2 = CondMLP(kMLP1.end);
static constexpr const CondMLP kMLP3 = CondMLP(kMLP2.end);
static constexpr const CondMLP kMLP4 = CondMLP(kMLP3.end);
static constexpr const CondMLP kMLP5 = CondMLP(kMLP4.end);
static constexpr const CondMLP kMLP6 = CondMLP(kMLP5.end);
static constexpr const CondMLP kMLP7 = CondMLP(kMLP6.end);

static constexpr const uint kParamByteCount    = kMLP7.end;
static constexpr const uint kParamElementCount = kParamByteCount / sizeof(half);

// Output layer starts here; everything from this offset to kMLPBlockBytes is
// zeroed at init so each layer begins as the identity transform.
static constexpr const uint kOutputWeightStart = kMLP0.offsets[kCondDepth].x;
static constexpr const uint kMLPBlockBytes     = kMLP0.end;

// ─── Single coupling step (shared by all layers, both flow types) ─────────────

struct LayerOut : IDifferentiable {
    float value;
    float logDet;
};

// Build the conditioner input CoopVec from a single scalar coordinate.
[Differentiable]
CoopVec<half, 1u> condInput(float c) { return CoopVecFromVector<half>(vector<float, 1>(c)); }

[Differentiable]
LayerOut coupleForward(MatrixBiasBuffer weights, CondMLP mlp, float transformed, float cond)
{
    let p = mlp.forward<LeakyReLU>(weights, condInput(cond));
#if USE_RQS
    let r = rqsForward<kNumBins>(transformed, p, kDomain);
#else
    let r = affineForward(transformed, p);
#endif
    LayerOut o = { r.value, r.logAbsDet };
    return o;
}

float coupleInverse(MatrixBiasBuffer weights, CondMLP mlp, float transformed, float cond)
{
    let p = mlp.forward<LeakyReLU>(weights, condInput(cond));
#if USE_RQS
    return rqsInverse<kNumBins>(transformed, p, kDomain).value;
#else
    return affineInverse(transformed, p).value;
#endif
}

// ─── Differentiable forward pass ─────────────────────────────────────────────

[Differentiable]
float forward(MatrixBiasBuffer weights, no_diff float2 x)
{
    float z0 = x.x;
    float z1 = x.y;
    float logDet = 0.0f;

    // even: cond z0 → transform z1;  odd: cond z1 → transform z0
    { let r = coupleForward(weights, kMLP0, z1, z0); z1 = r.value; logDet += r.logDet; }
    { let r = coupleForward(weights, kMLP1, z0, z1); z0 = r.value; logDet += r.logDet; }
    { let r = coupleForward(weights, kMLP2, z1, z0); z1 = r.value; logDet += r.logDet; }
    { let r = coupleForward(weights, kMLP3, z0, z1); z0 = r.value; logDet += r.logDet; }
    { let r = coupleForward(weights, kMLP4, z1, z0); z1 = r.value; logDet += r.logDet; }
    { let r = coupleForward(weights, kMLP5, z0, z1); z0 = r.value; logDet += r.logDet; }
    { let r = coupleForward(weights, kMLP6, z1, z0); z1 = r.value; logDet += r.logDet; }
    { let r = coupleForward(weights, kMLP7, z0, z1); z0 = r.value; logDet += r.logDet; }

    // Bivariate standard-Gaussian prior
    float logPrior = -0.5f * (z0 * z0 + z1 * z1) - log(2.0f * float.getPi());
    return logPrior + logDet;
}

float forward(ByteAddressBuffer weights, no_diff float2 x)
{
    return forward(MatrixBiasBuffer(weights), x);
}

// ─── Non-differentiable inverse flow (sampling) ───────────────────────────────

float2 sampleFlow(ByteAddressBuffer weights, float2 z)
{
    let buf = MatrixBiasBuffer(weights);
    float x0 = z.x;
    float x1 = z.y;

    // Invert in reverse layer order (7 → 0)
    x0 = coupleInverse(buf, kMLP7, x0, x1);
    x1 = coupleInverse(buf, kMLP6, x1, x0);
    x0 = coupleInverse(buf, kMLP5, x0, x1);
    x1 = coupleInverse(buf, kMLP4, x1, x0);
    x0 = coupleInverse(buf, kMLP3, x0, x1);
    x1 = coupleInverse(buf, kMLP2, x1, x0);
    x0 = coupleInverse(buf, kMLP1, x0, x1);
    x1 = coupleInverse(buf, kMLP0, x1, x0);

    return float2(x0, x1);
}
