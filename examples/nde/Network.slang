// examples/nde/Network.slang
// Normalizing-flow network for 2-D neural density estimation.
//
// Architecture: 8 alternating affine coupling layers, each with a small
// 1 → HIDDEN → HIDDEN → 2 conditioner MLP that produces (s_raw, t).
// Layers alternate which coordinate they condition on / transform:
//   even layers – condition on z0 (x), transform z1 (y)
//   odd  layers – condition on z1 (y), transform z0 (x)
//
// 8 layers (4 per dimension) are needed to model a multimodal 2-D density;
// 4 layers produce visible axis-aligned smearing and cannot separate modes.
//
// All conditioner MLPs are packed sequentially in one MatrixBiasBuffer.
//
// forward()    – [Differentiable], returns log p_θ(x) for NLL training
// sampleFlow() – inverse flow, maps Gaussian sample z → x ~ p_θ

import TSNN.LinearOps;
import TSNN.CoopVec;
import TSNN.Utils.MLP;
import TSNN.Activations;
import TSNN.Flows.AffineTransform;
import TSNN.Utils.Math;

// ─── Architecture constants ───────────────────────────────────────────────────

static constexpr const uint  kHidden    = 64u;
static constexpr const uint  kCondDepth = 2u;

// Conditioner output: [s_raw, t]
static constexpr const uint kParamsPerLayer = 2u;

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

// Helper: build the conditioner input CoopVec from a single scalar coordinate.
[Differentiable]
CoopVec<half, 1u> condInput(float c) { return CoopVecFromVector<half>(vector<float, 1>(c)); }

// ─── Differentiable forward pass ─────────────────────────────────────────────

[Differentiable]
float forward(MatrixBiasBuffer weights, no_diff float2 x)
{
    float z0 = x.x;
    float z1 = x.y;
    float logDet = 0.0f;

    // even: cond z0 → transform z1;  odd: cond z1 → transform z0
    { let r = affineForward(z1, kMLP0.forward<LeakyReLU>(weights, condInput(z0))); z1 = r.value; logDet += r.logAbsDet; }
    { let r = affineForward(z0, kMLP1.forward<LeakyReLU>(weights, condInput(z1))); z0 = r.value; logDet += r.logAbsDet; }
    { let r = affineForward(z1, kMLP2.forward<LeakyReLU>(weights, condInput(z0))); z1 = r.value; logDet += r.logAbsDet; }
    { let r = affineForward(z0, kMLP3.forward<LeakyReLU>(weights, condInput(z1))); z0 = r.value; logDet += r.logAbsDet; }
    { let r = affineForward(z1, kMLP4.forward<LeakyReLU>(weights, condInput(z0))); z1 = r.value; logDet += r.logAbsDet; }
    { let r = affineForward(z0, kMLP5.forward<LeakyReLU>(weights, condInput(z1))); z0 = r.value; logDet += r.logAbsDet; }
    { let r = affineForward(z1, kMLP6.forward<LeakyReLU>(weights, condInput(z0))); z1 = r.value; logDet += r.logAbsDet; }
    { let r = affineForward(z0, kMLP7.forward<LeakyReLU>(weights, condInput(z1))); z0 = r.value; logDet += r.logAbsDet; }

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
    x0 = affineInverse(x0, kMLP7.forward<LeakyReLU>(buf, condInput(x1))).value;
    x1 = affineInverse(x1, kMLP6.forward<LeakyReLU>(buf, condInput(x0))).value;
    x0 = affineInverse(x0, kMLP5.forward<LeakyReLU>(buf, condInput(x1))).value;
    x1 = affineInverse(x1, kMLP4.forward<LeakyReLU>(buf, condInput(x0))).value;
    x0 = affineInverse(x0, kMLP3.forward<LeakyReLU>(buf, condInput(x1))).value;
    x1 = affineInverse(x1, kMLP2.forward<LeakyReLU>(buf, condInput(x0))).value;
    x0 = affineInverse(x0, kMLP1.forward<LeakyReLU>(buf, condInput(x1))).value;
    x1 = affineInverse(x1, kMLP0.forward<LeakyReLU>(buf, condInput(x0))).value;

    return float2(x0, x1);
}
