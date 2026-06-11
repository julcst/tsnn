// examples/nde/Network.slang
// Normalizing-flow network for 2-D neural density estimation.
//
// 8 alternating coupling layers on the normalized domain [-kDomain, kDomain].
// Layers alternate which coordinate they condition on / transform:
//   even layers – condition on z0 (x), transform z1 (y)
//   odd  layers – condition on z1 (y), transform z0 (x)
//
// Three independent, compile-time-selectable pieces (this is library demo code;
// each piece is its own reusable struct in TSNN/Flows/):
//
//   USE_RQS          0 → affine coupling (s,t);  1 → rational-quadratic spline.
//                    RQS is more expressive (lower JSD) but only C1, so its
//                    density shows faint knot "creases" (raise kNumBins to
//                    thin them).  Affine is smoother but less expressive.
//
//   USE_ROTATIONS    Interleave Rotation2D mixing layers between coupling
//                    blocks (Glow-style).  Breaks axis-aligned structure so
//                    couplings can model correlations / creases spread out.
//
//   USE_UNIFORM_PRIOR 0 → StandardNormalPrior (unbounded; use WITH rotations);
//                    1 → UniformPrior on the box (use for bounded RQS WITHOUT
//                    rotations — a rotation would push latents out of support).
//
// Sensible combos:  affine + normal + rotations (default, smooth & expressive);
//                    RQS + uniform + no rotations (bounded, clean sampling).
//
// forward()    – [Differentiable], returns log p_θ(x) for NLL training
// sampleFlow() – inverse flow, maps a base sample z → x ~ p_θ
// samplePrior()– draw a base latent from the configured prior

// These can be overridden by the host via Slang compiler defines (main.py
// passes them through SlangCompilerOptions.defines); the values here are just
// the standalone fallbacks so the module still compiles on its own.
#ifndef USE_RQS
#define USE_RQS           0
#endif
#ifndef USE_ROTATIONS
#define USE_ROTATIONS     1
#endif
#ifndef USE_UNIFORM_PRIOR
#define USE_UNIFORM_PRIOR 0
#endif
// Conditioner-MLP activation (host overrides via defines):
//   0 ReLU  1 LeakyReLU  2 Swish/SiLU  3 GELU  4 ELU  5 Tanh  6 Mish
#ifndef ACTIVATION
#define ACTIVATION 1
#endif

import TSNN.LinearOps;
import TSNN.CoopVec;
import TSNN.Utils.MLP;
import TSNN.Activations;
import TSNN.Flows.Bijection;
import TSNN.Flows.AffineTransform;
import TSNN.Flows.RQS;
import TSNN.Flows.Rotation;
import TSNN.Flows.Prior;
import TSNN.Utils.Math;

// ─── Architecture constants ───────────────────────────────────────────────────

static constexpr const uint  kHidden    = 64u;
static constexpr const uint  kCondDepth = 2u;
// RQS bins.  [ForceUnroll] in RQS.slang unrolls every loop to kNumBins iterations,
// so this is a hard multiplier on per-thread shader complexity.  The GPU watchdog
// (NVRM XID 109 / CTX SWITCH TIMEOUT) fires when a kernel takes >~2 s.
// Safe ranges observed in practice:  8–16 bins for ≥4 layers;  32 bins for ≤2 layers.
// Mueller et al. use 32-bin piecewise-quadratic with 2 layers; 16-bin RQS is
// roughly equivalent (RQS is more expressive per bin).
static constexpr const uint  kNumBins   = 32u;
static constexpr const float kDomain    = 1.0f; // data/latent domain [-1, 1]
static const          float  kRotAngle  = 0.39269908f; // π/8 between coupling blocks

#if USE_RQS
static constexpr const uint kParamsPerLayer = 3u * kNumBins - 1u;
#else
static constexpr const uint kParamsPerLayer = 2u;
#endif

typealias CondMLP = MLP<1u, kHidden, kCondDepth, kParamsPerLayer>;

#if   ACTIVATION == 0
typealias Activation = ReLU;
#elif ACTIVATION == 2
typealias Activation = Swish;
#elif ACTIVATION == 3
typealias Activation = GELU;
#elif ACTIVATION == 4
typealias Activation = ELU;
#elif ACTIVATION == 5
typealias Activation = Tanh;
#elif ACTIVATION == 6
typealias Activation = Mish;
#else
typealias Activation = LeakyReLU;
#endif

// The configured base distribution, as a single shared constant (no per-call
// construction).  UniformPrior matches the bounded box; StandardNormalPrior is
// the unbounded default (use with rotations).
#if USE_UNIFORM_PRIOR
static const UniformPrior gPrior = UniformPrior(kDomain);
#else
static const StandardNormalPrior gPrior = StandardNormalPrior();
#endif

// ─── Four conditioner MLPs chained by byte offset ────────────────────────────
// Mueller et al. use 2 coupling layers with a deep U-net conditioner; we use 4
// with a simpler MLP conditioner to compensate, staying within GPU timeout bounds.

static constexpr const CondMLP kMLP0 = CondMLP(0u);
static constexpr const CondMLP kMLP1 = CondMLP(kMLP0.end);
static constexpr const CondMLP kMLP2 = CondMLP(kMLP1.end);
static constexpr const CondMLP kMLP3 = CondMLP(kMLP2.end);

static constexpr const uint kParamByteCount    = kMLP3.end;
static constexpr const uint kParamElementCount = kParamByteCount / sizeof(half);

// Identity init (He hidden layers + zeroed last layer per block) is handled by
// MLP::initParams, called once per kMLP0..kMLP3 block in Optimize.cs::resetMain.

// ─── Single coupling step (shared by all layers, both flow types) ─────────────

// Build the conditioner input CoopVec from a single scalar coordinate.
[Differentiable]
CoopVec<half, 1u> condInput(float c) { return CoopVecFromVector<half>(vector<float, 1>(c)); }

[Differentiable]
FlowStep coupleForward(MatrixBiasBuffer weights, CondMLP mlp, float transformed, float cond)
{
    let p = mlp.forward<Activation>(weights, condInput(cond));
#if USE_RQS
    return rqsForward<kNumBins>(transformed, p, kDomain);
#else
    return affineForward(transformed, p);
#endif
}

float coupleInverse(MatrixBiasBuffer weights, CondMLP mlp, float transformed, float cond)
{
    let p = mlp.forward<Activation>(weights, condInput(cond));
#if USE_RQS
    return rqsInverse<kNumBins>(transformed, p, kDomain).value;
#else
    return affineInverse(transformed, p).value;
#endif
}

// Like coupleInverse but returns the full FlowStep.  logAbsDet carries the
// INVERSE sign (−) per the Bijection.slang convention, so the caller must
// negate it to recover the forward log|det| at this point.
FlowStep coupleInverseStep(MatrixBiasBuffer weights, CondMLP mlp, float transformed, float cond)
{
    let p = mlp.forward<Activation>(weights, condInput(cond));
#if USE_RQS
    return rqsInverse<kNumBins>(transformed, p, kDomain);
#else
    return affineInverse(transformed, p);
#endif
}

// ─── Differentiable forward pass ─────────────────────────────────────────────

[Differentiable]
float forward(MatrixBiasBuffer weights, no_diff float2 x)
{
    float z0 = x.x;
    float z1 = x.y;
    float logDet = 0.0f;

    // Two coupling blocks (each updates both coordinates), with one optional
    // rotation re-mixing the axes between them.
    { let r = coupleForward(weights, kMLP0, z1, z0); z1 = r.value; logDet += r.logAbsDet; }
    { let r = coupleForward(weights, kMLP1, z0, z1); z0 = r.value; logDet += r.logAbsDet; }
#if USE_ROTATIONS
    { let zr = Rotation2D(kRotAngle).apply(float2(z0, z1)); z0 = zr.x; z1 = zr.y; }
#endif
    { let r = coupleForward(weights, kMLP2, z1, z0); z1 = r.value; logDet += r.logAbsDet; }
    { let r = coupleForward(weights, kMLP3, z0, z1); z0 = r.value; logDet += r.logAbsDet; }

    // base log-density (rotations have det 1 → no log-det term)
    return gPrior.logProb(float2(z0, z1)) + logDet;
}

float forward(ByteAddressBuffer weights, no_diff float2 x)
{
    return forward(MatrixBiasBuffer(weights), x);
}

// ─── Base sampling + non-differentiable inverse flow ──────────────────────────

float2 samplePrior(float2 u01) { return gPrior.sample(u01); }

float2 sampleFlow(ByteAddressBuffer weights, float2 z)
{
    return sampleFlowLogProb(weights, z).x;
}

// ─── Joint sample + log-prob (used by the KL training kernel) ─────────────────

struct FlowSample { float2 x; float logProb; }

// Returns x ~ q_θ AND log q_θ(x) in a single inverse pass.
// Avoids the redundant forward() call that sampleFlow() + forward() would
// require: two flow evaluations total instead of three.
//
// logDet accumulates the FORWARD log|det J_f(x)| by negating each inverse
// step's logAbsDet (signed − for inverse per Bijection.slang).
FlowSample sampleFlowLogProb(ByteAddressBuffer weights, float2 z)
{
    let buf = MatrixBiasBuffer(weights);
    float x0 = z.x;
    float x1 = z.y;
    float logDet = 0.0f;

    { let r = coupleInverseStep(buf, kMLP3, x0, x1); x0 = r.value; logDet -= r.logAbsDet; }
    { let r = coupleInverseStep(buf, kMLP2, x1, x0); x1 = r.value; logDet -= r.logAbsDet; }
#if USE_ROTATIONS
    { let zr = Rotation2D(kRotAngle).applyInverse(float2(x0, x1)); x0 = zr.x; x1 = zr.y; }
#endif
    { let r = coupleInverseStep(buf, kMLP1, x0, x1); x0 = r.value; logDet -= r.logAbsDet; }
    { let r = coupleInverseStep(buf, kMLP0, x1, x0); x1 = r.value; logDet -= r.logAbsDet; }

    // log q(x) = log p_0(z) + log|det J_f(x)|;  prior evaluated at the base latent z.
    return { float2(x0, x1), gPrior.logProb(z) + logDet };
}
