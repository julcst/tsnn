// examples/hggrid/Network.slang
// HGGrid: a hierarchical grid of truncated Gaussians for 2-D density estimation.
//
// TSNN/Mixtures/TruncatedGMM.slang fits a target well but every loop in its
// eval()/sample() is [ForceUnroll]ed over the component count N, so both cost
// and reverse-mode shader size grow linearly with N -- the AD path explodes
// well before enough components exist to resolve fine detail.
//
// HGGrid trades "many global Gaussians" for "few Gaussians *per cell*":
//   - a learned GRID_DIM x GRID_DIM categorical (Histogram) picks a cell
//   - a one-hot encoding of that cell drives a small MLP that predicts
//     NUM_GAUSSIANS truncated Gaussians over the cell's *local* coordinates
// Expressiveness now scales with the number of cells (cheap -- one softmax)
// instead of with the unrolled mixture size; the unrolled GMM loop stays at
// NUM_GAUSSIANS.
//
// Domain is the unit square [0,1]^2 (TruncatedGMM's native truncation).
//
// forward()     -- [Differentiable], returns log p_theta(x) for MLE/KL training
// sampleModel() -- ancestral sample: bin ~ Histogram, then Gaussian ~ GMM,
//                  returning (x, log p_theta(x)) in a single pass (needed by
//                  the KL training kernel; see Train.cs.slang)
//
// These can be overridden by the host via Slang compiler defines (main.py
// passes them through SlangCompilerOptions.defines); the values here are just
// the standalone fallbacks so the module still compiles on its own.
#ifndef GRID_DIM
#define GRID_DIM 8
#endif
#ifndef NUM_GAUSSIANS
#define NUM_GAUSSIANS 4
#endif
// Named HG_HIDDEN/HG_DEPTH (not HIDDEN/DEPTH) because those bare names are
// TSNN/Utils/MLP.slang's own generic parameter names (`MLP<INPUT, HIDDEN,
// DEPTH, OUTPUT>`) -- Slang's preprocessor is translation-unit-wide, so a
// `#define HIDDEN 64` here would rewrite MLP.slang's generic parameter list
// itself the moment it's imported, producing an unexpected-token parse error.
#ifndef HG_HIDDEN
#define HG_HIDDEN 64
#endif
#ifndef HG_DEPTH
#define HG_DEPTH 3
#endif
// Fine-MLP activation (host overrides via defines):
//   0 ReLU  1 LeakyReLU  2 Swish/SiLU  3 GELU  4 ELU  5 Tanh  6 Mish
#ifndef ACTIVATION
#define ACTIVATION 1
#endif
// Consumed by Optimize.cs.slang (sanitizer instrumentation), not by this file
// -- documented here alongside the other overridable defines.
#ifndef DEBUG_COUNTERS
#define DEBUG_COUNTERS 0
#endif

import TSNN.CoopVec;
import TSNN.LinearOps;
import TSNN.Utils.MLP;
import TSNN.Activations;
import TSNN.Utils.PCG32;
import TSNN.Utils.Math;
import TSNN.Mixtures.Histogram;
import TSNN.Mixtures.TruncatedGMM;
import TSNN.Encodings.OneHot;
import TSNN.Encodings.FeatureBuffer; // reuse atomicAddF32 for the histogram's scatter-add backward

// ─── Architecture constants ───────────────────────────────────────────────────

static constexpr const uint kGridDim      = GRID_DIM;
static constexpr const uint kNumCells     = kGridDim * kGridDim;
static constexpr const uint kNumGaussians = NUM_GAUSSIANS;
static constexpr const uint kHidden       = HG_HIDDEN;
static constexpr const uint kDepth        = HG_DEPTH;
static constexpr const uint kEncodedDim   = 2u * kGridDim; // one-hot(cell.x) ++ one-hot(cell.y)
// TruncatedGMM<N>::STRIDE = 1 (weight) + 2*D (mean) + ... = 1 + 2*2 for D=2.
static constexpr const uint kFineParamCount = kNumGaussians * 5u;

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

typealias FineMLP = MLP<kEncodedDim, kHidden, kDepth, kFineParamCount>;

static constexpr const FineMLP kFineMLP = FineMLP(0u);

static constexpr const uint kParamByteCount    = kFineMLP.end;
static constexpr const uint kParamElementCount = kParamByteCount / sizeof(half);

// ─── Histogram logit buffer (differentiable pointer pair) ─────────────────────
//
// The coarse categorical has only kNumCells (64) scalars, stored as plain
// float32 -- unlike the fine MLP's weights it is never touched by CoopVec
// matmul hardware, so there is nothing for a float16 mirror to buy. Shaped
// exactly like loadVec/loadVec_bwd in TSNN/Encodings/FeatureBuffer.slang:
// forward reads the whole array, backward scatters gradients with a CAS
// float32 atomic add, scrubbing non-finite contributions before the atomic so
// one bad sample can't poison every cell (Inf + x = Inf).

struct HistLogitBufferGrad : IDifferentiablePtrType
{
    typealias Differential = HistLogitBufferGrad;
    __init(RWByteAddressBuffer buf) { buffer = buf; }
    RWByteAddressBuffer buffer;
}

struct HistLogitBuffer : IDifferentiablePtrType
{
    typealias Differential = HistLogitBufferGrad;
    __init(ByteAddressBuffer buf) { buffer = buf; }
    ByteAddressBuffer buffer;
}

[BackwardDerivative(loadHistLogits_bwd)]
float[kNumCells] loadHistLogits(HistLogitBuffer b)
{
    float arr[kNumCells];
    [ForceUnroll]
    for (uint i = 0; i < kNumCells; ++i)
        arr[i] = b.buffer.LoadAligned<float>(i * sizeof(float));
    return arr;
}

// Histogram.eval computes a log-softmax, whose backward is dense: every one of
// the kNumCells logits gets a nonzero gradient from *every* thread, not just
// the bin it landed in. Scattering that directly -- one atomicAddF32 CAS loop
// per (thread, cell) pair -- means a kNumCells=64, threadCount=16384 dispatch
// contends up to ~1M times on just 64 addresses: measured at ~40ms/dispatch
// and scaling worse than linearly with batch size (see benchmark.py), which
// over a training run is enough sustained GPU saturation to stall the desktop
// compositor on a single-GPU machine. WaveActiveSum reduces each wave's
// (32 lanes on NVIDIA) contribution to one cell into a single value before the
// one surviving lane (WaveIsFirstLane) touches global memory, cutting the
// number of contended atomics by the wave size. Non-finite lanes are zeroed
// *before* the reduction, not after, so one poisoned sample can't turn an
// entire wave's otherwise-good gradient into NaN.
void loadHistLogits_bwd(DifferentialPtrPair<HistLogitBuffer> b, float dOut[kNumCells])
{
    [ForceUnroll]
    for (uint i = 0; i < kNumCells; ++i)
    {
        let g = dOut[i];
        let safeG = isfinite(g) ? g : 0.0f;
        let waveSum = WaveActiveSum(safeG);
        if (WaveIsFirstLane() && waveSum != 0.0f)
            atomicAddF32(b.d.buffer, i * sizeof(float), waveSum);
    }
}

// ─── Cell encoding ──────────────────────────────────────────────────────────
//
// One-hot(cell.x) ++ one-hot(cell.y) -> kEncodedDim floats -> a single
// CoopVecFromArray call.  oneHot's backward is a no-op (see OneHot.slang), so
// this carries no gradient into `cell` -- only into the fine MLP's weights
// downstream, which is exactly what we want (the cell index is discrete).
[Differentiable]
CoopVec<half, kEncodedDim> encodeCell(no_diff uint2 cell)
{
    let ex = oneHot<kGridDim>(cell.x);
    let ey = oneHot<kGridDim>(cell.y);

    float enc[kEncodedDim];
    [ForceUnroll]
    for (uint i = 0; i < kGridDim; ++i)
    {
        enc[i]             = ex[i];
        enc[kGridDim + i]  = ey[i];
    }
    return CoopVecFromArray<half, kEncodedDim>(enc);
}

// TSNN/Mixtures/TruncatedGMM.slang's logTruncGaussianPDF now computes its
// truncation-mass term via logTruncMass, which stays finite (and keeps a live
// gradient) across the entire real line for mean and logSigma using a stable
// erfcx-based log-CDF (TSNN/Mixtures/GaussianHelpers.slang) -- the old erf-sum
// mass collapsed to exactly zero once an unconstrained MLP head pushed a
// component's standardized truncation interval into one tail, which is why a
// clamp used to live here. Clamping mean/logSigma into a narrow "safe" band
// was itself the problem being worked around: it silently zeroed the gradient
// throughout the saturated region instead of fixing the underlying collapse.
// logSigma keeps a wide float-sanity clamp only -- exp(+-40) stays comfortably
// inside float32's range -- so it never binds during normal training and
// therefore never masks a real gradient the way the old [-2.5, 1.0] band did.
[Differentiable]
float[N] clampFineParams<let N : int>(float p[N])
{
    float out[N];
    [ForceUnroll]
    for (uint i = 0; i < N; ++i)
    {
        uint field = i % 5u;
        float v = p[i];
        if (field == 3u || field == 4u)  // logSigma: float-sanity bound only
            v = clamp(v, -40.0f, 40.0f);
        out[i] = v;
    }
    return out;
}

// ─── Differentiable forward pass ─────────────────────────────────────────────
//
// log p(x) = logHist(bin) + logFine(t) + 2*log(GRID_DIM)
//
// The 2*log(GRID_DIM) term is the Jacobian of (bin, t) -> x: mapping the unit
// square into a GRID_DIM x GRID_DIM grid of unit cells scales area by
// 1/GRID_DIM^2, so without this term the density would not integrate to 1
// (and a KL importance weight f(x)/q(x) would be off by a constant factor).
[Differentiable]
float forward(HistLogitBuffer histBuf, MatrixBiasBuffer weights, no_diff float2 x)
{
    float2 xs = x * float(kGridDim);
    // Round-trip safety: clamp the cell index so a sample that lands exactly
    // on x==1 (or a hair over it from upstream FP error) still resolves to a
    // valid cell instead of reading past the histogram's last bin.
    uint2 cell = clamp(uint2(floor(xs)), uint2(0u), uint2(kGridDim - 1u));
    float2 t   = xs - float2(cell);

    uint bin = grid2bin(cell, kGridDim);

    Histogram<kNumCells> hist;
    let histLogits = loadHistLogits(histBuf);
    float logHist = hist.eval(histLogits, bin);

    let encoded = encodeCell(cell);
    let fineOut = kFineMLP.forward<Activation>(weights, encoded);
    let fineParams = clampFineParams(CoopVecToArray(fineOut));

    TruncatedGMM<kNumGaussians> gmm;
    float logFine = gmm.eval(fineParams, t);

    return logHist + logFine + 2.0f * log(float(kGridDim));
}

float forward(ByteAddressBuffer histLogits, ByteAddressBuffer weights, no_diff float2 x)
{
    return forward(HistLogitBuffer(histLogits), MatrixBiasBuffer(weights), x);
}

// ─── Non-differentiable ancestral sampling ────────────────────────────────────

struct HGSample { float2 x; float logProb; }

// Draws x ~ p_theta AND returns log p_theta(x) in a single pass (needed by the
// KL training kernel, mirroring sampleFlowLogProb in examples/nde/Network.slang):
// reuses the histogram logits, fine-MLP output and sampled `t` that sampling
// already computed instead of re-loading everything through a second forward().
HGSample sampleModel(ByteAddressBuffer histLogitBuf, ByteAddressBuffer weightBuf, inout PCG32 rng)
{
    float histLogits[kNumCells];
    [ForceUnroll]
    for (uint i = 0; i < kNumCells; ++i)
        histLogits[i] = histLogitBuf.LoadAligned<float>(i * sizeof(float));

    Histogram<kNumCells> hist;
    uint bin = hist.sample(histLogits, rng.nextFloat());
    float logHist = hist.eval(histLogits, bin);
    uint2 cell = bin2grid(bin, kGridDim);

    let encoded = encodeCell(cell);
    let fineOut = kFineMLP.forward<Activation>(MatrixBiasBuffer(weightBuf), encoded);
    let fineParams = clampFineParams(CoopVecToArray(fineOut));

    TruncatedGMM<kNumGaussians> gmm;
    float2 t = gmm.sample(fineParams, rng);
    // Round-trip safety: TruncatedGMM.sample() goes through erfinv, which can
    // return a value marginally outside [0,1]; left unclamped that would push
    // x into a neighbouring cell, so forward() would score a different bin
    // than the one actually sampled -- a silent bias in KL-mode importance
    // weights. Clamping keeps the sample inside the cell it was drawn from.
    t = clamp(t, float2(0.0f), float2(1.0f - 1e-6f));
    float logFine = gmm.eval(fineParams, t);

    float2 x = (float2(cell) + t) / float(kGridDim);
    float logProb = logHist + logFine + 2.0f * log(float(kGridDim));

    HGSample s;
    s.x = x;
    s.logProb = logProb;
    return s;
}
