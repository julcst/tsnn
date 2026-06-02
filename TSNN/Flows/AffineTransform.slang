// TSNN/Flows/AffineTransform.slang
// Affine bijection: y = x · exp(s) + t,  log|dy/dx| = s.
//
// API (scalar form):
//   affineForward(no_diff x, s_raw, t) -> AffineResult   [Differentiable]
//   affineInverse(y, s_raw, t)         -> AffineResult
//
// API (CoopVec form – preferred at call sites):
//   affineForward(no_diff x, CoopVec<half,2>) -> AffineResult   [Differentiable]
//   affineInverse(y,         CoopVec<half,2>) -> AffineResult
//
// The MLP outputs an *unbounded* raw log-scale s_raw; we soft-clamp it to
//   s = kAffineMaxScale · (2·sigmoid(s_raw) − 1) ∈ (−kAffineMaxScale, kAffineMaxScale)
// Pure affine coupling needs no activation on s in theory, but in practice an
// unbounded s makes exp(s) explode to NaN during training (observed directly).
// The bound is generous (±4 ⇒ scale ∈ [0.018, 55]), enough to represent e.g. a
// 10× scale (s≈2.3) while keeping exp(s) finite.  Implemented via exp() only,
// since exp has a registered backward in CoopVec.slang (tanh may not).
//
// Identity init: MLP output weights & biases = 0 ⇒ s_raw = 0 ⇒ s = 0, t = 0.

import TSNN.CoopVec;

static const float kAffineMaxScale = 4.0f;

// ─── Result type ──────────────────────────────────────────────────────────────

struct AffineResult : IDifferentiable {
    typealias Differential = AffineResult;

    float value;      // transformed coordinate
    float logAbsDet;  // log|dy/dx| (= s for forward, = -s for inverse)

    static AffineResult dzero()
    {
        AffineResult r;
        r.value     = 0.0f;
        r.logAbsDet = 0.0f;
        return r;
    }

    static AffineResult dadd(AffineResult a, AffineResult b)
    {
        AffineResult r;
        r.value     = a.value     + b.value;
        r.logAbsDet = a.logAbsDet + b.logAbsDet;
        return r;
    }
};

// sigmoid using only exp() — exp has a registered backward in CoopVec.slang.
[Differentiable]
float affineSigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Soft-clamped effective log-scale.
[Differentiable]
float affineScale(float s_raw) { return kAffineMaxScale * (2.0f * affineSigmoid(s_raw) - 1.0f); }

// ─── Scalar-form forward ──────────────────────────────────────────────────────

// NOTE: x (the transformed coordinate) is *differentiable*, not no_diff.  In a
// multi-layer coupling flow x is the output of previous layers, so the Jacobian
// must chain through it; marking it no_diff severs backprop to earlier layers'
// parameters and yields a wrong gradient.  The raw data input is passed in as a
// no_diff value at the top level, which correctly contributes zero data-gradient.
[Differentiable]
AffineResult affineForward(float x, float s_raw, float t)
{
    float s = affineScale(s_raw);
    AffineResult r;
    r.value     = x * exp(s) + t;
    r.logAbsDet = s;
    return r;
}

// ─── CoopVec-form forward (cv = [s_raw, t]) ──────────────────────────────────

[Differentiable]
AffineResult affineForward(float x, CoopVec<half, 2u> cv)
{
    let arr = CoopVecToArray(cv);
    return affineForward(x, arr[0], arr[1]);
}

// ─── Scalar-form inverse ──────────────────────────────────────────────────────

AffineResult affineInverse(float y, float s_raw, float t)
{
    float s = affineScale(s_raw);
    AffineResult r;
    r.value     = (y - t) * exp(-s);
    r.logAbsDet = -s;
    return r;
}

// ─── CoopVec-form inverse ─────────────────────────────────────────────────────

AffineResult affineInverse(float y, CoopVec<half, 2u> cv)
{
    let arr = CoopVecToArray(cv);
    return affineInverse(y, arr[0], arr[1]);
}
