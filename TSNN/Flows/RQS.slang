// TSNN/Flows/RQS.slang
// Robust Rational Quadratic Spline (RQS) bijection.
//
// Implements the monotone piecewise rational-quadratic transform from:
//   Durkan et al., "Neural Spline Flows" (NeurIPS 2019).
//
// Robustness is enforced via per-bin minimum width/height and a minimum
// knot-derivative, preventing degenerate (zero-width or zero-derivative) bins.
//
// API (array form):
//   rqsForward<K>(no_diff x, w[K], h[K], d[K-1], tail) -> RQSResult
//   rqsInverse<K>(y, w[K], h[K], d[K-1], tail)         -> RQSResult
//
// API (CoopVec form – preferred at call sites):
//   rqsForward<K>(no_diff x, CoopVec<half,3K-1>, tail)  -> RQSResult
//   rqsInverse<K>(y,         CoopVec<half,3K-1>, tail)  -> RQSResult
//
// The CoopVec overloads unpack (w, h, d) from the packed MLP output internally,
// which is the natural call convention from a conditioner MLP.
//
// Parameter packing (matches Python NDE project):
//   [0..K-1]    unnormalised bin widths
//   [K..2K-1]   unnormalised bin heights
//   [2K..3K-2]  unnormalised interior knot derivatives
//   Boundary derivatives are fixed to the identity-slope constant.

import TSNN.CoopVec;

static const float kRQSMinBinWidth   = 1e-3f;
static const float kRQSMinBinHeight  = 1e-3f;
static const float kRQSMinDerivative = 1e-3f;

// softplus_inv(1 - kRQSMinDerivative) ≈ 0.5413
// Use as the initial unnormalised boundary derivative so knot slopes start at 1.
static const float kRQSBoundaryUnnorm = 0.5413f;

// ─── Result type ──────────────────────────────────────────────────────────────

struct RQSResult : IDifferentiable {
    typealias Differential = RQSResult;

    float value;      // transformed coordinate
    float logAbsDet;  // log|dy/dx| (forward) or -log|dy/dx| (inverse)

    static RQSResult dzero()
    {
        RQSResult r;
        r.value     = 0.0f;
        r.logAbsDet = 0.0f;
        return r;
    }

    static RQSResult dadd(RQSResult a, RQSResult b)
    {
        RQSResult r;
        r.value     = a.value     + b.value;
        r.logAbsDet = a.logAbsDet + b.logAbsDet;
        return r;
    }
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

[Differentiable]
float rqsSoftplus(float x)
{
    return x > 20.0f ? x : log(1.0f + exp(x));
}

// ─── Array-form forward transform ────────────────────────────────────────────

[Differentiable]
RQSResult rqsForward<let K : uint>(
    no_diff float   x,
    float           w[K],
    float           h[K],
    float           d[K - 1u],
    no_diff float   tail
)
{
    if (x <= -tail || x >= tail)
    {
        RQSResult r = { x, 0.0f };
        return r;
    }

    float widths[K];
    float maxW = w[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxW = max(maxW, w[i]);
    float sumW = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { widths[i] = exp(w[i] - maxW); sumW += widths[i]; }
    float invSumW = 1.0f / sumW;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        widths[i] = kRQSMinBinWidth + (1.0f - kRQSMinBinWidth * float(K)) * widths[i] * invSumW;

    float heights[K];
    float maxH = h[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxH = max(maxH, h[i]);
    float sumH = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { heights[i] = exp(h[i] - maxH); sumH += heights[i]; }
    float invSumH = 1.0f / sumH;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        heights[i] = kRQSMinBinHeight + (1.0f - kRQSMinBinHeight * float(K)) * heights[i] * invSumH;

    float cumW[K + 1u];
    float cumH[K + 1u];
    cumW[0] = -tail;  cumH[0] = -tail;
    [ForceUnroll] for (uint i = 0; i < K; i++)
    {
        cumW[i + 1] = cumW[i] + widths[i]  * (2.0f * tail);
        cumH[i + 1] = cumH[i] + heights[i] * (2.0f * tail);
    }
    cumW[K] = tail;  cumH[K] = tail;
    [ForceUnroll] for (uint i = 0; i < K; i++)
    {
        widths[i]  = cumW[i + 1] - cumW[i];
        heights[i] = cumH[i + 1] - cumH[i];
    }

    float deriv[K + 1u];
    deriv[0] = kRQSMinDerivative + rqsSoftplus(kRQSBoundaryUnnorm);
    [ForceUnroll] for (uint i = 0; i < K - 1u; i++)
        deriv[i + 1] = kRQSMinDerivative + rqsSoftplus(d[i]);
    deriv[K] = kRQSMinDerivative + rqsSoftplus(kRQSBoundaryUnnorm);

    int bin = 0;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        bin = (x >= cumW[i + 1]) ? int(i + 1) : bin;
    bin = min(bin, int(K) - 1);

    float bw    = widths[bin];
    float bh    = heights[bin];
    float dk    = deriv[bin];
    float dk1   = deriv[bin + 1];
    float cw    = cumW[bin];
    float ch    = cumH[bin];
    float delta = bh / bw;

    float theta  = clamp((x - cw) / bw, 0.0f, 1.0f);
    float t1mt   = theta * (1.0f - theta);
    float num    = bh * (delta * theta * theta + dk * t1mt);
    float denom  = delta + (dk + dk1 - 2.0f * delta) * t1mt;
    float y      = ch + num / denom;

    float dnumer = delta * delta * (
        dk1 * theta * theta + 2.0f * delta * t1mt + dk * (1.0f - theta) * (1.0f - theta)
    );
    float logDet = log(dnumer) - 2.0f * log(abs(denom));

    RQSResult r = { y, logDet };
    return r;
}

// ─── CoopVec-form forward (unpacks packed MLP output internally) ──────────────

[Differentiable]
RQSResult rqsForward<let K : uint>(
    no_diff float               x,
    CoopVec<half, 3u * K - 1u> cv,
    no_diff float               tail
)
{
    let arr = CoopVecToArray(cv);
    float w[K], h[K], d[K - 1u];
    [ForceUnroll] for (uint i = 0u; i < K;        i++) w[i] = arr[i];
    [ForceUnroll] for (uint i = 0u; i < K;        i++) h[i] = arr[K + i];
    [ForceUnroll] for (uint i = 0u; i < K - 1u;   i++) d[i] = arr[2u * K + i];
    return rqsForward<K>(x, w, h, d, tail);
}

// ─── Array-form inverse transform ────────────────────────────────────────────

RQSResult rqsInverse<let K : uint>(
    float   y,
    float   w[K],
    float   h[K],
    float   d[K - 1u],
    float   tail
)
{
    if (y <= -tail || y >= tail)
    {
        RQSResult r = { y, 0.0f };
        return r;
    }

    float widths[K];
    float maxW = w[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxW = max(maxW, w[i]);
    float sumW = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { widths[i] = exp(w[i] - maxW); sumW += widths[i]; }
    float invSumW = 1.0f / sumW;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        widths[i] = kRQSMinBinWidth + (1.0f - kRQSMinBinWidth * float(K)) * widths[i] * invSumW;

    float heights[K];
    float maxH = h[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxH = max(maxH, h[i]);
    float sumH = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { heights[i] = exp(h[i] - maxH); sumH += heights[i]; }
    float invSumH = 1.0f / sumH;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        heights[i] = kRQSMinBinHeight + (1.0f - kRQSMinBinHeight * float(K)) * heights[i] * invSumH;

    float cumW[K + 1u];
    float cumH[K + 1u];
    cumW[0] = -tail;  cumH[0] = -tail;
    [ForceUnroll] for (uint i = 0; i < K; i++)
    {
        cumW[i + 1] = cumW[i] + widths[i]  * (2.0f * tail);
        cumH[i + 1] = cumH[i] + heights[i] * (2.0f * tail);
    }
    cumW[K] = tail;  cumH[K] = tail;
    [ForceUnroll] for (uint i = 0; i < K; i++)
    {
        widths[i]  = cumW[i + 1] - cumW[i];
        heights[i] = cumH[i + 1] - cumH[i];
    }

    float deriv[K + 1u];
    deriv[0] = kRQSMinDerivative + rqsSoftplus(kRQSBoundaryUnnorm);
    [ForceUnroll] for (uint i = 0; i < K - 1u; i++)
        deriv[i + 1] = kRQSMinDerivative + rqsSoftplus(d[i]);
    deriv[K] = kRQSMinDerivative + rqsSoftplus(kRQSBoundaryUnnorm);

    int bin = 0;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        bin = (y >= cumH[i + 1]) ? int(i + 1) : bin;
    bin = min(bin, int(K) - 1);

    float bw    = widths[bin];
    float bh    = heights[bin];
    float dk    = deriv[bin];
    float dk1   = deriv[bin + 1];
    float cw    = cumW[bin];
    float ch    = cumH[bin];
    float delta = bh / bw;
    float s     = y - ch;
    float A     = dk + dk1 - 2.0f * delta;

    float a    = s * A + bh * (delta - dk);
    float b    = bh * dk - s * A;
    float c    = -delta * s;
    float disc = max(b * b - 4.0f * a * c, 0.0f);
    float theta = (2.0f * c) / (-b - sqrt(disc));
    theta = clamp(theta, 0.0f, 1.0f);

    float x      = theta * bw + cw;
    float t1mt   = theta * (1.0f - theta);
    float denom  = delta + A * t1mt;
    float dnumer = delta * delta * (
        dk1 * theta * theta + 2.0f * delta * t1mt + dk * (1.0f - theta) * (1.0f - theta)
    );
    float logDet = log(dnumer) - 2.0f * log(abs(denom));

    RQSResult r = { x, -logDet };
    return r;
}

// ─── CoopVec-form inverse ─────────────────────────────────────────────────────

RQSResult rqsInverse<let K : uint>(
    float               y,
    CoopVec<half, 3u * K - 1u> cv,
    float               tail
)
{
    let arr = CoopVecToArray(cv);
    float w[K], h[K], d[K - 1u];
    [ForceUnroll] for (uint i = 0u; i < K;       i++) w[i] = arr[i];
    [ForceUnroll] for (uint i = 0u; i < K;       i++) h[i] = arr[K + i];
    [ForceUnroll] for (uint i = 0u; i < K - 1u;  i++) d[i] = arr[2u * K + i];
    return rqsInverse<K>(y, w, h, d, tail);
}
