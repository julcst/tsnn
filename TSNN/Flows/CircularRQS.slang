// TSNN/Flows/CircularRQS.slang
// Circular (periodic) Rational Quadratic Spline bijection.
//
// Same monotone piecewise rational-quadratic transform as RQS.slang (Durkan et al.,
// "Neural Spline Flows", NeurIPS 2019), but wired for a coordinate that lives on a
// circle of circumference 2·tail instead of a bounded interval [-tail, tail]:
//
//   · RQS.slang fixes both boundary derivatives to an identity slope and returns the
//     identity transform (with logAbsDet = 0) outside [-tail, tail] — appropriate for
//     a coordinate with genuine endpoints (e.g. cosθ).
//   · CircularRQS instead identifies x = -tail with x = +tail (same point on the
//     circle) and shares ONE learned derivative between the first and last knot, so
//     the spline is C¹-continuous across the wrap. There is no "outside the domain":
//     any x is first wrapped modulo the circumference, so the seam is treated as an
//     ordinary interior knot rather than a boundary.
//
// This is the standard "circular spline flow" construction (e.g. Rezende et al.,
// "Normalizing Flows on Tori and Spheres", ICML 2020) applied to Durkan's RQS —
// appropriate for the azimuthal angle φ of NISTracer's cylindrical direction
// parameterisation (NIS_DIR_PARAM=1 in Common.slang), where φ=0 and φ=2π are the
// same direction and the learned density/derivative must agree there.
//
// API (array form):
//   crqsForward<K>(x, w[K], h[K], d[K], tail) -> FlowStep   [Differentiable]
//   crqsInverse<K>(y, w[K], h[K], d[K], tail) -> FlowStep
//
// API (CoopVec form – preferred at call sites):
//   crqsForward<K>(x, CoopVec<half,3K>, tail)  -> FlowStep   [Differentiable]
//   crqsInverse<K>(y, CoopVec<half,3K>, tail)  -> FlowStep
//
// The packed parameter order is [w(K) | h(K) | d(K)] — one more derivative than
// RQS.slang's [w(K) | h(K) | d(K-1)] since all K knot derivatives are learned (the
// wrap knot reuses d[0], it is not a separate parameter).
//
// NOTE: mirrors RQS.slang's inlining discipline — the spline decode is kept inlined
// in each entry point (factoring it into helpers that return arrays/structs through
// the autodiff path miscompiles on this Slang backend, per RQS.slang:24-28).

import TSNN.CoopVec;
import TSNN.Flows.Bijection;

static const float kCRQSMinBinWidth   = 1e-3f;
static const float kCRQSMinBinHeight  = 1e-3f;
static const float kCRQSMinDerivative = 1e-3f;

[Differentiable]
float crqsSoftplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// Wrap x into [-tail, tail) — identifies x = tail with x = -tail (same circle point).
// Measure-preserving (derivative 1 a.e.), so it never contributes to logAbsDet.
// `no_diff` prefixes the floor(...) subexpression (same idiom as NIS.slang's
// `no_diff encodeOneBlob(...)`): the wrap COUNT is detached from autodiff, so the
// only differentiable term left is `x` itself — d(wrap)/dx = 1 a.e., as required
// for a measure-preserving reparameterisation.
[Differentiable]
float crqsWrap(float x, no_diff float tail)
{
    float span = 2.0f * tail;
    float k = no_diff floor((x + tail) / span);
    return x - span * k;
}

// ─── Array-form forward transform ────────────────────────────────────────────

[Differentiable]
FlowStep crqsForward<let K : uint>(float xIn, float w[K], float h[K], float d[K], no_diff float tail)
{
    float x = crqsWrap(xIn, tail);

    // widths: softmax + min-width floor, scaled to span 2·tail
    float widths[K];
    float maxW = w[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxW = max(maxW, w[i]);
    float sumW = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { widths[i] = exp(w[i] - maxW); sumW += widths[i]; }
    float invSumW = 1.0f / sumW;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        widths[i] = kCRQSMinBinWidth + (1.0f - kCRQSMinBinWidth * float(K)) * widths[i] * invSumW;

    // heights: softmax + min-height floor
    float heights[K];
    float maxH = h[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxH = max(maxH, h[i]);
    float sumH = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { heights[i] = exp(h[i] - maxH); sumH += heights[i]; }
    float invSumH = 1.0f / sumH;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        heights[i] = kCRQSMinBinHeight + (1.0f - kCRQSMinBinHeight * float(K)) * heights[i] * invSumH;

    // cumulative knot positions over [-tail, tail]
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

    // knot derivatives: softplus + floor, ALL K learned; knot K wraps to knot 0
    // (same value) instead of RQS.slang's fixed identity-slope boundary.
    float deriv[K + 1u];
    [ForceUnroll] for (uint i = 0; i < K; i++)
        deriv[i] = kCRQSMinDerivative + crqsSoftplus(d[i]);
    deriv[K] = deriv[0];

    int bin = 0;
    [ForceUnroll] for (uint i = 0; i < K; i++) bin = (x >= cumW[i + 1]) ? int(i + 1) : bin;
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

    FlowStep r = { y, logDet };
    return r;
}

// ─── Array-form inverse transform ────────────────────────────────────────────

FlowStep crqsInverse<let K : uint>(float yIn, float w[K], float h[K], float d[K], float tail)
{
    float y = crqsWrap(yIn, tail);

    float widths[K];
    float maxW = w[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxW = max(maxW, w[i]);
    float sumW = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { widths[i] = exp(w[i] - maxW); sumW += widths[i]; }
    float invSumW = 1.0f / sumW;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        widths[i] = kCRQSMinBinWidth + (1.0f - kCRQSMinBinWidth * float(K)) * widths[i] * invSumW;

    float heights[K];
    float maxH = h[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxH = max(maxH, h[i]);
    float sumH = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { heights[i] = exp(h[i] - maxH); sumH += heights[i]; }
    float invSumH = 1.0f / sumH;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        heights[i] = kCRQSMinBinHeight + (1.0f - kCRQSMinBinHeight * float(K)) * heights[i] * invSumH;

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
    [ForceUnroll] for (uint i = 0; i < K; i++)
        deriv[i] = kCRQSMinDerivative + crqsSoftplus(d[i]);
    deriv[K] = deriv[0];

    int bin = 0;
    [ForceUnroll] for (uint i = 0; i < K; i++) bin = (y >= cumH[i + 1]) ? int(i + 1) : bin;
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

    FlowStep r = { x, -logDet };
    return r;
}

// ─── CoopVec-form overloads (unpack packed [w|h|d] internally) ────────────────

[Differentiable]
FlowStep crqsForward<let K : uint>(float x, CoopVec<half, 3u * K> cv, no_diff float tail)
{
    let arr = CoopVecToArray(cv);
    float w[K], h[K], d[K];
    [ForceUnroll] for (uint i = 0u; i < K; i++) w[i] = arr[i];
    [ForceUnroll] for (uint i = 0u; i < K; i++) h[i] = arr[K + i];
    [ForceUnroll] for (uint i = 0u; i < K; i++) d[i] = arr[2u * K + i];
    return crqsForward<K>(x, w, h, d, tail);
}

FlowStep crqsInverse<let K : uint>(float y, CoopVec<half, 3u * K> cv, float tail)
{
    let arr = CoopVecToArray(cv);
    float w[K], h[K], d[K];
    [ForceUnroll] for (uint i = 0u; i < K; i++) w[i] = arr[i];
    [ForceUnroll] for (uint i = 0u; i < K; i++) h[i] = arr[K + i];
    [ForceUnroll] for (uint i = 0u; i < K; i++) d[i] = arr[2u * K + i];
    return crqsInverse<K>(y, w, h, d, tail);
}
