// TSNN/Flows/RQS.slang
// Robust Rational Quadratic Spline (RQS) bijection.
//
// Implements the monotone piecewise rational-quadratic transform from:
//   Durkan et al., "Neural Spline Flows" (NeurIPS 2019).
//
// Robustness is enforced via per-bin minimum width/height and a minimum
// knot-derivative, preventing degenerate (zero-width or zero-derivative) bins.
//
// API:
//   rqsForward<K>  – data → latent, returns float2(y, log|dy/dx|)
//   rqsInverse<K>  – latent → data,  returns float2(x, -log|dy/dx|)
//
// The functions are [Differentiable] w.r.t. the spline parameter arrays
// (w, h, d) and treat the transformed value as no_diff, so gradients flow
// back to the conditioner MLP during training.
//
// Parameter packing convention (matches Python NDE project):
//   w[K]   – unnormalised bin widths  (raw MLP output, indices 0..K-1)
//   h[K]   – unnormalised bin heights (raw MLP output, indices K..2K-1)
//   d[K-1] – unnormalised interior knot derivatives (indices 2K..3K-2)
//   Boundary derivatives are fixed to the identity-slope constant below.

static const float kRQSMinBinWidth   = 1e-3f;
static const float kRQSMinBinHeight  = 1e-3f;
static const float kRQSMinDerivative = 1e-3f;

// Unnormalised boundary derivative that gives ≈ 1.0 after softplus + min:
//   softplus(x) + kRQSMinDerivative = 1   =>   x = log(exp(1-1e-3) - 1) ≈ 0.5413
static const float kRQSBoundaryUnnorm = 0.5413f;

// ─── Helpers ──────────────────────────────────────────────────────────────────

// Numerically stable log(1 + exp(x)).
[Differentiable]
float rqsSoftplus(float x)
{
    return x > 20.0f ? x : log(1.0f + exp(x));
}

// ─── Forward transform ────────────────────────────────────────────────────────

// Maps x → y.  Returns float2(y, log|dy/dx|).
// Outside [-tail, tail] the transform is identity (log-det = 0).
// x is treated as no_diff: we only backpropagate through w, h, d.
[Differentiable]
float2 rqsForward<let K : uint>(
    no_diff float   x,
    float           w[K],       // unnormalised bin widths
    float           h[K],       // unnormalised bin heights
    float           d[K - 1u],  // unnormalised interior derivatives
    no_diff float   tail        // tail bound (domain = [-tail, tail])
)
{
    // Linear tail: identity outside spline domain
    if (x <= -tail || x >= tail)
        return float2(x, 0.0f);

    // ── Normalise widths via softmax + minimum enforcement ────────────────────
    float widths[K];
    float maxW = w[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxW = max(maxW, w[i]);
    float sumW = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { widths[i] = exp(w[i] - maxW); sumW += widths[i]; }
    float invSumW = 1.0f / sumW;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        widths[i] = kRQSMinBinWidth + (1.0f - kRQSMinBinWidth * float(K)) * widths[i] * invSumW;

    // ── Normalise heights ─────────────────────────────────────────────────────
    float heights[K];
    float maxH = h[0];
    [ForceUnroll] for (uint i = 1; i < K; i++) maxH = max(maxH, h[i]);
    float sumH = 0.0f;
    [ForceUnroll] for (uint i = 0; i < K; i++) { heights[i] = exp(h[i] - maxH); sumH += heights[i]; }
    float invSumH = 1.0f / sumH;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        heights[i] = kRQSMinBinHeight + (1.0f - kRQSMinBinHeight * float(K)) * heights[i] * invSumH;

    // ── Compute cumulative knot positions ─────────────────────────────────────
    float cumW[K + 1u];
    float cumH[K + 1u];
    cumW[0] = -tail;  cumH[0] = -tail;
    [ForceUnroll] for (uint i = 0; i < K; i++)
    {
        cumW[i + 1] = cumW[i] + widths[i]  * (2.0f * tail);
        cumH[i + 1] = cumH[i] + heights[i] * (2.0f * tail);
    }
    // Hard-clamp boundaries to absorb floating-point drift
    cumW[K] = tail;  cumH[K] = tail;

    // Re-derive widths/heights from the clamped cumulative arrays
    [ForceUnroll] for (uint i = 0; i < K; i++)
    {
        widths[i]  = cumW[i + 1] - cumW[i];
        heights[i] = cumH[i + 1] - cumH[i];
    }

    // ── Build knot derivatives ────────────────────────────────────────────────
    // Boundaries are fixed to ≈ 1 (identity slope); interior from network.
    float deriv[K + 1u];
    deriv[0] = kRQSMinDerivative + rqsSoftplus(kRQSBoundaryUnnorm);
    [ForceUnroll] for (uint i = 0; i < K - 1u; i++)
        deriv[i + 1] = kRQSMinDerivative + rqsSoftplus(d[i]);
    deriv[K] = kRQSMinDerivative + rqsSoftplus(kRQSBoundaryUnnorm);

    // ── Find bin: largest k s.t. cumW[k] <= x, clamped to [0, K-1] ───────────
    int bin = 0;
    [ForceUnroll] for (uint i = 0; i < K; i++)
        bin = (x >= cumW[i + 1]) ? int(i + 1) : bin;
    bin = min(bin, int(K) - 1);

    // ── Gather per-bin parameters ─────────────────────────────────────────────
    float bw    = widths[bin];
    float bh    = heights[bin];
    float dk    = deriv[bin];
    float dk1   = deriv[bin + 1];
    float cw    = cumW[bin];
    float ch    = cumH[bin];
    float delta = bh / bw;          // slope between corner knots

    // ── Rational-quadratic formula ────────────────────────────────────────────
    float theta  = clamp((x - cw) / bw, 0.0f, 1.0f);
    float t1mt   = theta * (1.0f - theta);
    float num    = bh * (delta * theta * theta + dk * t1mt);
    float denom  = delta + (dk + dk1 - 2.0f * delta) * t1mt;
    float y      = ch + num / denom;

    // log|dy/dx| = log(numerator of d²) - 2·log(denominator)
    float dnumer = delta * delta * (
        dk1 * theta * theta +
        2.0f * delta * t1mt +
        dk  * (1.0f - theta) * (1.0f - theta)
    );
    float logDet = log(dnumer) - 2.0f * log(abs(denom));

    return float2(y, logDet);
}

// ─── Inverse transform ────────────────────────────────────────────────────────

// Maps y → x by solving the rational-quadratic quadratic for theta.
// Returns float2(x, -log|dy/dx|).
// Not tagged [Differentiable]: inverse is only used for sampling, not training.
float2 rqsInverse<let K : uint>(
    float   y,
    float   w[K],
    float   h[K],
    float   d[K - 1u],
    float   tail
)
{
    if (y <= -tail || y >= tail)
        return float2(y, 0.0f);

    // Same normalisation as forward
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

    // Find bin by searching cumulative heights (inverse direction)
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

    // Solve a·θ² + b·θ + c = 0 for θ (numerically stable quadratic formula)
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
        dk1 * theta * theta +
        2.0f * delta * t1mt +
        dk  * (1.0f - theta) * (1.0f - theta)
    );
    float logDet = log(dnumer) - 2.0f * log(abs(denom));

    return float2(x, -logDet);  // negate log-det for inverse
}
