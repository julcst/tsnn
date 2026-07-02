// TSNN/Optimizers/Sanitizer.slang
// Gradient sanitisation for the optimiser hot path.
//
// A single bad backward pass — a NaN/Inf from an overflowed accumulator, or a
// huge spike from an unlucky batch — can poison the Adam moment buffers and
// stall the whole run.  `sanitize` scrubs non-finite values to a fallback and
// (optionally) clamps the magnitude, so the optimiser only ever sees finite,
// bounded gradients.
//
// `sanitizeDebug` does the same but tallies how often each pathology fires into
// a SanitizeDebugCounter — that tally is the signal you tune the loss scale and
// clip threshold against (too much overflow → lower the loss scale; too much
// clipping → raise it, or raise the clip).

// Replace NaN/Inf with `fallback`, then clamp to [-clip, clip].
// `clip <= 0` disables clamping (pure non-finite scrub).
vector<float, N> sanitize<uint N>(vector<float, N> g, float clip = 0.0f, float fallback = 0.0f)
{
    [ForceUnroll]
    for (uint i = 0; i < N; ++i)
        g[i] = (isnan(g[i]) || isinf(g[i])) ? fallback : g[i];
    if (clip > 0.0f)
        g = clamp(g, vector<float, N>(-clip), vector<float, N>(clip));
    return g;
}

float sanitize(float g, float clip = 0.0f, float fallback = 0.0f)
{
    g = (isnan(g) || isinf(g)) ? fallback : g;
    if (clip > 0.0f)
        g = clamp(g, -clip, clip);
    return g;
}

// Per-thread tally of gradient pathologies, flushed once at the end of the
// kernel to keep the atomic traffic off the inner loop.  Mirrors the 4 × uint32
// layout the host reads back:
//   [0] nonFinite – NaN/Inf gradients (accumulator overflowed)
//   [1] clipped   – |g| exceeded the clip threshold (saturation)
//   [2] zero      – exactly-zero gradients (underflow / dead-unit proxy)
//   [3] total     – elements processed
struct SanitizeDebugCounter
{
    uint nonFinite;
    uint clipped;
    uint zero;
    uint total;

    __init()
    {
        nonFinite = 0u;
        clipped   = 0u;
        zero      = 0u;
        total     = 0u;
    }

    // Classify one raw (pre-sanitise) gradient element against `clip`.
    [mutating]
    void record(float g, float clip)
    {
        total++;
        if (isnan(g) || isinf(g))
        {
            nonFinite++;            // accumulator overflowed
        }
        else
        {
            if (g == 0.0f)                       zero++;     // underflowed / dead
            if (clip > 0.0f && abs(g) > clip)    clipped++;  // clip saturated
        }
    }

    [mutating]
    void record<uint N>(vector<float, N> g, float clip)
    {
        [ForceUnroll]
        for (uint i = 0; i < N; ++i)
            record(g[i], clip);
    }

    // Atomically merge this thread's tally into a 4 × uint32 counter block at
    // `byteOffset` (default 0).  Pass a non-zero offset to flush several
    // independent counters (e.g. per-buffer) into one backing buffer.
    void flush(RWByteAddressBuffer buffer, uint byteOffset = 0u)
    {
        uint prev;
        buffer.InterlockedAdd(byteOffset + 0u,  nonFinite, prev);
        buffer.InterlockedAdd(byteOffset + 4u,  clipped,   prev);
        buffer.InterlockedAdd(byteOffset + 8u,  zero,      prev);
        buffer.InterlockedAdd(byteOffset + 12u, total,     prev);
    }
}

// Sanitise + tally in one pass.  The pathology classification uses the raw
// gradient so the counts reflect what actually came out of the backward pass.
vector<float, N> sanitizeDebug<uint N>(vector<float, N> g, float clip,
                                       inout SanitizeDebugCounter counter,
                                       float fallback = 0.0f)
{
    counter.record(g, clip);
    return sanitize(g, clip, fallback);
}
