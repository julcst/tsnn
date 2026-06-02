// TSNN/Flows/CouplingLayer.slang
// Coupling-layer wrappers around the RQS transform.
//
// A coupling layer in a normalizing flow splits the input into two parts:
//   z_cond – the conditioning dimension (passed through unchanged)
//   z_mod  – the dimension being transformed
//
// The conditioner MLP maps z_cond → spline parameters (w, h, d), which
// define the monotone RQS bijection applied to z_mod.
//
// applyCouplingForward  – used during training (differentiable w.r.t. w, h, d)
// applyCouplingInverse  – used during sampling (not differentiable)
//
// Both return float2(transformed_value, log_abs_det).

import TSNN.Flows.RQS;

// Forward coupling: transform z_mod given RQS params derived from z_cond.
// z_mod is no_diff (data), gradients flow through w, h, d → conditioner MLP.
[Differentiable]
float2 applyCouplingForward<let K : uint>(
    no_diff float   z_mod,  // value to transform (no gradient needed)
    float           w[K],   // unnormalised bin widths  from conditioner MLP
    float           h[K],   // unnormalised bin heights from conditioner MLP
    float           d[K - 1u], // unnormalised interior derivatives from conditioner MLP
    no_diff float   tail    // linear-tail bound
)
{
    return rqsForward<K>(z_mod, w, h, d, tail);
}

// Inverse coupling: recover z_mod from y_mod (for sampling, no backprop needed).
float2 applyCouplingInverse<let K : uint>(
    float   y_mod,
    float   w[K],
    float   h[K],
    float   d[K - 1],
    float   tail
)
{
    return rqsInverse<K>(y_mod, w, h, d, tail);
}
