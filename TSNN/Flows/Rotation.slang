// TSNN/Flows/Rotation.slang
// 2-D rotation as an invertible flow layer (Glow's 1×1 "linear mixing", at 2-D).
//
// Coupling layers only ever transform one axis at a time, so axis-aligned
// structure builds up: affine couplings struggle to model correlations, and
// spline couplings leave visible knot "creases" along the x/y gridlines.
// Interleaving a rotation between coupling blocks re-mixes the axes, so each
// block acts in a different frame and that structure is spread out / averaged.
//
// A rotation is orthogonal with det = +1, so its Jacobian log-determinant is
// exactly 0 — it contributes nothing to the flow's log-density and needs no
// log-det bookkeeping at the call site.
//
// Note: a rotation enlarges a bounded region (the corners of [-h,h]² map
// outside the box), so it is incompatible with a hard-bounded base such as
// UniformPrior.  Use it with an unbounded base (StandardNormalPrior).

struct Rotation2D
{
    float c; // cos(angle)
    float s; // sin(angle)

    __init(float angle) { c = cos(angle); s = sin(angle); }

    // Forward mix:  R · z
    [Differentiable]
    [NoDiffThis] // the rotation parameters are fixed, not learned, so no gradient
    float2 apply(float2 z)
    {
        return float2(c * z.x - s * z.y,
                      s * z.x + c * z.y);
    }

    // Inverse mix:  Rᵀ · z
    float2 applyInverse(float2 z)
    {
        return float2( c * z.x + s * z.y,
                      -s * z.x + c * z.y);
    }
}
