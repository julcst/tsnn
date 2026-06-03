// TSNN/Flows/Prior.slang
// Base distributions for 2-D normalizing flows.
//
// A flow defines p_x(x) = p_z(f(x)) · |det J_f(x)|, where p_z is this base
// (prior).  For training we need p_z to be *evaluatable* (log-density, used in
// the differentiable forward pass) and for generation *samplable* (map uniform
// noise → a latent draw, then run the inverse flow).
//
// IPrior captures exactly those two operations.  Concrete priors:
//   · StandardNormalPrior — N(0, I); unbounded support; rotation-invariant, so
//     it composes cleanly with Rotation2D mixing layers and unbounded (affine)
//     couplings.
//   · UniformPrior        — uniform on [-h, h]²; the natural base for a flow
//     whose couplings map within a bounded box (e.g. RQS with linear tails).
//     Do NOT combine with Rotation2D: a rotation pushes latents outside the box
//     where the density is zero (log-density −∞).

interface IPrior
{
    // Log base-density at latent z. Differentiable in z: during training z is a
    // function of the network weights, so this gradient must flow.
    [Differentiable] float logProb(float2 z);

    // Map uniform u ∈ [0,1)² to a latent sample z ~ p_z.
    float2 sample(no_diff float2 u01);
}

// ─── Standard bivariate normal N(0, I) ────────────────────────────────────────

struct StandardNormalPrior : IPrior
{
    [Differentiable]
    float logProb(float2 z)
    {
        return -0.5f * (z.x * z.x + z.y * z.y) - log(2.0f * float.getPi());
    }

    // Box–Muller: two uniforms → two independent standard normals.
    float2 sample(no_diff float2 u01)
    {
        float r   = sqrt(-2.0f * log(max(u01.x, 1e-7f)));
        float phi = 2.0f * float.getPi() * u01.y;
        return float2(r * cos(phi), r * sin(phi));
    }
}

// ─── Uniform on [-halfWidth, halfWidth]² ──────────────────────────────────────

struct UniformPrior : IPrior
{
    float halfWidth;

    __init(float halfWidth) { this.halfWidth = halfWidth; }

    // Constant inside the box.  A bounded flow (couplings that map the box onto
    // itself) keeps z in-support by construction, so we return the constant
    // log-density unconditionally; its gradient w.r.t. z is zero.
    [Differentiable]
    float logProb(float2 z)
    {
        return -log(4.0f * halfWidth * halfWidth); // -log((2·halfWidth)²)
    }

    float2 sample(no_diff float2 u01)
    {
        return (u01 * 2.0f - 1.0f) * halfWidth;
    }
}
