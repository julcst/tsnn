// TSNN/Flows/Bijection.slang
// Shared result type for 1-D flow bijections.
//
// Every scalar coupling transform (affine, rational-quadratic spline, …) and the
// coupling layer that wraps them produce the same two numbers: the transformed
// coordinate and the log of the local |Jacobian| (signed: + for forward, − for
// inverse).  They used to each carry an identical copy of this struct with a
// hand-written dzero/dadd; this is the single shared definition.
//
// Slang synthesises the IDifferentiable members (Differential, dzero, dadd)
// automatically because every stored field is itself differentiable, so there
// is nothing to write by hand.
struct FlowStep : IDifferentiable
{
    float value;      // transformed coordinate
    float logAbsDet;  // log|dy/dx|, signed (+ forward, − inverse)
}
