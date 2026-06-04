// TSNN/Optimizers/AdamW.slang
// AdamW — Adam with *decoupled* weight decay (Loshchilov & Hutter, 2019).
//
// Plain Adam with an L2 penalty folds the decay term into the gradient, so it is
// scaled by the adaptive 1/√v̂ denominator — parameters with large gradient
// history get decayed less, which couples regularisation to the gradient scale.
// AdamW instead applies the decay directly to the weights, outside the adaptive
// step:
//
//   θ ← θ − lr·( m̂ / (√v̂ + ε)  +  wd·θ )
//
// This is the same update as Adam when weightDecay = 0, but decoupling makes the
// regularisation strength independent of the gradient statistics, which usually
// generalises better and adds a mild pull toward 0 that improves stability.
//
// Shares AdamConfig with Adam (uses its weightDecay + invLossScale fields).

import Adam;

struct AdamW
{
    AdamConfig config;
    float invSqrtBiasCorrection2;
    float stepSize;

    __init(float t, AdamConfig config = AdamConfig())
    {
        this.config = config;
        let biasCorrection1 = max(1.0 - pow(config.beta1, t), 1e-20);
        this.stepSize = config.learningRate / biasCorrection1;
        this.invSqrtBiasCorrection2 = rsqrt(max(1.0 - pow(config.beta2, t), 1e-20));
    }

    bool step<uint N>(inout vector<float, N> param, inout vector<float, N> moment1,
                      inout vector<float, N> moment2, vector<float, N> gradient)
    {
        gradient *= config.invLossScale;
        moment1 = lerp(gradient, moment1, config.beta1);
        moment2 = lerp(gradient * gradient, moment2, config.beta2);
        let denom = sqrt(moment2) * invSqrtBiasCorrection2 + config.epsilon;
        // Adaptive Adam step + decoupled weight decay (decay uses the raw lr, not
        // the bias-corrected stepSize, per the AdamW formulation).
        let step = (moment1 / denom) * stepSize + (config.learningRate * config.weightDecay) * param;
        param -= step;
        return dot(step, step) > 1e-6;
    }
};
