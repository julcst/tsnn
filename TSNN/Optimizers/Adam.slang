// Adam (Kingma & Ba, 2015) with optional *decoupled* weight decay, i.e. AdamW
// (Loshchilov & Hutter, 2019).  The two optimisers differ only in a single extra
// term, so they share one implementation: when `weightDecay == 0` you get plain
// Adam, otherwise you get AdamW.
//
// Plain Adam with an L2 penalty folds the decay into the gradient, so it is
// scaled by the adaptive 1/√v̂ denominator — parameters with a large gradient
// history get decayed less, coupling regularisation to the gradient scale.
// AdamW instead applies the decay directly to the weights, outside the adaptive
// step:
//
//   θ ← θ − lr·( m̂ / (√v̂ + ε)  +  wd·θ )
//
// Decoupling makes the regularisation strength independent of the gradient
// statistics (usually generalises better) and adds a mild pull toward 0 that
// improves stability.
struct AdamConfig
{
    float learningRate = 1e-3;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-8;
    float invLossScale = 1.0;
    float weightDecay = 0.0;  // 0 → plain Adam; > 0 → decoupled weight decay (AdamW)
};

struct Adam
{
    // Parameters
    AdamConfig config;
    // Constant parameters derived from the above
    float invSqrtBiasCorrection2;
    float stepSize;

    __init(float t, AdamConfig config = AdamConfig())
    {
        this.config = config;

        let biasCorrection1 = max(1.0 - pow(config.beta1, t), 1e-20);
        this.stepSize = config.learningRate / biasCorrection1;
        this.invSqrtBiasCorrection2 = rsqrt(max(1.0 - pow(config.beta2, t), 1e-20));
    }

    bool step<uint N>(inout vector<float, N> param, inout vector<float, N> moment1, inout vector<float, N> moment2, vector<float, N> gradient)
    {
        gradient *= config.invLossScale;
        moment1 = lerp(gradient, moment1, config.beta1);
        moment2 = lerp(gradient * gradient, moment2, config.beta2);
        let denom = sqrt(moment2) * invSqrtBiasCorrection2 + config.epsilon;
        var step = (moment1 / denom) * stepSize;
        // AdamW: decoupled weight decay applied outside the adaptive step, using
        // the raw lr (not the bias-corrected stepSize).  Guarded so plain Adam
        // (weightDecay == 0) pays nothing — when the host fixes weightDecay at a
        // compile-time-known value the branch folds away entirely.
        if (config.weightDecay != 0.0f)
            step += (config.learningRate * config.weightDecay) * param;
        param -= step;
        return dot(step, step) > 1e-6; // Return whether any parameter had a significant update
    }
};