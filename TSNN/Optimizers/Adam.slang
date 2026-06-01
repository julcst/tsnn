struct AdamConfig
{
    float learningRate = 1e-3;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-8;
    float invLossScale = 1.0;
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
        let step = (moment1 / denom) * stepSize;
        param -= step;
        return dot(step, step) > 1e-6; // Return whether any parameter had a significant update
    }
};