struct Adam
{
    // Parameters
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    // Constant parameters derived from the above
    float invSqrtBiasCorrection2;
    float stepSize;
    float invLossScale;

    __init(float learningRate, float beta1, float beta2, float epsilon, float t, float lossScale = 1.0)
    {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        let biasCorrection1 = max(1.0 - pow(beta1, t), 1e-20);
        this.stepSize = learningRate / biasCorrection1;
        this.invSqrtBiasCorrection2 = rsqrt(max(1.0 - pow(beta2, t), 1e-20));
        this.invLossScale = 1.0 / lossScale;
    }

    bool step<uint N>(inout vector<float, N> param, inout vector<float, N> moment1, inout vector<float, N> moment2, vector<float, N> gradient)
    {
        gradient *= invLossScale;
        moment1 = lerp(gradient, moment1, beta1);
        moment2 = lerp(gradient * gradient, moment2, beta2);
        let denom = sqrt(moment2) * invSqrtBiasCorrection2 + epsilon;
        let step = (moment1 / denom) * stepSize;
        param -= step;
        return dot(step, step) > 1e-6; // Return whether any parameter had a significant update
    }
    
};