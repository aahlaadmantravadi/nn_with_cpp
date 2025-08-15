#include "nn/optimizers/Adam.h"



Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon)
    : Optimizer{learning_rate}, beta1{beta1}, beta2{beta2}, epsilon{epsilon}
{
}



void Adam::update(Tensor &weights, const Tensor &grad_weights)
{
    const void *key = static_cast<const void *>(&weights);
    auto &moments = state_by_param[key];

    if (moments.m.getSize() == 0)
    {
        moments.m = Tensor{weights.getShape()};
        moments.v = Tensor{weights.getShape()};
        for (size_t i = 0; i < moments.m.getSize(); i++)
        {
            moments.m.getCpuData()[i] = 0.0f;
            moments.v.getCpuData()[i] = 0.0f;
        }
        moments.t = 0;
    }

    moments.t++;

    for (size_t i = 0; i < weights.getSize(); i++)
    {
        moments.m.getCpuData()[i] = beta1 * moments.m.getCpuData()[i] + (1 - beta1) * grad_weights.getCpuData()[i];
        moments.v.getCpuData()[i] = beta2 * moments.v.getCpuData()[i] + (1 - beta2) * grad_weights.getCpuData()[i] * grad_weights.getCpuData()[i];

        float m_hat = moments.m.getCpuData()[i] / (1 - std::pow(beta1, moments.t));
        float v_hat = moments.v.getCpuData()[i] / (1 - std::pow(beta2, moments.t));

        weights.getCpuData()[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}
