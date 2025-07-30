#include "nn/optimizers/Adam.h"

Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon)
    : Optimizer(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

void Adam::update(Tensor& weights, const Tensor& grad_weights) {
    if (m_weights.getSize() == 0) {
        m_weights = Tensor(weights.getShape());
        v_weights = Tensor(weights.getShape());
        for (size_t i = 0; i < m_weights.getSize(); ++i) {
            m_weights.getCpuData()[i] = 0.0f;
            v_weights.getCpuData()[i] = 0.0f;
        }
    }

    t++;

    for (size_t i = 0; i < weights.getSize(); ++i) {
        m_weights.getCpuData()[i] = beta1 * m_weights.getCpuData()[i] + (1 - beta1) * grad_weights.getCpuData()[i];
        v_weights.getCpuData()[i] = beta2 * v_weights.getCpuData()[i] + (1 - beta2) * grad_weights.getCpuData()[i] * grad_weights.getCpuData()[i];

        float m_hat = m_weights.getCpuData()[i] / (1 - std::pow(beta1, t));
        float v_hat = v_weights.getCpuData()[i] / (1 - std::pow(beta2, t));

        weights.getCpuData()[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}
