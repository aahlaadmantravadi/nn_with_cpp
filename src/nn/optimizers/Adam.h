#pragma once

#include "nn/optimizers/Optimizer.h"
#include "nn/Tensor.h"

class Adam : public Optimizer {
public:
    Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8);
    void update(Tensor& weights, const Tensor& grad_weights) override;

private:
    float beta1;
    float beta2;
    float epsilon;
    Tensor m_weights;
    Tensor v_weights;
    int t;
};
