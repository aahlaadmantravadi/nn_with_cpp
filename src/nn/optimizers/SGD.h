#pragma once

#include "nn/optimizers/Optimizer.h"

class SGD : public Optimizer {
public:
    SGD(float learning_rate = 0.01f);
    void update(Tensor& weights, const Tensor& grad_weights) override;
};
