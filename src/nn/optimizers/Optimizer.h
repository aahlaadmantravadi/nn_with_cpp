#pragma once



#include "nn/Tensor.h"



class Optimizer
{
public:
    Optimizer(float learning_rate = 0.01f);
    virtual ~Optimizer() = default;
    virtual void update(Tensor &weights, const Tensor &grad_weights) = 0;
    void setLearningRate(float lr) { learning_rate = lr; }

protected:
    float learning_rate;
};
