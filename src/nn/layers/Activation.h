#pragma once

#include "nn/layers/Layer.h"
#include "nn/nn_types.h"
#include <functional>

class Activation : public Layer {
public:
    Activation(ActivationType type);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

private:
    ActivationType type;
    std::function<float(float)> activation_func;
    std::function<float(float)> derivative_func;
    Tensor last_input;
};
