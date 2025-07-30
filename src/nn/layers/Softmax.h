#pragma once

#include "nn/layers/Layer.h"

class Softmax : public Layer {
public:
    Softmax();
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

private:
    Tensor last_output;
};
