#pragma once

#include "nn/Tensor.h"

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update(class Optimizer& optimizer) {} // Default no-op for layers without weights
    const Tensor& getLastOutput() const { return last_output; }
protected:
    Tensor last_input;
    Tensor last_output;
};
