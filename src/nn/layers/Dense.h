#pragma once

#include "nn/layers/Layer.h"
#include "nn/optimizers/Optimizer.h"
#include "nn/nn_types.h"

class Dense : public Layer {
public:
    Dense(size_t input_size, size_t output_size);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update(Optimizer& optimizer) override;
    
    // Set backend type for this layer
    void setBackendType(Backend type) { backendType = type; }
    Backend getBackendType() const { return backendType; }

    Tensor weights;
    Tensor biases;

private:
    Tensor grad_weights;
    Tensor grad_biases;
    Backend backendType = Backend::CPU; // Default to CPU
};
