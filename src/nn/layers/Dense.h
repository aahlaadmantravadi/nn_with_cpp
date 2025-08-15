#pragma once



#include "nn/layers/Layer.h"
#include "nn/nn_types.h"
#include "nn/optimizers/Optimizer.h"



class Dense final : public Layer
{
public:
    Dense(size_t input_size, size_t output_size);
    [[nodiscard]] Tensor forward(const Tensor & input) override;
    [[nodiscard]] Tensor backward(const Tensor & grad_output) override;
    void update(Optimizer & optimizer) override;

    void setBackendType(Backend type) { backendType = type; }
    [[nodiscard]] Backend getBackendType() const { return backendType; }

    Tensor weights;
    Tensor biases;

private:
    Tensor grad_weights;
    Tensor grad_biases;
    Backend backendType = Backend::CPU;
};
