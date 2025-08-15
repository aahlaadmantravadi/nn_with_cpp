#pragma once



#include "nn/layers/Layer.h"



class Softmax final : public Layer
{
public:
    Softmax();
    [[nodiscard]] Tensor forward(const Tensor & input) override;
    [[nodiscard]] Tensor backward(const Tensor & grad_output) override;

private:
    Tensor last_output;
};
