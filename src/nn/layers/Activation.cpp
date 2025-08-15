#include "nn/layers/Activation.h"



#include <cmath>
#include <stdexcept>



Activation::Activation(ActivationType type) : type{type}
{
    switch (type)
    {
        case ActivationType::ReLU:
            activation_func = [](float x) { return std::max(0.0f, x); };
            derivative_func = [](float x) { return (x > 0) ? 1.0f : 0.0f; };
            break;
        case ActivationType::Sigmoid:
            activation_func = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
            derivative_func = [this](float x)
            {
                float s = activation_func(x);
                return (s * (1.0f - s));
            };
            break;
        default:
            throw std::invalid_argument("Unsupported activation type provided to Activation layer. Softmax is a separate layer.");
    }
}



Tensor Activation::forward(const Tensor &input)
{
    this->last_input = input;
    Tensor output{input.getShape()};
    for (size_t i = 0; i < input.getSize(); i++)
    {
        output.getCpuData()[i] = activation_func(input.getCpuData()[i]);
    }
    return output;
}



Tensor Activation::backward(const Tensor &grad_output)
{
    Tensor grad_input{grad_output.getShape()};
    for (size_t i = 0; i < grad_output.getSize(); i++)
    {
        grad_input.getCpuData()[i] = grad_output.getCpuData()[i] * derivative_func(this->last_input.getCpuData()[i]);
    }
    return grad_input;
}
