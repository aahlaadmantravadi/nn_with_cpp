#include "nn/optimizers/SGD.h"

SGD::SGD(float learning_rate) : Optimizer(learning_rate) {}

void SGD::update(Tensor& weights, const Tensor& grad_weights) {
    for (size_t i = 0; i < weights.getSize(); ++i) {
        weights.getCpuData()[i] -= learning_rate * grad_weights.getCpuData()[i];
    }
}
