#pragma once

#include "nn/Tensor.h"
#include "nn/layers/Layer.h"
#include "nn/Loss.h"
#include "nn/optimizers/Optimizer.h"
#include "nn/nn_types.h"
#include <vector>
#include <memory>

class Model {
public:
    Model();
    void add(std::unique_ptr<Layer> layer);
    void compile(std::unique_ptr<Loss> loss_func, std::unique_ptr<Optimizer> optimizer);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad);
    float train_step(const Tensor& X_batch, const Tensor& y_batch);
    
    // Evaluate the model on test data and return loss and accuracy
    std::pair<float, float> evaluate(const Tensor& X_test, const Tensor& y_test);
    
    const std::vector<std::unique_ptr<Layer>>& getLayers() const { return layers; }
    
    // Set backend for all layers that support it
    void setBackend(Backend type);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Loss> loss_func;
    std::unique_ptr<Optimizer> optimizer;
    Backend backendType = Backend::CPU;
};
