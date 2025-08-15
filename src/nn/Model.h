#pragma once



#include "nn/Loss.h"
#include "nn/Tensor.h"
#include "nn/layers/Layer.h"
#include "nn/nn_types.h"
#include "nn/optimizers/Optimizer.h"



#include <memory>
#include <vector>



class Model
{
public:
    Model();
    void add(std::unique_ptr<Layer> layer);

    void compile(std::unique_ptr<Loss> loss_func, std::unique_ptr<Optimizer> optimizer);

    [[nodiscard]] Tensor forward(const Tensor &input);

    void backward(const Tensor &grad);

    [[nodiscard]] float train_step(const Tensor &X_batch, const Tensor &y_batch);

    // Evaluate the model on test data and return loss and accuracy
    [[nodiscard]] std::pair<float, float> evaluate(const Tensor &X_test, const Tensor &y_test);

    [[nodiscard]] const std::vector<std::unique_ptr<Layer>> &getLayers() const { return layers; }

    // Set backend for all layers that support it
    void setBackend(Backend type);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Loss> loss_func;
    std::unique_ptr<Optimizer> optimizer;
    Backend backendType = Backend::CPU;
};
