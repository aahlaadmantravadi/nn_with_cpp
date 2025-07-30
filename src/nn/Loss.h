#pragma once

#include "nn/Tensor.h"

class Loss {
public:
    virtual ~Loss() = default;
    virtual float forward(const Tensor& y_pred, const Tensor& y_true) = 0;
    virtual Tensor backward(const Tensor& y_pred, const Tensor& y_true) = 0;
};

class MeanSquaredError : public Loss {
public:
    float forward(const Tensor& y_pred, const Tensor& y_true) override;
    Tensor backward(const Tensor& y_pred, const Tensor& y_true) override;
};

class CrossEntropyLoss : public Loss {
public:
    float forward(const Tensor& y_pred, const Tensor& y_true) override;
    Tensor backward(const Tensor& y_pred, const Tensor& y_true) override;
};
