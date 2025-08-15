#pragma once


#include "nn/Tensor.h"


class Loss
{
public:
    virtual ~Loss() = default;
    [[nodiscard]] virtual float forward(const Tensor &y_pred, const Tensor &y_true) = 0;
    [[nodiscard]] virtual Tensor backward(const Tensor &y_pred, const Tensor &y_true) = 0;
};


class MeanSquaredError final : public Loss
{
public:
    [[nodiscard]] float forward(const Tensor &y_pred, const Tensor &y_true) override;
    [[nodiscard]] Tensor backward(const Tensor &y_pred, const Tensor &y_true) override;
};


class CrossEntropyLoss final : public Loss
{
public:
    [[nodiscard]] float forward(const Tensor &y_pred, const Tensor &y_true) override;
    [[nodiscard]] Tensor backward(const Tensor &y_pred, const Tensor &y_true) override;
};
