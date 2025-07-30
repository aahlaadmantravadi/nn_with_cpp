// =============================================================================
// File: src/nn/Loss.cpp
// =============================================================================
//
// Description: Implements the Mean Squared Error (MSE) loss function.
//              Loss = mean((y_pred - y_true)^2)
//              Gradient = 2 * (y_pred - y_true) / n_elements
//
// =============================================================================

#include "nn/Loss.h"
#include <stdexcept>
#include <cmath>

// --- MeanSquaredError Implementation ---

float MeanSquaredError::forward(const Tensor& y_pred, const Tensor& y_true) {
    if (y_pred.getSize() != y_true.getSize()) {
        throw std::invalid_argument("Prediction and true label tensors must have the same size for MSE.");
    }

    const float* pred_data = y_pred.getCpuData();
    const float* true_data = y_true.getCpuData();
    size_t size = y_pred.getSize();
    float sum_sq_err = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        float diff = pred_data[i] - true_data[i];
        sum_sq_err += diff * diff;
    }

    return sum_sq_err / size;
}

Tensor MeanSquaredError::backward(const Tensor& y_pred, const Tensor& y_true) {
    if (y_pred.getSize() != y_true.getSize()) {
        throw std::invalid_argument("Prediction and true label tensors must have the same size for MSE backward pass.");
    }

    Tensor gradient(y_pred.getShape());
    float* grad_data = gradient.getCpuData();
    const float* pred_data = y_pred.getCpuData();
    const float* true_data = y_true.getCpuData();
    size_t size = y_pred.getSize();

    for (size_t i = 0; i < size; ++i) {
        grad_data[i] = 2.0f * (pred_data[i] - true_data[i]) / size;
    }

    return gradient;
}
