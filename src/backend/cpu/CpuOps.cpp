#include "backend/cpu/CpuOps.h"
#include <stdexcept>

void CpuOps::matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.getCols() != b.getRows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    if (c.getRows() != a.getRows() || c.getCols() != b.getCols()) {
        throw std::invalid_argument("Output tensor C has incorrect dimensions.");
    }

    for (size_t i = 0; i < a.getRows(); ++i) {
        for (size_t j = 0; j < b.getCols(); ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < a.getCols(); ++k) {
                sum += a.get(i, k) * b.get(k, j);
            }
            c.set(i, j, sum);
        }
    }
}

void CpuOps::add(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.getSize() != b.getSize() || a.getSize() != c.getSize()) {
        throw std::invalid_argument("Tensors must have the same size for addition.");
    }
    for (size_t i = 0; i < a.getSize(); ++i) {
        c.getCpuData()[i] = a.getCpuData()[i] + b.getCpuData()[i];
    }
}

void CpuOps::relu(const Tensor& a, Tensor& b) {
    if (a.getSize() != b.getSize()) {
        throw std::invalid_argument("Tensors must have the same size for ReLU.");
    }
    for (size_t i = 0; i < a.getSize(); ++i) {
        b.getCpuData()[i] = std::max(0.0f, a.getCpuData()[i]);
    }
}
