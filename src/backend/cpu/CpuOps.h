#pragma once

#include "nn/Tensor.h"

class CpuOps {
public:
    static void matmul(const Tensor& a, const Tensor& b, Tensor& c);
    static void add(const Tensor& a, const Tensor& b, Tensor& c);
    static void relu(const Tensor& a, Tensor& b);
};
