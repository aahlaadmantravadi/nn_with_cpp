// =============================================================================
// File: src/nn/Tensor.cpp
// =============================================================================
//
// Description: Implements the Tensor class. This includes memory management
//              for both CPU and GPU, data transfers, and initialization logic.
//              CUDA runtime API calls are used for GPU operations.
//
// =============================================================================

#include "nn/Tensor.h"
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <iostream>

// Only include CUDA headers if CUDA is available, as determined by CMake
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// --- Constructors and Destructor ---

Tensor::Tensor() : totalSize(0), cpu_data(nullptr), gpu_data(nullptr) {}

Tensor::Tensor(const std::vector<size_t>& shape)
    : shape(shape), totalSize(0), cpu_data(nullptr), gpu_data(nullptr) {
    calculateSize();
    allocateCpu();
}

Tensor::~Tensor() {
    freeCpu();
    freeGpu();
}

void Tensor::calculateSize() {
    if (shape.empty()) {
        totalSize = 0;
        return;
    }
    // FIX: Initialize accumulate with a size_t to prevent conversion warnings.
    totalSize = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
}

// --- Copy and Move Semantics ---

Tensor::Tensor(const Tensor& other) : shape(other.shape), totalSize(other.totalSize), cpu_data(nullptr), gpu_data(nullptr) {
    if (other.cpu_data) {
        allocateCpu();
        std::copy(other.cpu_data, other.cpu_data + totalSize, cpu_data);
    }
    if (other.gpu_data) {
        #ifdef __CUDACC__
            allocateGpu();
            cudaMemcpy(gpu_data, other.gpu_data, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
        #else
            // Handle case where CUDA is not compiled but trying to copy a GPU tensor
            // This should ideally not happen in a well-structured program.
            std::cerr << "Warning: Attempting to copy a GPU tensor in a non-CUDA build." << std::endl;
        #endif
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }
    freeCpu();
    freeGpu();

    shape = other.shape;
    totalSize = other.totalSize;
    
    if (other.cpu_data) {
        allocateCpu();
        std::copy(other.cpu_data, other.cpu_data + totalSize, cpu_data);
    }
    if (other.gpu_data) {
        #ifdef __CUDACC__
            allocateGpu();
            cudaMemcpy(gpu_data, other.gpu_data, totalSize * sizeof(float), cudaMemcpyDeviceToDevice);
        #else
             std::cerr << "Warning: Attempting to assign a GPU tensor in a non-CUDA build." << std::endl;
        #endif
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape(std::move(other.shape)), totalSize(other.totalSize),
      cpu_data(other.cpu_data), gpu_data(other.gpu_data) {
    // The moved-from object should be in a valid but empty state
    other.cpu_data = nullptr;
    other.gpu_data = nullptr;
    other.totalSize = 0;
    other.shape.clear();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    freeCpu();
    freeGpu();

    shape = std::move(other.shape);
    totalSize = other.totalSize;
    cpu_data = other.cpu_data;
    gpu_data = other.gpu_data;

    other.cpu_data = nullptr;
    other.gpu_data = nullptr;
    other.totalSize = 0;
    other.shape.clear();
    
    return *this;
}


// --- Public Methods ---

void Tensor::initializeRandom() {
    if (!cpu_data) {
        allocateCpu();
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    // Use a standard normal distribution for initialization (common practice)
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (size_t i = 0; i < totalSize; ++i) {
        cpu_data[i] = dist(gen);
    }
}

void Tensor::reshape(const std::vector<size_t>& newShape) {
    // FIX: Initialize accumulate with a size_t to prevent conversion warnings.
    size_t newSize = std::accumulate(newShape.begin(), newShape.end(), (size_t)1, std::multiplies<size_t>());
    if (newSize != totalSize) {
        throw std::invalid_argument("Cannot reshape tensor: total number of elements must be preserved.");
    }
    shape = newShape;
}

// --- Memory Management ---

void Tensor::allocateCpu() {
    if (cpu_data) return; // Already allocated
    if (totalSize == 0) return;
    cpu_data = new float[totalSize];
}

void Tensor::allocateGpu() {
    #ifdef __CUDACC__
        if (gpu_data) return; // Already allocated
        if (totalSize == 0) return;
        cudaError_t err = cudaMalloc(&gpu_data, totalSize * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to allocate GPU memory: ") + cudaGetErrorString(err));
        }
    #else
        throw std::runtime_error("Cannot allocate GPU memory: not compiled with CUDA support.");
    #endif
}

void Tensor::toGpu() {
    #ifdef __CUDACC__
        if (!cpu_data) {
            throw std::runtime_error("Cannot move to GPU: CPU data does not exist.");
        }
        if (!gpu_data) {
            allocateGpu();
        }
        cudaError_t err = cudaMemcpy(gpu_data, cpu_data, totalSize * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to copy data to GPU: ") + cudaGetErrorString(err));
        }
    #else
        throw std::runtime_error("Cannot move to GPU: not compiled with CUDA support.");
    #endif
}

void Tensor::toCpu() {
    #ifdef __CUDACC__
        if (!gpu_data) {
            throw std::runtime_error("Cannot move to CPU: GPU data does not exist.");
        }
        if (!cpu_data) {
            allocateCpu();
        }
        cudaError_t err = cudaMemcpy(cpu_data, gpu_data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to copy data to CPU: ") + cudaGetErrorString(err));
        }
    #else
        throw std::runtime_error("Cannot move to CPU: not compiled with CUDA support.");
    #endif
}

void Tensor::freeCpu() {
    if (cpu_data) {
        delete[] cpu_data;
        cpu_data = nullptr;
    }
}

void Tensor::freeGpu() {
    #ifdef __CUDACC__
        if (gpu_data) {
            cudaFree(gpu_data);
            gpu_data = nullptr;
        }
    #else
        // No-op if not compiled with CUDA
    #endif
}

// --- Element Access ---

float Tensor::get(size_t row, size_t col) const {
    if (row >= getRows() || col >= getCols()) {
        throw std::out_of_range("Tensor access out of range.");
    }
    return cpu_data[row * getCols() + col];
}

void Tensor::set(size_t row, size_t col, float value) {
    if (row >= getRows() || col >= getCols()) {
        throw std::out_of_range("Tensor access out of range.");
    }
    cpu_data[row * getCols() + col] = value;
}

// --- Tensor Operations ---

Tensor Tensor::getRow(size_t row) const {
    if (row >= getRows()) {
        throw std::out_of_range("Row access out of range.");
    }
    Tensor row_vec({1, getCols()});
    std::copy(cpu_data + row * getCols(), cpu_data + (row + 1) * getCols(), row_vec.cpu_data);
    return row_vec;
}

Tensor Tensor::diag(const Tensor& v) {
    if (v.getRows() != 1) {
        throw std::invalid_argument("diag expects a row vector.");
    }
    size_t n = v.getCols();
    Tensor result({n, n});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result.set(i, j, (i == j) ? v.get(0, i) : 0.0f);
        }
    }
    return result;
}

Tensor Tensor::outer(const Tensor& v1, const Tensor& v2) {
    if (v1.getRows() != 1 || v2.getRows() != 1) {
        throw std::invalid_argument("outer expects row vectors.");
    }
    size_t n1 = v1.getCols();
    size_t n2 = v2.getCols();
    Tensor result({n1, n2});
    for (size_t i = 0; i < n1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            result.set(i, j, v1.get(0, i) * v2.get(0, j));
        }
    }
    return result;
}

Tensor Tensor::multiply(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Element-wise multiplication requires tensors of the same shape.");
    }
    Tensor result(shape);
    for (size_t i = 0; i < totalSize; ++i) {
        result.cpu_data[i] = cpu_data[i] * other.cpu_data[i];
    }
    return result;
}

Tensor Tensor::transpose() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("Transpose is only supported for 2D tensors.");
    }
    Tensor result({getCols(), getRows()});
    for (size_t i = 0; i < getRows(); ++i) {
        for (size_t j = 0; j < getCols(); ++j) {
            result.set(j, i, get(i, j));
        }
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Element-wise subtraction requires tensors of the same shape.");
    }
    Tensor result(shape);
    for (size_t i = 0; i < totalSize; ++i) {
        result.cpu_data[i] = cpu_data[i] - other.cpu_data[i];
    }
    return result;
}
