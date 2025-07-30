// =============================================================================
// File: src/backend/gpu/GpuOps.cu
// =============================================================================
//
// Description: Implements the CUDA kernels for high-performance GPU
//              computation. This includes a tiled matrix multiplication kernel
//              for efficiency and simpler element-wise kernels.
//
// =============================================================================

#include "backend/gpu/GpuOps.cuh"
#include "nn/Tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

// --- CUDA Kernel Definitions ---

// Define tile width for shared memory optimization in matmul
#define TILE_WIDTH 32

// CUDA Kernel for Tiled Matrix Multiplication: C = A * B
__global__ void matmulKernel(const float* A, const float* B, float* C, int m, int k, int n) {
    // Shared memory for tiles of A and B
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    // Thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column for this thread's element in C
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    // Loop over the tiles of A and B required to compute the C element
    for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load tile of A into shared memory
        if (row < m && (t * TILE_WIDTH + tx) < k) {
            sA[ty][tx] = A[row * k + (t * TILE_WIDTH + tx)];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        if (col < n && (t * TILE_WIDTH + ty) < k) {
            sB[ty][tx] = B[(t * TILE_WIDTH + ty) * n + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Multiply the two tiles and accumulate the result
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += sA[ty][i] * sB[i][tx];
        }
        
        // Synchronize to make sure all threads are done with the current tile
        __syncthreads();
    }

    // Write the final result to global memory
    if (row < m && col < n) {
        C[row * n + col] = Cvalue;
    }
}

// CUDA Kernel for element-wise addition
__global__ void addKernel(const float* A, const float* B, float* C, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA Kernel for ReLU activation
__global__ void reluKernel(float* A, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmaxf(0.0f, A[idx]);
    }
}


// --- C++ Wrapper Functions (Implementation of GpuOps) ---

void GpuOps::matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    size_t m = A.getRows();
    size_t k = A.getCols();
    size_t n = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication.");
    }

    // Get raw GPU data pointers
    const float* a_data = A.getGpuData();
    const float* b_data = B.getGpuData();
    float* c_data = C.getGpuData();

    if (!a_data || !b_data || !c_data) {
        throw std::runtime_error("matmul: One or more tensors are not on the GPU.");
    }

    // Define grid and block dimensions for the kernel launch
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the kernel
    matmulKernel<<<numBlocks, threadsPerBlock>>>(a_data, b_data, c_data, m, k, n);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA matmul kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void GpuOps::add(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.getSize() != B.getSize() || A.getSize() != C.getSize()) {
        throw std::invalid_argument("Tensors must have the same size for addition.");
    }

    const float* a_data = A.getGpuData();
    const float* b_data = B.getGpuData();
    float* c_data = C.getGpuData();

    if (!a_data || !b_data || !c_data) {
        throw std::runtime_error("add: One or more tensors are not on the GPU.");
    }
    
    size_t size = A.getSize();
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    addKernel<<<numBlocks, threadsPerBlock>>>(a_data, b_data, c_data, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA add kernel launch failed: ") + cudaGetErrorString(err));
    }
}

void GpuOps::relu(Tensor& A) {
    float* a_data = A.getGpuData();
    if (!a_data) {
        throw std::runtime_error("relu: Tensor is not on the GPU.");
    }

    size_t size = A.getSize();
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    reluKernel<<<numBlocks, threadsPerBlock>>>(a_data, size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA relu kernel launch failed: ") + cudaGetErrorString(err));
    }
}
