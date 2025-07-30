// =============================================================================
// File: src/backend/gpu/GpuOps.cuh
// =============================================================================
//
// Description: Declares the GPU-based operations (CUDA). This header is
//              processed by NVCC and exposes C++-callable wrapper functions
//              that launch the CUDA kernels defined in GpuOps.cu.
//
// =============================================================================

#ifndef GPU_OPS_CUH
#define GPU_OPS_CUH

class Tensor; // Forward declaration

class GpuOps {
public:
    // Wrapper function to launch the matrix multiplication kernel: C = A * B
    static void matmul(const Tensor& A, const Tensor& B, Tensor& C);
    
    // Wrapper function to launch the element-wise addition kernel: C = A + B
    static void add(const Tensor& A, const Tensor& B, Tensor& C);

    // Wrapper function to launch the ReLU activation kernel: A = max(0, A)
    static void relu(Tensor& A);

    // ... other CUDA operation wrappers will go here
};

#endif // GPU_OPS_CUH
