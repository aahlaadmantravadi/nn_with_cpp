// =============================================================================
// File: src/nn/Tensor.h
// =============================================================================
//
// Description: Declares the Tensor class, the fundamental data structure for
//              the neural network. A Tensor represents a multi-dimensional
//              matrix and is responsible for managing its own memory on both
//              the CPU (host) and optionally the GPU (device).
//
// =============================================================================

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef> // for size_t
#include <memory>  // for std::shared_ptr
#include <functional> // for std::function

class Tensor {
public:
    // --- Constructors and Destructor ---

    // Default constructor
    Tensor();

    // Constructor to create a Tensor with a specific shape.
    // Data is not initialized.
    explicit Tensor(const std::vector<size_t>& shape);

    // Destructor to free allocated memory.
    ~Tensor();

    // --- Copy and Move Semantics ---
    // Tensors can be large, so we must handle copying and moving carefully.
    
    // Copy constructor
    Tensor(const Tensor& other);
    // Copy assignment operator
    Tensor& operator=(const Tensor& other);
    // Move constructor
    Tensor(Tensor&& other) noexcept;
    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept;

    // --- Public Methods ---

    // Initializes the tensor with random values (e.g., for weights).
    void initializeRandom();

    // Reshapes the tensor. The total number of elements must remain the same.
    void reshape(const std::vector<size_t>& newShape);

    // --- Memory Management ---

    // Allocates memory on the CPU.
    void allocateCpu();
    // Allocates memory on the GPU.
    void allocateGpu();

    // Moves data from CPU memory to GPU memory.
    void toGpu();
    // Moves data from GPU memory to CPU memory.
    void toCpu();

    // Frees CPU memory.
    void freeCpu();
    // Frees GPU memory.
    void freeGpu();

    // --- Getters ---

    const std::vector<size_t>& getShape() const { return shape; }
    size_t getRows() const { return shape.empty() ? 0 : shape[0]; }
    size_t getCols() const { return shape.size() < 2 ? 0 : shape[1]; }
    size_t getSize() const { return totalSize; }
    
    // Get a raw pointer to the CPU data.
    float* getCpuData() { return cpu_data; }
    const float* getCpuData() const { return cpu_data; }

    // Get a raw pointer to the GPU data.
    float* getGpuData() { return gpu_data; }
    const float* getGpuData() const { return gpu_data; }

    bool isOnGpu() const { return gpu_data != nullptr; }

    // --- Element Access ---
    float get(size_t row, size_t col) const;
    void set(size_t row, size_t col, float value);

    // --- Tensor Operations ---
    Tensor getRow(size_t row) const;
    static Tensor diag(const Tensor& v);
    static Tensor outer(const Tensor& v1, const Tensor& v2);
    Tensor multiply(const Tensor& other) const;
    Tensor transpose() const;
    Tensor operator-(const Tensor& other) const;

private:
    // --- Private Helper Methods ---
    void calculateSize();

    // --- Member Variables ---
    std::vector<size_t> shape; // e.g., {batch_size, channels, height, width}
    size_t totalSize;          // Total number of elements in the tensor

    float* cpu_data;           // Pointer to data on the host (CPU)
    float* gpu_data;           // Pointer to data on the device (GPU)
};

#endif // TENSOR_H
