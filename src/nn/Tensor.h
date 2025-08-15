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

#pragma once



// --- Standard Includes ---
#include <cstddef>
#include <vector>



class Tensor
{
public:
    // --- Constructors and Destructor ---

    Tensor();



    explicit Tensor(const std::vector<size_t>& shape);



    ~Tensor();



    // --- Copy and Move Semantics ---

    Tensor(const Tensor& other);

    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;

    Tensor& operator=(Tensor&& other) noexcept;



    // --- Public Methods ---

    void initializeRandom();



    void reshape(const std::vector<size_t>& newShape);



    // --- Memory Management ---

    void allocateCpu();

    void allocateGpu();

    void toGpu();

    void toCpu();

    void freeCpu();

    void freeGpu();



    // --- Getters ---

    [[nodiscard]] const std::vector<size_t>& getShape() const noexcept { return shape; }

    [[nodiscard]] size_t getRows() const noexcept { return shape.empty() ? 0 : shape[0]; }

    [[nodiscard]] size_t getCols() const noexcept { return shape.size() < 2 ? 0 : shape[1]; }

    [[nodiscard]] size_t getSize() const noexcept { return totalSize; }

    [[nodiscard]] float *getCpuData() noexcept { return cpu_data; }

    [[nodiscard]] const float *getCpuData() const noexcept { return cpu_data; }

    [[nodiscard]] float *getGpuData() noexcept { return gpu_data; }

    [[nodiscard]] const float *getGpuData() const noexcept { return gpu_data; }

    [[nodiscard]] bool isOnGpu() const noexcept { return gpu_data != nullptr; }



    // --- Element Access ---

    float get(size_t row, size_t col) const;

    void set(size_t row, size_t col, float value);



    // --- Tensor Operations ---

    [[nodiscard]] Tensor getRow(size_t row) const;

    [[nodiscard]] static Tensor diag(const Tensor& v);

    [[nodiscard]] static Tensor outer(const Tensor& v1, const Tensor& v2);

    [[nodiscard]] Tensor multiply(const Tensor& other) const;

    [[nodiscard]] Tensor transpose() const;

    [[nodiscard]] Tensor operator-(const Tensor& other) const;



private:
    // --- Private Helper Methods ---

    void calculateSize();



    // --- Member Variables ---

    std::vector<size_t> shape;
    size_t totalSize;
    float *cpu_data;
    float *gpu_data;
};

