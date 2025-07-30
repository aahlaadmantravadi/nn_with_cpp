#include "nn/layers/Dense.h"
#include "backend/cpu/CpuOps.h"
#include "backend/gpu/GpuOps.cuh"
#include "nn/nn_types.h"
#include <iostream>

Dense::Dense(size_t input_size, size_t output_size)
    : weights({input_size, output_size}), biases({1, output_size}),
      grad_weights({input_size, output_size}), grad_biases({1, output_size}) {
    weights.initializeRandom();
    biases.initializeRandom();
}

Tensor Dense::forward(const Tensor& input) {
    this->last_input = input;
    Tensor output({input.getRows(), weights.getCols()});
    
    // Choose the appropriate backend for matrix multiplication
    try {
        if (backendType == Backend::GPU) {
            // Ensure tensors are on GPU
            if (!input.isOnGpu()) {
                const_cast<Tensor&>(input).toGpu();
            }
            
            if (!weights.isOnGpu()) {
                weights.toGpu();
            }
            
            if (!biases.isOnGpu()) {
                biases.toGpu();
            }
            
            // Allocate output on GPU
            output.allocateGpu();
            
            // Try GPU operations
            GpuOps::matmul(input, weights, output);
            
            // Move result back to CPU for further operations
            output.toCpu();
            
            // Apply biases
            for (size_t i = 0; i < output.getRows(); ++i) {
                for (size_t j = 0; j < output.getCols(); ++j) {
                    output.set(i, j, output.get(i, j) + biases.get(0, j));
                }
            }
        } else {
            // CPU operations - fallback by default
            CpuOps::matmul(input, weights, output);
            for (size_t i = 0; i < output.getRows(); ++i) {
                for (size_t j = 0; j < output.getCols(); ++j) {
                    output.set(i, j, output.get(i, j) + biases.get(0, j));
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass: " << e.what() 
                  << ". Falling back to CPU implementation." << std::endl;
        
        // If GPU operations fail, fall back to CPU
        backendType = Backend::CPU;
        CpuOps::matmul(input, weights, output);
        for (size_t i = 0; i < output.getRows(); ++i) {
            for (size_t j = 0; j < output.getCols(); ++j) {
                output.set(i, j, output.get(i, j) + biases.get(0, j));
            }
        }
    }
    
    this->last_output = output;
    return output;
}

Tensor Dense::backward(const Tensor& grad_output) {
    Tensor last_input_T = last_input.transpose();
    Tensor weights_T = weights.transpose();
    Tensor grad_input({grad_output.getRows(), weights_T.getCols()});
    
    try {
        if (backendType == Backend::GPU) {
            // Ensure tensors are on GPU
            if (!last_input_T.isOnGpu()) {
                last_input_T.toGpu();
            }
            
            if (!grad_output.isOnGpu()) {
                const_cast<Tensor&>(grad_output).toGpu();
            }
            
            if (!weights_T.isOnGpu()) {
                weights_T.toGpu();
            }
            
            // Allocate output tensors on GPU
            grad_weights.allocateGpu();
            grad_input.allocateGpu();
            
            // GPU implementation of backward pass
            GpuOps::matmul(last_input_T, grad_output, grad_weights);
            
            // Move results back to CPU
            grad_weights.toCpu();
            
            // Sum gradients for biases across the batch (CPU for now)
            for(size_t j = 0; j < grad_output.getCols(); ++j) {
                float sum = 0;
                for(size_t i = 0; i < grad_output.getRows(); ++i) {
                    sum += grad_output.get(i, j);
                }
                grad_biases.set(0, j, sum / grad_output.getRows()); // Average gradient
            }
            
            // Do matrix multiplication for input gradients
            GpuOps::matmul(grad_output, weights_T, grad_input);
            
            // Move result back to CPU
            grad_input.toCpu();
        } else {
            // CPU implementation (default)
            CpuOps::matmul(last_input_T, grad_output, grad_weights);
            
            // Sum gradients for biases across the batch
            for(size_t j = 0; j < grad_output.getCols(); ++j) {
                float sum = 0;
                for(size_t i = 0; i < grad_output.getRows(); ++i) {
                    sum += grad_output.get(i, j);
                }
                grad_biases.set(0, j, sum / grad_output.getRows()); // Average gradient
            }
            
            CpuOps::matmul(grad_output, weights_T, grad_input);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during backward pass: " << e.what() 
                  << ". Falling back to CPU implementation." << std::endl;
        
        // If GPU operations fail, fall back to CPU
        backendType = Backend::CPU;
        
        CpuOps::matmul(last_input_T, grad_output, grad_weights);
        
        // Sum gradients for biases across the batch
        for(size_t j = 0; j < grad_output.getCols(); ++j) {
            float sum = 0;
            for(size_t i = 0; i < grad_output.getRows(); ++i) {
                sum += grad_output.get(i, j);
            }
            grad_biases.set(0, j, sum / grad_output.getRows()); // Average gradient
        }
        
        CpuOps::matmul(grad_output, weights_T, grad_input);
    }
    
    return grad_input;
}

void Dense::update(Optimizer& optimizer) {
    optimizer.update(weights, grad_weights);
    optimizer.update(biases, grad_biases);
}
