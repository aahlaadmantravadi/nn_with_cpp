#include "nn/Loss.h"



#include <cmath>
#include <limits>
#include <stdexcept>


// --- MeanSquaredError Implementation ---

float MeanSquaredError::forward(const Tensor &y_pred, const Tensor &y_true)
{
    if(y_pred.getShape() != y_true.getShape())
    {
        throw std::invalid_argument("Shapes of y_pred and y_true must be the same.");
    }
    float sum = 0.0f;
    for(size_t i = 0; i < y_pred.getSize(); i++)
    {
        float diff = y_pred.getCpuData()[i] - y_true.getCpuData()[i];
        sum += diff * diff;
    }
    return (sum / y_pred.getSize());
}


Tensor MeanSquaredError::backward(const Tensor &y_pred, const Tensor &y_true)
{
    if(y_pred.getShape() != y_true.getShape())
    {
        throw std::invalid_argument("Shapes of y_pred and y_true must be the same.");
    }
    Tensor grad(y_pred.getShape());
    for(size_t i = 0; i < y_pred.getSize(); i++)
    {
        grad.getCpuData()[i] = 2 * (y_pred.getCpuData()[i] - y_true.getCpuData()[i]) / y_pred.getSize();
    }
    return grad;
}


// --- CrossEntropyLoss Implementation ---

float CrossEntropyLoss::forward(const Tensor &y_pred, const Tensor &y_true)
{
    // Check if shapes are compatible
    if(y_pred.getRows() != y_true.getRows())
    {
        throw std::invalid_argument("Number of rows in y_pred and y_true must match.");
    }
    
    // Handle different target formats (one-hot or class indices)
    float loss = 0.0f;
    
    if(y_pred.getShape() == y_true.getShape())
    {
        // One-hot encoded targets (same shape as predictions)
        for(size_t i = 0; i < y_pred.getRows(); i++)
        {
            for(size_t j = 0; j < y_pred.getCols(); j++)
            {
                if(y_true.get(i, j) > 0.0f)
                {
                    loss -= std::log(std::max(y_pred.get(i, j), 1e-9f));
                }
            }
        }
    }
    else if(y_true.getCols() == 1)
    {
        // Class indices format (one column with class indices)
        for(size_t i = 0; i < y_pred.getRows(); i++)
        {
            int class_idx = static_cast<int>(y_true.get(i, 0));
            if((class_idx >= 0) && (class_idx < static_cast<int>(y_pred.getCols())))
            {
                loss -= std::log(std::max(y_pred.get(i, class_idx), 1e-9f));
            }
        }
    }
    else
    {
        throw std::invalid_argument("Incompatible shapes for cross entropy loss.");
    }
    
    return (loss / y_pred.getRows());
}


Tensor CrossEntropyLoss::backward(const Tensor &y_pred, const Tensor &y_true)
{
    // Initialize gradient with same shape as predictions
    Tensor grad(y_pred.getShape());
    
    if(y_pred.getShape() == y_true.getShape())
    {
        // One-hot encoded targets
        for(size_t i = 0; i < y_pred.getRows(); i++)
        {
            for(size_t j = 0; j < y_pred.getCols(); j++)
            {
                // For cross-entropy with softmax, gradient is (prediction - target)
                grad.set(i, j, (y_pred.get(i, j) - y_true.get(i, j)) / y_pred.getRows());
            }
        }
    }
    else if(y_true.getCols() == 1)
    {
        // Class indices format
        for(size_t i = 0; i < y_pred.getRows(); i++)
        {
            // Copy predictions as initial gradient (for softmax this works well)
            for(size_t j = 0; j < y_pred.getCols(); j++)
            {
                grad.set(i, j, y_pred.get(i, j) / y_pred.getRows());
            }
            
            // Subtract 1 from the true class gradient
            int class_idx = static_cast<int>(y_true.get(i, 0));
            if((class_idx >= 0) && (class_idx < static_cast<int>(y_pred.getCols())))
            {
                grad.set(i, class_idx, (y_pred.get(i, class_idx) - 1.0f) / y_pred.getRows());
            }
        }
    }
    else
    {
        throw std::invalid_argument("Incompatible shapes for cross entropy loss gradient.");
    }
    
    return grad;
}
