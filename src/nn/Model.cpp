#include "nn/Model.h"
#include "nn/layers/Layer.h"
#include "nn/layers/Dense.h"
#include "nn/Loss.h"
#include "nn/optimizers/Optimizer.h"

Model::Model() {}

void Model::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

void Model::compile(std::unique_ptr<Loss> loss_func, std::unique_ptr<Optimizer> optimizer) {
    this->loss_func = std::move(loss_func);
    this->optimizer = std::move(optimizer);
}

Tensor Model::forward(const Tensor& input) {
    Tensor current_output = input;
    for (auto& layer : layers) {
        current_output = layer->forward(current_output);
    }
    return current_output;
}

void Model::backward(const Tensor& grad) {
    Tensor current_grad = grad;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        current_grad = (*it)->backward(current_grad);
    }
}

float Model::train_step(const Tensor& X_batch, const Tensor& y_batch) {
    Tensor y_pred = this->forward(X_batch);
    float loss = loss_func->forward(y_pred, y_batch);
    Tensor grad = loss_func->backward(y_pred, y_batch);
    this->backward(grad);
    for (auto& layer : layers) {
        layer->update(*optimizer);
    }
    return loss;
}

std::pair<float, float> Model::evaluate(const Tensor& X_test, const Tensor& y_test) {
    // Forward pass on test data
    Tensor y_pred = this->forward(X_test);
    
    // Calculate loss
    float loss = loss_func->forward(y_pred, y_test);
    
    // Calculate accuracy
    size_t correct_predictions = 0;
    size_t total_predictions = X_test.getRows();
    
    // Different handling based on y_test shape (one-hot vs class indices)
    if (y_test.getShape() == y_pred.getShape()) {
        // One-hot encoded targets
        for (size_t i = 0; i < y_test.getRows(); ++i) {
            // Find predicted class (highest probability)
            size_t pred_class = 0;
            float max_prob = y_pred.get(i, 0);
            for (size_t j = 1; j < y_pred.getCols(); ++j) {
                if (y_pred.get(i, j) > max_prob) {
                    max_prob = y_pred.get(i, j);
                    pred_class = j;
                }
            }
            
            // Find true class (1 in one-hot encoding)
            size_t true_class = 0;
            for (size_t j = 0; j < y_test.getCols(); ++j) {
                if (y_test.get(i, j) > 0.5f) {  // Threshold for one-hot
                    true_class = j;
                    break;
                }
            }
            
            if (pred_class == true_class) {
                correct_predictions++;
            }
        }
    } else if (y_test.getCols() == 1) {
        // Class indices
        for (size_t i = 0; i < y_test.getRows(); ++i) {
            // Find predicted class (highest probability)
            size_t pred_class = 0;
            float max_prob = y_pred.get(i, 0);
            for (size_t j = 1; j < y_pred.getCols(); ++j) {
                if (y_pred.get(i, j) > max_prob) {
                    max_prob = y_pred.get(i, j);
                    pred_class = j;
                }
            }
            
            // True class is directly given
            size_t true_class = static_cast<size_t>(y_test.get(i, 0));
            
            if (pred_class == true_class) {
                correct_predictions++;
            }
        }
    }
    
    float accuracy = static_cast<float>(correct_predictions) / static_cast<float>(total_predictions);
    return std::make_pair(loss, accuracy);
}

void Model::setBackend(Backend type) {
    backendType = type;
    
    // Propagate to all layers that support backend selection
    for (auto& layer : layers) {
        if (auto* dense_layer = dynamic_cast<Dense*>(layer.get())) {
            dense_layer->setBackendType(type);
        }
        // Add other layer types that support backend selection here
    }
}
