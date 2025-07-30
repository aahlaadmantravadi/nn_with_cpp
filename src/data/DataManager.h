// =============================================================================
// File: src/data/DataManager.h
// =============================================================================
//
// Description: Declares the DataManager class, responsible for downloading,
//              loading, and providing batches of data from datasets like MNIST.
//
// =============================================================================

#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H

#include "nn/Tensor.h"
#include "utils/Gemini.h"
#include "nn/nn_types.h"
#include <string>
#include <vector>
#include <utility> // for std::pair
#include <memory>

class DataManager {
public:
    DataManager();

    // Main method to load a dataset. It will download it if not present.
    bool loadDataset(Dataset dataset);

    // Get a batch of training data.
    std::pair<Tensor, Tensor> getTrainBatch(size_t batch_size);

    // Get a batch of testing data.
    std::pair<Tensor, Tensor> getTestBatch(size_t batch_size);

    // FIX: Add a public getter for the training data size.
    size_t getTrainSamplesCount() const;
    
    // Get all test data for evaluation
    Tensor getTestData() const;
    
    // Get all test labels for evaluation
    Tensor getTestLabels() const;

private:
    // --- MNIST Specific Methods ---
    bool checkMnistFiles();
    void downloadMnist();
    void loadMnist();
    void loadMnistDirect(); // Direct download and load without intermediate files
    void loadMnistFallback(); // Fallback to built-in mini-MNIST dataset
    Tensor loadMnistImages(const std::string& path);
    Tensor loadMnistLabels(const std::string& path);
    Tensor parseMnistImages(const std::vector<unsigned char>& data);
    Tensor parseMnistLabels(const std::vector<unsigned char>& data);

    // --- Member Variables ---
    Dataset currentDataset;
    
    Tensor X_train, y_train; // Training data and labels
    Tensor X_test, y_test;   // Testing data and labels

    size_t train_pos; // Current position in the training set for batching
    size_t test_pos;  // Current position in the testing set
    std::unique_ptr<Gemini> gemini;
};

#endif // DATA_MANAGER_H
