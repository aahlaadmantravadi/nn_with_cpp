// =============================================================================
// File: src/data/DataManager.h
// =============================================================================
//
// Description: Declares the DataManager class, responsible for downloading,
//              loading, and providing batches of data from datasets like MNIST.
//
// =============================================================================

#pragma once



#include "nn/Tensor.h"
#include "nn/nn_types.h"
#include "utils/Gemini.h"



#include <memory>
#include <string>
#include <utility>
#include <vector>



// Forward declaration
struct DatasetInfo;



class DataManager
{
public:
    DataManager();

    // Main method to load a dataset. It will download it if not present.
    [[nodiscard]] bool loadDataset(Dataset dataset);

    // New AI-driven dataset loading method
    [[nodiscard]] bool loadDatasetFromInfo(const struct DatasetInfo &dataset_info);

    // Get a batch of training data.
    [[nodiscard]] std::pair<Tensor, Tensor> getTrainBatch(size_t batch_size);

    // Get a batch of testing data.
    [[nodiscard]] std::pair<Tensor, Tensor> getTestBatch(size_t batch_size);

    // FIX: Add a public getter for the training data size.
    [[nodiscard]] size_t getTrainSamplesCount() const;

    // Get all test data for evaluation
    [[nodiscard]] Tensor getTestData() const;

    // Get all test labels for evaluation
    [[nodiscard]] Tensor getTestLabels() const;

    // Get dataset statistics for architecture inference
    struct DatasetStats
    {
        size_t num_samples;
        size_t input_size;
        size_t num_classes;
        std::vector<int> input_shape;
        std::string modality;
    };
    [[nodiscard]] DatasetStats getDatasetStats() const;

private:
    // --- MNIST Specific Methods ---
    [[nodiscard]] bool checkMnistFiles();
    void downloadMnist();
    void loadMnist();
    void loadMnistDirect(); // Direct download and load without intermediate files
    void loadMnistFallback(); // Fallback to built-in mini-MNIST dataset
    [[nodiscard]] Tensor loadMnistImages(const std::string &path);
    [[nodiscard]] Tensor loadMnistLabels(const std::string &path);
    [[nodiscard]] Tensor parseMnistImages(const std::vector<unsigned char> &data);
    [[nodiscard]] Tensor parseMnistLabels(const std::vector<unsigned char> &data);

    // --- CIFAR-10 Specific Methods ---
    void loadCifar10();
    void parseCifar10FromTar(const std::vector<unsigned char> &tar_data, std::vector<unsigned char> &train_data, std::vector<unsigned char> &test_data);
    void parseCifarBinaryBatch(const std::vector<unsigned char> &batch, Tensor &X_out, Tensor &y_out, size_t offset_images, size_t offset_labels);

    // --- Generic Dataset Loading Methods ---
    void loadGenericDataset(const struct DatasetInfo &dataset_info);
    void loadImageFolderDataset(const std::string &extracted_path, const struct DatasetInfo &dataset_info);
    void loadCsvDataset(const std::string &file_path, const struct DatasetInfo &dataset_info);
    [[nodiscard]] std::vector<unsigned char> downloadAndExtract(const std::string &url, const std::string &format);
    void extractArchive(const std::vector<unsigned char> &archive_data, const std::string &format, const std::string &extract_path);

    // --- Member Variables ---
    Dataset currentDataset;

    Tensor X_train, y_train; // Training data and labels
    Tensor X_test, y_test;   // Testing data and labels

    size_t train_pos; // Current position in the training set for batching
    size_t test_pos;  // Current position in the testing set
    std::unique_ptr<Gemini> gemini;
    std::vector<size_t> train_indices;

    // Dataset metadata for AI inference
    DatasetStats current_stats;
};
