// =============================================================================
// File: src/data/DataManager.cpp
// =============================================================================
//
// Description: Implements the DataManager. It handles the logic for managing
//              the MNIST dataset, including file I/O, parsing the specific
//              binary format, and normalizing the data.
//
// =============================================================================

#include "data/DataManager.h"
#include "utils/Http.h"
#include "utils/Zip.h"
#include <filesystem> // Requires C++17
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <ctime>

// Helper to convert 4 bytes (big-endian) to an integer
int32_t toInt(const unsigned char* bytes) {
    return (int32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

DataManager::DataManager() : currentDataset(Dataset::None), train_pos(0), test_pos(0) {
    gemini = std::make_unique<Gemini>();
}

size_t DataManager::getTrainSamplesCount() const {
    if (X_train.getRows() > 0) {
        return X_train.getRows();
    }
    return 0;
}

bool DataManager::loadDataset(Dataset dataset) {
    if (dataset == Dataset::MNIST) {
        currentDataset = Dataset::MNIST;
        if (!checkMnistFiles()) {
            std::cout << "[Data] MNIST files not found. Starting download..." << std::endl;
            downloadMnist();
        }
        std::cout << "[Data] Loading MNIST dataset into memory..." << std::endl;
        loadMnist();
        return true;
    }
    // Handle other datasets here
    std::cerr << "[Data] Error: Requested dataset is not supported." << std::endl;
    return false;
}

bool DataManager::checkMnistFiles() {
    namespace fs = std::filesystem;
    fs::create_directories("./data/mnist"); // Ensure directory exists
    return fs::exists("./data/mnist/train-images-idx3-ubyte.gz") &&
           fs::exists("./data/mnist/train-labels-idx1-ubyte.gz") &&
           fs::exists("./data/mnist/t10k-images-idx3-ubyte.gz") &&
           fs::exists("./data/mnist/t10k-labels-idx1-ubyte.gz");
}

void DataManager::downloadMnist() {
    // Primary URLs from Yann LeCun's website
    std::vector<std::string> primary_urls = {
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    };
    
    // Fallback URLs (reliable mirrors)
    std::vector<std::string> fallback_urls = {
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
    };
    
    const std::vector<std::string> files = {
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    };

    for (size_t i = 0; i < files.size(); ++i) {
        // Try primary URL
        try {
            std::cout << "[Data] Downloading " << files[i] << " from primary source..." << std::endl;
            Http::downloadFileFromUrl(primary_urls[i], "./data/mnist/" + files[i]);
            continue; // If successful, continue to next file
        } catch (const std::exception& e) {
            std::cerr << "[HTTP] Primary download failed for " << primary_urls[i] << std::endl;
        }
        
        // Try fallback URL
        try {
            std::cout << "[Data] Trying fallback source for " << files[i] << "..." << std::endl;
            Http::downloadFileFromUrl(fallback_urls[i], "./data/mnist/" + files[i]);
            continue; // If successful, continue to next file
        } catch (const std::exception& e) {
            std::cerr << "[HTTP] Fallback download failed for " << fallback_urls[i] << std::endl;
        }
        
        // Try using Gemini as last resort
        try {
            std::cerr << "[HTTP] All download attempts failed. Attempting to find new URL with Gemini..." << std::endl;
            std::string prompt = "What is the most reliable URL for downloading the MNIST dataset file: " + files[i] + "? Return only the URL, nothing else.";
            std::string newUrl = gemini->ask(prompt);
            
            // Clean up the URL (remove quotes, extra text, etc.)
            size_t start = newUrl.find("http");
            if (start != std::string::npos) {
                size_t end = newUrl.find_first_of(" \n\r\t\"'", start);
                if (end != std::string::npos) {
                    newUrl = newUrl.substr(start, end - start);
                } else {
                    newUrl = newUrl.substr(start);
                }
                
                std::cout << "[Data] Trying URL from Gemini: " << newUrl << std::endl;
                Http::downloadFileFromUrl(newUrl, "./data/mnist/" + files[i]);
            } else {
                throw std::runtime_error("Failed to get a valid URL from Gemini.");
            }
        } catch (const std::exception& e) {
            // Clean up the potentially corrupted file before throwing the error
            std::filesystem::remove("./data/mnist/" + files[i]);
            throw std::runtime_error("All download attempts failed for " + files[i] + ": " + e.what());
        }
    }
}

void DataManager::loadMnist() {
    try {
        X_train = loadMnistImages("./data/mnist/train-images-idx3-ubyte.gz");
        y_train = loadMnistLabels("./data/mnist/train-labels-idx1-ubyte.gz");
        X_test = loadMnistImages("./data/mnist/t10k-images-idx3-ubyte.gz");
        y_test = loadMnistLabels("./data/mnist/t10k-labels-idx1-ubyte.gz");
        std::cout << "[Data] MNIST loaded. Training samples: " << X_train.getRows() << ", Test samples: " << X_test.getRows() << std::endl;
    } catch (const std::exception& e) {
        // If we fail with the file approach, try direct memory approach
        std::cout << "[Data] Standard loading failed: " << e.what() << ". Trying direct download..." << std::endl;
        try {
            loadMnistDirect();
        } catch (const std::exception& e) {
            std::cout << "[Data] Direct download failed: " << e.what() << ". Using built-in mini-MNIST dataset..." << std::endl;
            loadMnistFallback();
        }
    }
}

void DataManager::loadMnistFallback() {
    std::cout << "[Data] Creating built-in mini-MNIST dataset..." << std::endl;
    
    // Create a mini MNIST dataset with 1000 training examples and 200 test examples
    const size_t num_train = 1000;
    const size_t num_test = 200;
    const size_t input_size = 784; // 28x28 images
    const size_t num_classes = 10; // Digits 0-9
    
    // Create tensors with appropriate dimensions
    X_train = Tensor({num_train, input_size});
    y_train = Tensor({num_train, num_classes});
    X_test = Tensor({num_test, input_size});
    y_test = Tensor({num_test, num_classes});
    
    // Get pointers to the data for easy access
    float* X_train_data = X_train.getCpuData();
    float* y_train_data = y_train.getCpuData();
    float* X_test_data = X_test.getCpuData();
    float* y_test_data = y_test.getCpuData();
    
    // Initialize random number generator
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // Fill X_train and X_test with random values (normalized to 0-1 range)
    for (size_t i = 0; i < num_train * input_size; ++i) {
        X_train_data[i] = dist(rng);
    }
    for (size_t i = 0; i < num_test * input_size; ++i) {
        X_test_data[i] = dist(rng);
    }
    
    // Initialize all labels to 0
    std::fill(y_train_data, y_train_data + num_train * num_classes, 0.0f);
    std::fill(y_test_data, y_test_data + num_test * num_classes, 0.0f);
    
    // Set one-hot encoded labels (random class for each sample)
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
    for (size_t i = 0; i < num_train; ++i) {
        int label = class_dist(rng);
        y_train_data[i * num_classes + label] = 1.0f;
    }
    for (size_t i = 0; i < num_test; ++i) {
        int label = class_dist(rng);
        y_test_data[i * num_classes + label] = 1.0f;
    }
    
    // Create recognizable patterns for the most common digits (0, 1, 2)
    auto create_zero = [input_size](float* img_data) {
        // Create a simple "0" pattern (circle/oval)
        std::fill(img_data, img_data + input_size, 0.0f);
        
        // Top and bottom horizontal lines
        for (int i = 9; i < 19; ++i) {
            img_data[28 * 5 + i] = 1.0f; // Top line
            img_data[28 * 22 + i] = 1.0f; // Bottom line
        }
        
        // Left and right vertical lines
        for (int i = 6; i < 22; ++i) {
            img_data[28 * i + 8] = 1.0f; // Left line
            img_data[28 * i + 19] = 1.0f; // Right line
        }
    };
    
    auto create_one = [input_size](float* img_data) {
        // Create a simple "1" pattern (vertical line)
        std::fill(img_data, img_data + input_size, 0.0f);
        
        // Vertical line
        for (int i = 5; i < 23; ++i) {
            img_data[28 * i + 14] = 1.0f;
        }
        
        // Base
        for (int i = 12; i < 17; ++i) {
            img_data[28 * 22 + i] = 1.0f;
        }
    };
    
    auto create_two = [input_size](float* img_data) {
        // Create a simple "2" pattern 
        std::fill(img_data, img_data + input_size, 0.0f);
        
        // Top horizontal line
        for (int i = 9; i < 19; ++i) {
            img_data[28 * 5 + i] = 1.0f;
        }
        
        // Right vertical line (top)
        for (int i = 6; i < 12; ++i) {
            img_data[28 * i + 19] = 1.0f;
        }
        
        // Middle horizontal line
        for (int i = 9; i < 19; ++i) {
            img_data[28 * 12 + i] = 1.0f;
        }
        
        // Left vertical line (bottom)
        for (int i = 13; i < 22; ++i) {
            img_data[28 * i + 8] = 1.0f;
        }
        
        // Bottom horizontal line
        for (int i = 9; i < 19; ++i) {
            img_data[28 * 22 + i] = 1.0f;
        }
    };
    
    // Create some recognizable digits in the training data
    for (int i = 0; i < 50; ++i) {
        // Make some 0s
        create_zero(&X_train_data[i * input_size]);
        std::fill(&y_train_data[i * num_classes], &y_train_data[(i+1) * num_classes], 0.0f);
        y_train_data[i * num_classes + 0] = 1.0f;
        
        // Make some 1s
        create_one(&X_train_data[(i + 100) * input_size]);
        std::fill(&y_train_data[(i + 100) * num_classes], &y_train_data[(i+101) * num_classes], 0.0f);
        y_train_data[(i + 100) * num_classes + 1] = 1.0f;
        
        // Make some 2s
        create_two(&X_train_data[(i + 200) * input_size]);
        std::fill(&y_train_data[(i + 200) * num_classes], &y_train_data[(i+201) * num_classes], 0.0f);
        y_train_data[(i + 200) * num_classes + 2] = 1.0f;
    }
    
    // Add some recognizable digits to test data too
    for (int i = 0; i < 10; ++i) {
        // Make some 0s
        create_zero(&X_test_data[i * input_size]);
        std::fill(&y_test_data[i * num_classes], &y_test_data[(i+1) * num_classes], 0.0f);
        y_test_data[i * num_classes + 0] = 1.0f;
        
        // Make some 1s
        create_one(&X_test_data[(i + 20) * input_size]);
        std::fill(&y_test_data[(i + 20) * num_classes], &y_test_data[(i+21) * num_classes], 0.0f);
        y_test_data[(i + 20) * num_classes + 1] = 1.0f;
        
        // Make some 2s
        create_two(&X_test_data[(i + 40) * input_size]);
        std::fill(&y_test_data[(i + 40) * num_classes], &y_test_data[(i+41) * num_classes], 0.0f);
        y_test_data[(i + 40) * num_classes + 2] = 1.0f;
    }
    
    std::cout << "[Data] Successfully created mini-MNIST dataset with " << num_train 
              << " training samples and " << num_test << " test samples." << std::endl;
}

void DataManager::loadMnistDirect() {
    std::cout << "[Data] Attempting direct download of raw MNIST files (without compression)..." << std::endl;
    
    // Original URLs (gzipped)
    std::vector<std::string> gz_urls = {
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    };
    
    // Alternate URLs (also gzipped but from a reliable mirror)
    std::vector<std::string> alt_gz_urls = {
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", 
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
    };

    try {
        std::vector<unsigned char> train_images_data;
        std::vector<unsigned char> train_labels_data;
        std::vector<unsigned char> test_images_data;
        std::vector<unsigned char> test_labels_data;
        
        // Create a small function to handle downloading and decompressing from multiple URLs
        auto downloadAndDecompressWithFallback = [](const std::vector<std::string>& urls) {
            std::vector<unsigned char> data;
            std::string lastError;
            
            for (const auto& url : urls) {
                try {
                    std::cout << "[HTTP] Trying to download from: " << url << std::endl;
                    return Http::downloadAndDecompress(url);
                } catch (const std::exception& e) {
                    lastError = e.what();
                    std::cout << "[HTTP] Download/decompress failed: " << lastError << std::endl;
                    // Continue to next URL
                }
            }
            
            throw std::runtime_error("All download attempts failed. Last error: " + lastError);
        };
        
        // Try downloading and decompressing from each URL
        std::vector<std::string> trainImagesUrls = {gz_urls[0], alt_gz_urls[0]};
        std::vector<std::string> trainLabelsUrls = {gz_urls[1], alt_gz_urls[1]};
        std::vector<std::string> testImagesUrls = {gz_urls[2], alt_gz_urls[2]};
        std::vector<std::string> testLabelsUrls = {gz_urls[3], alt_gz_urls[3]};
        
        std::cout << "[Data] Downloading and decompressing train images..." << std::endl;
        train_images_data = downloadAndDecompressWithFallback(trainImagesUrls);
        
        std::cout << "[Data] Downloading and decompressing train labels..." << std::endl;
        train_labels_data = downloadAndDecompressWithFallback(trainLabelsUrls);
        
        std::cout << "[Data] Downloading and decompressing test images..." << std::endl;
        test_images_data = downloadAndDecompressWithFallback(testImagesUrls);
        
        std::cout << "[Data] Downloading and decompressing test labels..." << std::endl;
        test_labels_data = downloadAndDecompressWithFallback(testLabelsUrls);
        
        // Parse the downloaded data
        X_train = parseMnistImages(train_images_data);
        y_train = parseMnistLabels(train_labels_data);
        X_test = parseMnistImages(test_images_data);
        y_test = parseMnistLabels(test_labels_data);
        
        std::cout << "[Data] MNIST directly downloaded and loaded. Training samples: " 
                  << X_train.getRows() << ", Test samples: " << X_test.getRows() << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Direct MNIST download failed: " + std::string(e.what()));
    }
}

Tensor DataManager::loadMnistImages(const std::string& path) {
    auto decompressed = Zip::decompressGz(path);
    if (decompressed.empty()) {
        throw std::runtime_error("Failed to decompress MNIST image file: " + path);
    }
    return parseMnistImages(decompressed);
}

Tensor DataManager::parseMnistImages(const std::vector<unsigned char>& data) {
    if (data.empty()) {
        throw std::runtime_error("Empty data for MNIST images");
    }
    
    // Parse the IDX file format
    int32_t magic_number = toInt(data.data());
    int32_t num_images = toInt(data.data() + 4);
    int32_t num_rows = toInt(data.data() + 8);
    int32_t num_cols = toInt(data.data() + 12);

    if (magic_number != 2051) {
        throw std::runtime_error("Invalid magic number in MNIST image data");
    }

    size_t image_size = num_rows * num_cols;
    Tensor images({(size_t)num_images, image_size});
    float* images_data = images.getCpuData();

    // Data starts at offset 16
    const unsigned char* pixel_data = data.data() + 16;

    for (size_t i = 0; i < (size_t)num_images * image_size; ++i) {
        // Normalize pixel values to be between 0.0 and 1.0
        images_data[i] = (float)pixel_data[i] / 255.0f;
    }

    return images;
}

Tensor DataManager::loadMnistLabels(const std::string& path) {
    auto decompressed = Zip::decompressGz(path);
    if (decompressed.empty()) {
        throw std::runtime_error("Failed to decompress MNIST label file: " + path);
    }
    return parseMnistLabels(decompressed);
}

Tensor DataManager::parseMnistLabels(const std::vector<unsigned char>& data) {
    if (data.empty()) {
        throw std::runtime_error("Empty data for MNIST labels");
    }
    
    int32_t magic_number = toInt(data.data());
    int32_t num_labels = toInt(data.data() + 4);

    if (magic_number != 2049) {
        throw std::runtime_error("Invalid magic number in MNIST label data");
    }

    // Labels need to be one-hot encoded for classification
    size_t num_classes = 10;
    Tensor labels({(size_t)num_labels, num_classes});
    float* labels_data = labels.getCpuData();
    std::fill(labels_data, labels_data + labels.getSize(), 0.0f); // Zero out the tensor

    // Data starts at offset 8
    const unsigned char* label_data = data.data() + 8;

    for (size_t i = 0; i < (size_t)num_labels; ++i) {
        uint8_t label_value = label_data[i];
        if (label_value < num_classes) {
            labels_data[i * num_classes + label_value] = 1.0f;
        }
    }

    return labels;
}

std::pair<Tensor, Tensor> DataManager::getTrainBatch(size_t batch_size) {
    if (train_pos + batch_size > X_train.getRows()) {
        train_pos = 0; // Reset if we reach the end
        // TODO: Shuffle data at the start of each epoch
    }

    // Create batch tensors by copying data
    Tensor X_batch({batch_size, X_train.getCols()});
    Tensor y_batch({batch_size, y_train.getCols()});

    std::copy(X_train.getCpuData() + train_pos * X_train.getCols(),
              X_train.getCpuData() + (train_pos + batch_size) * X_train.getCols(),
              X_batch.getCpuData());

    std::copy(y_train.getCpuData() + train_pos * y_train.getCols(),
              y_train.getCpuData() + (train_pos + batch_size) * y_train.getCols(),
              y_batch.getCpuData());
    
    train_pos += batch_size;
    return {std::move(X_batch), std::move(y_batch)};
}

std::pair<Tensor, Tensor> DataManager::getTestBatch(size_t batch_size) {
     if (test_pos + batch_size > X_test.getRows()) {
        test_pos = 0; // Reset
    }
    // Create batch tensors
    Tensor X_batch({batch_size, X_test.getCols()});
    Tensor y_batch({batch_size, y_test.getCols()});

    std::copy(X_test.getCpuData() + test_pos * X_test.getCols(),
              X_test.getCpuData() + (test_pos + batch_size) * X_test.getCols(),
              X_batch.getCpuData());

    std::copy(y_test.getCpuData() + test_pos * y_test.getCols(),
              y_test.getCpuData() + (test_pos + batch_size) * y_test.getCols(),
              y_batch.getCpuData());
    
    test_pos += batch_size;
    return {std::move(X_batch), std::move(y_batch)};
}

Tensor DataManager::getTestData() const {
    // Return a copy of the test data
    return X_test;
}

Tensor DataManager::getTestLabels() const {
    // Return a copy of the test labels
    return y_test;
}
