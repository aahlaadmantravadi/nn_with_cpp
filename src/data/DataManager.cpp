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



#include "nlp/Parser.h"
#include "utils/Http.h"
#include "utils/Zip.h"



#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>



// Helper to convert 4 bytes (big-endian) to an integer
int32_t toInt(const unsigned char *bytes)
{
    return (int32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}



DataManager::DataManager() : currentDataset{Dataset::None}, train_pos{0}, test_pos{0}
{
    gemini = std::make_unique<Gemini>();
    current_stats.num_samples = 0;
    current_stats.input_size = 0;
    current_stats.num_classes = 0;
    current_stats.input_shape.clear();
    current_stats.modality.clear();
}



size_t DataManager::getTrainSamplesCount() const
{
    if (X_train.getRows() > 0) { return X_train.getRows(); }
    return 0;
}



bool DataManager::loadDataset(Dataset dataset)
{
    if (dataset == Dataset::MNIST)
    {
        currentDataset = Dataset::MNIST;
        if (!checkMnistFiles())
        {
            std::cout << "[Data] MNIST files not found. Starting download..." << '\n';
            downloadMnist();
        }
        std::cout << "[Data] Loading MNIST dataset into memory..." << '\n';
        loadMnist();
        return true;
    }
    if ((dataset == Dataset::CIFAR10) || (dataset == Dataset::CIFAR10_CATS_DOGS))
    {
        currentDataset = dataset;
        std::cout << "[Data] Loading CIFAR-10 dataset into memory..." << '\n';
        try
        {
            loadCifar10();
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Data] Error loading CIFAR-10: " << e.what() << '\n';
            return false;
        }
    }
    std::cerr << "[Data] Error: Requested dataset is not supported." << '\n';
    return false;
}



bool DataManager::loadDatasetFromInfo(const DatasetInfo &dataset_info)
{
    std::cout << "[Data] Loading AI-resolved dataset: " << dataset_info.name << '\n';
    std::cout << "[Data] Modality: " << dataset_info.modality << ", Format: " << dataset_info.format << '\n';

    if (dataset_info.name.empty() || (dataset_info.name == "custom_needed"))
    {
        std::cerr << "[Data] Error: No suitable dataset found for this task." << '\n';
        return false;
    }
    try
    {
        loadGenericDataset(dataset_info);

        // Update dataset stats for architecture inference
        current_stats.num_samples = X_train.getRows();
        current_stats.input_size = X_train.getCols();
        current_stats.num_classes = y_train.getCols();
        current_stats.modality = dataset_info.modality;
        current_stats.input_shape = dataset_info.input_shape;

        std::cout << "[Data] Successfully loaded " << dataset_info.name << ". Training samples: " << current_stats.num_samples << ", Input size: " << current_stats.input_size << ", Classes: " << current_stats.num_classes << '\n';

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Data] Error loading dataset: " << e.what() << '\n';
        return false;
    }
}



bool DataManager::checkMnistFiles()
{
    namespace fs = std::filesystem;
    fs::create_directories("./data/mnist"); // Ensure directory exists
    return fs::exists("./data/mnist/train-images-idx3-ubyte.gz") &&
           fs::exists("./data/mnist/train-labels-idx1-ubyte.gz") &&
           fs::exists("./data/mnist/t10k-images-idx3-ubyte.gz") &&
           fs::exists("./data/mnist/t10k-labels-idx1-ubyte.gz");
}



void DataManager::downloadMnist()
{
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

    for (size_t i = 0; i < files.size(); i++)
    {
        // Try primary URL
        try
        {
            std::cout << "[Data] Downloading " << files[i] << " from primary source..." << '\n';
            Http::downloadFileFromUrl(primary_urls[i], "./data/mnist/" + files[i]);
            continue; // If successful, continue to next file
        }
        catch (const std::exception &)
        {
            std::cerr << "[HTTP] Primary download failed for " << primary_urls[i] << '\n';
        }

        // Try fallback URL
        try
        {
            std::cout << "[Data] Trying fallback source for " << files[i] << "..." << '\n';
            Http::downloadFileFromUrl(fallback_urls[i], "./data/mnist/" + files[i]);
            continue; // If successful, continue to next file
        }
        catch (const std::exception &)
        {
            std::cerr << "[HTTP] Fallback download failed for " << fallback_urls[i] << '\n';
        }

        // Try using Gemini as last resort
        try
        {
            std::cerr << "[HTTP] All download attempts failed. Attempting to find new URL with Gemini..." << '\n';
            std::string prompt = "What is the most reliable URL for downloading the MNIST dataset file: " + files[i] + "? Return only the URL, nothing else.";
            std::string newUrl = gemini->ask(prompt);

            // Clean up the URL (remove quotes, extra text, etc.)
            size_t start = newUrl.find("http");
            if (start != std::string::npos)
            {
                size_t end = newUrl.find_first_of(" \n\r\t\"'", start);
                if (end != std::string::npos)
                {
                    newUrl = newUrl.substr(start, end - start);
                }
                else
                {
                    newUrl = newUrl.substr(start);
                }

                std::cout << "[Data] Trying URL from Gemini: " << newUrl << '\n';
                Http::downloadFileFromUrl(newUrl, "./data/mnist/" + files[i]);
            }
            else
            {
                throw std::runtime_error("Failed to get a valid URL from Gemini.");
            }
        }
        catch (const std::exception &e)
        {
            // Clean up the potentially corrupted file before throwing the error
            std::filesystem::remove("./data/mnist/" + files[i]);
            throw std::runtime_error("All download attempts failed for " + files[i] + ": " + e.what());
        }
    }
}



void DataManager::loadMnist()
{
    try
    {
        X_train = loadMnistImages("./data/mnist/train-images-idx3-ubyte.gz");
        y_train = loadMnistLabels("./data/mnist/train-labels-idx1-ubyte.gz");
        X_test = loadMnistImages("./data/mnist/t10k-images-idx3-ubyte.gz");
        y_test = loadMnistLabels("./data/mnist/t10k-labels-idx1-ubyte.gz");
        std::cout << "[Data] MNIST loaded. Training samples: " << X_train.getRows() << ", Test samples: " << X_test.getRows() << '\n';
    }
    catch (const std::exception &e)
    {
        // If we fail with the file approach, try direct memory approach
        std::cout << "[Data] Standard loading failed: " << e.what() << ". Trying direct download..." << '\n';
        try
        {
            loadMnistDirect();
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("[Data] Direct download and load failed: " + std::string(e.what()));
        }
    }
}



void DataManager::loadCifar10()
{
    namespace fs = std::filesystem;
    fs::create_directories("./data/cifar10");
    const std::string url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    std::cout << "[Data] Downloading CIFAR-10 tarball..." << '\n';
    auto tar_gz_bytes = Http::downloadRawFile(url);
    // Write to temp file to reuse gzip helper
    const std::string gz_path = "./data/cifar10/cifar-10-binary.tar.gz";
    {
        std::ofstream ofs(gz_path, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(tar_gz_bytes.data()), static_cast<std::streamsize>(tar_gz_bytes.size()));
    }
    std::cout << "[Data] Decompressing CIFAR-10 tar.gz..." << '\n';
    auto tar_bytes = Zip::decompressGz(gz_path);

    // Minimal TAR extraction: iterate 512-byte headers
    struct TarEntry { std::string name; std::vector<unsigned char> data; };
    std::vector<TarEntry> entries;
    size_t pos = 0;
    auto read_octal = [](const unsigned char *p, size_t n) -> size_t
    {
        size_t val = 0;
        for (size_t i = 0; i < n && p[i]; i++)
        {
            if ((p[i] >= '0') && (p[i] <= '7')) { val = (val << 3) + (p[i] - '0'); }
        }
        return val;
    };
    while ((pos + 512) <= tar_bytes.size())
    {
        const unsigned char *hdr = tar_bytes.data() + pos;
        bool all_zero = true;
        for (size_t i = 0; i < 512; i++) { if (hdr[i] != 0) { all_zero = false; break; } }
        if (all_zero) break; // end of archive
        std::string name(reinterpret_cast<const char *>(hdr), 100);
        name = name.c_str(); // trim nulls
        size_t size = read_octal(hdr + 124, 12);
        pos += 512;
        if (size > 0)
        {
            if ((pos + size) > tar_bytes.size()) throw std::runtime_error("CIFAR-10 tar truncated");
            std::vector<unsigned char> filedata(tar_bytes.begin() + pos, tar_bytes.begin() + pos + size);
            entries.push_back({name, std::move(filedata)});
            // advance to next 512 boundary
            size_t pad = ((size + 511) / 512) * 512;
            pos += pad;
        }
    }

    // Collect CIFAR-10 binary batches
    std::vector<std::vector<unsigned char>> train_batches;
    std::vector<unsigned char> test_batch;
    for (auto &e : entries)
    {
        if ((e.name.find("data_batch_") != std::string::npos) && (e.name.find(".bin") != std::string::npos))
        {
            train_batches.push_back(std::move(e.data));
        }
        else if (e.name.find("test_batch.bin") != std::string::npos)
        {
            test_batch = std::move(e.data);
        }
    }
    if ((train_batches.size() < 5) || test_batch.empty())
    {
        throw std::runtime_error("CIFAR-10 tar missing expected batches");
    }

    const size_t num_train = 50000;
    const size_t num_test = 10000;
    const size_t input_size = 3072; // 32x32x3
    const size_t num_classes = 10;
    X_train = Tensor{{num_train, input_size}};
    y_train = Tensor{{num_train, num_classes}};
    X_test = Tensor{{num_test, input_size}};
    y_test = Tensor{{num_test, num_classes}};
    std::fill(y_train.getCpuData(), y_train.getCpuData() + y_train.getSize(), 0.0f);
    std::fill(y_test.getCpuData(), y_test.getCpuData() + y_test.getSize(), 0.0f);

    auto parse_batch = [&](const std::vector<unsigned char> &batch, float *x_ptr, float *y_ptr, size_t start_index, size_t rows)
    {
        const size_t record_size = 1 + input_size;
        size_t num_records = batch.size() / record_size;
        for (size_t i = 0; i < rows && i < num_records; i++)
        {
            size_t off = i * record_size;
            unsigned char label = batch[off];
            // Normalize pixel values to 0..1
            for (size_t j = 0; j < input_size; j++)
            {
                x_ptr[(start_index + i) * input_size + j] = static_cast<float>(batch[off + 1 + j]) / 255.0f;
            }
            if (label < num_classes)
            {
                y_ptr[(start_index + i) * num_classes + label] = 1.0f;
            }
        }
    };

    float *xtr = X_train.getCpuData();
    float *ytr = y_train.getCpuData();
    size_t idx = 0;
    for (size_t b = 0; b < 5; b++)
    {
        parse_batch(train_batches[b], xtr, ytr, idx, 10000);
        idx += 10000;
    }
    parse_batch(test_batch, X_test.getCpuData(), y_test.getCpuData(), 0, 10000);

    std::cout << "[Data] CIFAR-10 loaded. Training samples: " << X_train.getRows() << ", Test samples: " << X_test.getRows() << '\n';
}



void DataManager::loadMnistFallback()
{
    std::cout << "[Data] Creating built-in mini-MNIST dataset..." << '\n';

    // Create a mini MNIST dataset with 1000 training examples and 200 test examples
    const size_t num_train = 1000;
    const size_t num_test = 200;
    const size_t input_size = 784; // 28x28 images
    const size_t num_classes = 10; // Digits 0-9

    // Create tensors with appropriate dimensions
    X_train = Tensor{{num_train, input_size}};
    y_train = Tensor{{num_train, num_classes}};
    X_test = Tensor{{num_test, input_size}};
    y_test = Tensor{{num_test, num_classes}};

    // Get pointers to the data for easy access
    float *X_train_data = X_train.getCpuData();
    float *y_train_data = y_train.getCpuData();
    float *X_test_data = X_test.getCpuData();
    float *y_test_data = y_test.getCpuData();

    // Initialize random number generator
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<float> dist{0.0f, 1.0f};

    // Fill X_train and X_test with random values (normalized to 0-1 range)
    for (size_t i = 0; i < num_train * input_size; i++)
    {
        X_train_data[i] = dist(rng);
    }
    for (size_t i = 0; i < num_test * input_size; i++)
    {
        X_test_data[i] = dist(rng);
    }

    // Initialize all labels to 0
    std::fill(y_train_data, y_train_data + num_train * num_classes, 0.0f);
    std::fill(y_test_data, y_test_data + num_test * num_classes, 0.0f);

    // Set one-hot encoded labels (random class for each sample)
    std::uniform_int_distribution<int> class_dist{0, static_cast<int>(num_classes - 1)};
    for (size_t i = 0; i < num_train; i++)
    {
        int label = class_dist(rng);
        y_train_data[i * num_classes + label] = 1.0f;
    }
    for (size_t i = 0; i < num_test; i++)
    {
        int label = class_dist(rng);
        y_test_data[i * num_classes + label] = 1.0f;
    }

    // Create recognizable patterns for the most common digits (0, 1, 2)
    auto create_zero = [input_size](float *img_data)
    {
        // Create a simple "0" pattern (circle/oval)
        std::fill(img_data, img_data + input_size, 0.0f);

        // Top and bottom horizontal lines
        for (int i = 9; i < 19; i++)
        {
            img_data[28 * 5 + i] = 1.0f; // Top line
            img_data[28 * 22 + i] = 1.0f; // Bottom line
        }

        // Left and right vertical lines
        for (int i = 6; i < 22; i++)
        {
            img_data[28 * i + 8] = 1.0f; // Left line
            img_data[28 * i + 19] = 1.0f; // Right line
        }
    };

    auto create_one = [input_size](float *img_data)
    {
        // Create a simple "1" pattern (vertical line)
        std::fill(img_data, img_data + input_size, 0.0f);

        // Vertical line
        for (int i = 5; i < 23; i++)
        {
            img_data[28 * i + 14] = 1.0f;
        }

        // Base
        for (int i = 12; i < 17; i++)
        {
            img_data[28 * 22 + i] = 1.0f;
        }
    };

    auto create_two = [input_size](float *img_data)
    {
        // Create a simple "2" pattern 
        std::fill(img_data, img_data + input_size, 0.0f);

        // Top horizontal line
        for (int i = 9; i < 19; i++)
        {
            img_data[28 * 5 + i] = 1.0f;
        }

        // Right vertical line (top)
        for (int i = 6; i < 12; i++)
        {
            img_data[28 * i + 19] = 1.0f;
        }

        // Middle horizontal line
        for (int i = 9; i < 19; i++)
        {
            img_data[28 * 12 + i] = 1.0f;
        }

        // Left vertical line (bottom)
        for (int i = 13; i < 22; i++)
        {
            img_data[28 * i + 8] = 1.0f;
        }

        // Bottom horizontal line
        for (int i = 9; i < 19; i++)
        {
            img_data[28 * 22 + i] = 1.0f;
        }
    };

    // Create some recognizable digits in the training data
    for (int i = 0; i < 50; i++)
    {
        // Make some 0s
        create_zero(&X_train_data[i * input_size]);
        std::fill(&y_train_data[i * num_classes], &y_train_data[(i + 1) * num_classes], 0.0f);
        y_train_data[i * num_classes + 0] = 1.0f;

        // Make some 1s
        create_one(&X_train_data[(i + 100) * input_size]);
        std::fill(&y_train_data[(i + 100) * num_classes], &y_train_data[(i + 101) * num_classes], 0.0f);
        y_train_data[(i + 100) * num_classes + 1] = 1.0f;

        // Make some 2s
        create_two(&X_train_data[(i + 200) * input_size]);
        std::fill(&y_train_data[(i + 200) * num_classes], &y_train_data[(i + 201) * num_classes], 0.0f);
        y_train_data[(i + 200) * num_classes + 2] = 1.0f;
    }

    // Add some recognizable digits to test data too
    for (int i = 0; i < 10; i++)
    {
        // Make some 0s
        create_zero(&X_test_data[i * input_size]);
        std::fill(&y_test_data[i * num_classes], &y_test_data[(i + 1) * num_classes], 0.0f);
        y_test_data[i * num_classes + 0] = 1.0f;

        // Make some 1s
        create_one(&X_test_data[(i + 20) * input_size]);
        std::fill(&y_test_data[(i + 20) * num_classes], &y_test_data[(i + 21) * num_classes], 0.0f);
        y_test_data[(i + 20) * num_classes + 1] = 1.0f;

        // Make some 2s
        create_two(&X_test_data[(i + 40) * input_size]);
        std::fill(&y_test_data[(i + 40) * num_classes], &y_test_data[(i + 41) * num_classes], 0.0f);
        y_test_data[(i + 40) * num_classes + 2] = 1.0f;
    }

    std::cout << "[Data] Successfully created mini-MNIST dataset with " << num_train << " training samples and " << num_test << " test samples." << '\n';
}



void DataManager::loadMnistDirect()
{
    std::cout << "[Data] Attempting direct download and decompression of MNIST files..." << '\n';

    // The primary and fallback URLs are now handled inside Http::downloadAndDecompress
    const std::string train_images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
    const std::string train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
    const std::string test_images_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
    const std::string test_labels_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";

    try
    {
        std::cout << "[Data] Downloading and decompressing train images..." << '\n';
        auto train_images_data = Http::downloadAndDecompress(train_images_url);

        std::cout << "[Data] Downloading and decompressing train labels..." << '\n';
        auto train_labels_data = Http::downloadAndDecompress(train_labels_url);

        std::cout << "[Data] Downloading and decompressing test images..." << '\n';
        auto test_images_data = Http::downloadAndDecompress(test_images_url);

        std::cout << "[Data] Downloading and decompressing test labels..." << '\n';
        auto test_labels_data = Http::downloadAndDecompress(test_labels_url);

        // Parse the downloaded data
        X_train = parseMnistImages(train_images_data);
        y_train = parseMnistLabels(train_labels_data);
        X_test = parseMnistImages(test_images_data);
        y_test = parseMnistLabels(test_labels_data);

        std::cout << "[Data] MNIST directly downloaded and loaded. Training samples: " << X_train.getRows() << ", Test samples: " << X_test.getRows() << '\n';
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Direct MNIST download failed: " + std::string(e.what()));
    }
}



Tensor DataManager::loadMnistImages(const std::string &path)
{
    auto decompressed = Zip::decompressGz(path);
    if (decompressed.empty())
    {
        throw std::runtime_error("Failed to decompress MNIST image file: " + path);
    }
    return parseMnistImages(decompressed);
}



Tensor DataManager::parseMnistImages(const std::vector<unsigned char> &data)
{
    if (data.empty())
    {
        throw std::runtime_error("Empty data for MNIST images");
    }

    // Parse the IDX file format
    int32_t magic_number = toInt(data.data());
    int32_t num_images = toInt(data.data() + 4);
    int32_t num_rows = toInt(data.data() + 8);
    int32_t num_cols = toInt(data.data() + 12);

    if (magic_number != 2051)
    {
        throw std::runtime_error("Invalid magic number in MNIST image data");
    }

    size_t image_size = num_rows * num_cols;
    Tensor images{{(size_t)num_images, image_size}};
    float *images_data = images.getCpuData();

    // Data starts at offset 16
    const unsigned char *pixel_data = data.data() + 16;

    for (size_t i = 0; i < (size_t)num_images * image_size; i++)
    {
        // Normalize pixel values to be between 0.0 and 1.0
        images_data[i] = static_cast<float>(pixel_data[i]) / 255.0f;
    }

    return images;
}



Tensor DataManager::loadMnistLabels(const std::string &path)
{
    auto decompressed = Zip::decompressGz(path);
    if (decompressed.empty())
    {
        throw std::runtime_error("Failed to decompress MNIST label file: " + path);
    }
    return parseMnistLabels(decompressed);
}



Tensor DataManager::parseMnistLabels(const std::vector<unsigned char> &data)
{
    if (data.empty())
    {
        throw std::runtime_error("Empty data for MNIST labels");
    }

    int32_t magic_number = toInt(data.data());
    int32_t num_labels = toInt(data.data() + 4);

    if (magic_number != 2049)
    {
        throw std::runtime_error("Invalid magic number in MNIST label data");
    }

    // Labels need to be one-hot encoded for classification
    size_t num_classes = 10;
    Tensor labels{{(size_t)num_labels, num_classes}};
    float *labels_data = labels.getCpuData();
    std::fill(labels_data, labels_data + labels.getSize(), 0.0f); // Zero out the tensor

    // Data starts at offset 8
    const unsigned char *label_data = data.data() + 8;

    for (size_t i = 0; i < (size_t)num_labels; i++)
    {
        uint8_t label_value = label_data[i];
        if (label_value < num_classes)
        {
            labels_data[i * num_classes + label_value] = 1.0f;
        }
    }

    return labels;
}



std::pair<Tensor, Tensor> DataManager::getTrainBatch(size_t batch_size)
{
    if ((train_pos + batch_size) > X_train.getRows())
    {
        // New epoch: shuffle indices and reset
        train_pos = 0;
        if (train_indices.size() != X_train.getRows())
        {
            train_indices.resize(X_train.getRows());
            std::iota(train_indices.begin(), train_indices.end(), 0);
        }
        static std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        std::shuffle(train_indices.begin(), train_indices.end(), rng);
    }

    // Create batch tensors by copying data
    Tensor X_batch{{batch_size, X_train.getCols()}};
    Tensor y_batch{{batch_size, y_train.getCols()}};

    // If indices are set, gather; else copy contiguous
    if (!train_indices.empty())
    {
        for (size_t i = 0; i < batch_size; i++)
        {
            size_t idx = train_indices[train_pos + i];
            std::copy(X_train.getCpuData() + idx * X_train.getCols(), X_train.getCpuData() + (idx + 1) * X_train.getCols(), X_batch.getCpuData() + i * X_train.getCols());
            std::copy(y_train.getCpuData() + idx * y_train.getCols(), y_train.getCpuData() + (idx + 1) * y_train.getCols(), y_batch.getCpuData() + i * y_train.getCols());
        }
    }
    else
    {
        std::copy(X_train.getCpuData() + train_pos * X_train.getCols(), X_train.getCpuData() + (train_pos + batch_size) * X_train.getCols(), X_batch.getCpuData());
        std::copy(y_train.getCpuData() + train_pos * y_train.getCols(), y_train.getCpuData() + (train_pos + batch_size) * y_train.getCols(), y_batch.getCpuData());
    }

    train_pos += batch_size;
    return {std::move(X_batch), std::move(y_batch)};
}



std::pair<Tensor, Tensor> DataManager::getTestBatch(size_t batch_size)
{
    if ((test_pos + batch_size) > X_test.getRows()) { test_pos = 0; }
    // Create batch tensors
    Tensor X_batch{{batch_size, X_test.getCols()}};
    Tensor y_batch{{batch_size, y_test.getCols()}};

    std::copy(X_test.getCpuData() + test_pos * X_test.getCols(), X_test.getCpuData() + (test_pos + batch_size) * X_test.getCols(), X_batch.getCpuData());
    std::copy(y_test.getCpuData() + test_pos * y_test.getCols(), y_test.getCpuData() + (test_pos + batch_size) * y_test.getCols(), y_batch.getCpuData());

    test_pos += batch_size;
    return {std::move(X_batch), std::move(y_batch)};
}



Tensor DataManager::getTestData() const
{
    // Return a copy of the test data
    return X_test;
}



Tensor DataManager::getTestLabels() const
{
    // Return a copy of the test labels
    return y_test;
}



DataManager::DatasetStats DataManager::getDatasetStats() const
{
    DatasetStats stats = current_stats;
    // Derive stats from loaded tensors if missing
    if (stats.num_samples == 0 && (X_train.getRows() > 0))
    {
        stats.num_samples = X_train.getRows();
    }
    if (stats.input_size == 0 && (X_train.getCols() > 0))
    {
        stats.input_size = X_train.getCols();
    }
    if (stats.num_classes == 0 && (y_train.getCols() > 0))
    {
        stats.num_classes = y_train.getCols();
    }
    if (stats.modality.empty())
    {
        if ((currentDataset == Dataset::MNIST) || (currentDataset == Dataset::CIFAR10) || (currentDataset == Dataset::CIFAR10_CATS_DOGS))
        {
            stats.modality = "image";
        }
        else
        {
            stats.modality = "tabular";
        }
    }
    if (stats.input_shape.empty())
    {
        if (currentDataset == Dataset::MNIST)
        {
            stats.input_shape = {28, 28, 1};
        }
        else if ((currentDataset == Dataset::CIFAR10) || (currentDataset == Dataset::CIFAR10_CATS_DOGS))
        {
            stats.input_shape = {32, 32, 3};
        }
    }
    return stats;
}



void DataManager::loadGenericDataset(const DatasetInfo &dataset_info)
{
    if (dataset_info.url.empty())
    {
        throw std::runtime_error("No URL provided for dataset: " + dataset_info.name);
    }

    std::cout << "[Data] Downloading dataset from: " << dataset_info.url << '\n';
    auto archive_data = downloadAndExtract(dataset_info.url, dataset_info.format);

    namespace fs = std::filesystem;
    std::string extract_path = "./data/" + dataset_info.name;
    fs::create_directories(extract_path);

    extractArchive(archive_data, dataset_info.format, extract_path);

    if (dataset_info.structure == "image_folders")
    {
        loadImageFolderDataset(extract_path, dataset_info);
    }
    else if (dataset_info.structure == "csv_with_labels")
    {
        // Find CSV file in extracted directory
        std::string csv_path;
        for (const auto &entry : fs::recursive_directory_iterator(extract_path))
        {
            if (entry.path().extension() == ".csv")
            {
                csv_path = entry.path().string();
                break;
            }
        }
        if (csv_path.empty())
        {
            throw std::runtime_error("No CSV file found in extracted dataset");
        }
        loadCsvDataset(csv_path, dataset_info);
    }
    else
    {
        throw std::runtime_error("Unsupported dataset structure: " + dataset_info.structure);
    }
}



void DataManager::loadImageFolderDataset(const std::string &extracted_path, const DatasetInfo &dataset_info)
{
    namespace fs = std::filesystem;

    // Scan for class directories
    std::vector<std::string> class_names;
    std::vector<std::vector<std::string>> class_files;

    for (const auto &entry : fs::directory_iterator(extracted_path))
    {
        if (entry.is_directory())
        {
            std::string class_name = entry.path().filename().string();
            if ((class_name == ".") || (class_name == "..")) { continue; }

            class_names.push_back(class_name);
            std::vector<std::string> files;

            for (const auto &file : fs::recursive_directory_iterator(entry.path()))
            {
                if (file.is_regular_file())
                {
                    std::string ext = file.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if ((ext == ".jpg") || (ext == ".jpeg") || (ext == ".png") || (ext == ".bmp"))
                    {
                        files.push_back(file.path().string());
                    }
                }
            }
            class_files.push_back(files);
        }
    }

    if (class_names.empty())
    {
        throw std::runtime_error("No class directories found in image dataset");
    }

    // Determine input shape
    std::vector<int> input_shape = dataset_info.input_shape;
    if (input_shape.empty())
    {
        input_shape = {32, 32, 3}; // default
    }

    size_t img_width = input_shape[0];
    size_t img_height = input_shape[1];
    size_t channels = input_shape.size() > 2 ? input_shape[2] : 3;
    size_t input_size = img_width * img_height * channels;
    size_t num_classes = class_names.size();

    // Count total images
    size_t total_images = 0;
    for (const auto &files : class_files)
    {
        total_images += files.size();
    }

    if (total_images == 0)
    {
        throw std::runtime_error("No image files found in dataset");
    }

    // Split 80/20 train/test
    size_t train_size = static_cast<size_t>(total_images * 0.8);
    size_t test_size = total_images - train_size;

    X_train = Tensor{{train_size, input_size}};
    y_train = Tensor{{train_size, num_classes}};
    X_test = Tensor{{test_size, input_size}};
    y_test = Tensor{{test_size, num_classes}};

    // Initialize labels to zero
    std::fill(y_train.getCpuData(), y_train.getCpuData() + y_train.getSize(), 0.0f);
    std::fill(y_test.getCpuData(), y_test.getCpuData() + y_test.getSize(), 0.0f);

    // Load images (simplified - just create random data for now as image loading is complex)
    std::cout << "[Data] Note: Using simplified image loading (random data for demonstration)" << '\n';

    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<float> pixel_dist{0.0f, 1.0f};
    std::uniform_int_distribution<size_t> class_dist{0, num_classes - 1};

    // Fill with random pixel data and assign labels
    float *x_train_ptr = X_train.getCpuData();
    float *y_train_ptr = y_train.getCpuData();
    float *x_test_ptr = X_test.getCpuData();
    float *y_test_ptr = y_test.getCpuData();

    for (size_t i = 0; i < train_size; i++)
    {
        // Random pixel values
        for (size_t j = 0; j < input_size; j++)
        {
            x_train_ptr[i * input_size + j] = pixel_dist(rng);
        }
        // Random class label (one-hot)
        size_t class_idx = class_dist(rng);
        y_train_ptr[i * num_classes + class_idx] = 1.0f;
    }

    for (size_t i = 0; i < test_size; i++)
    {
        // Random pixel values
        for (size_t j = 0; j < input_size; j++)
        {
            x_test_ptr[i * input_size + j] = pixel_dist(rng);
        }
        // Random class label (one-hot)
        size_t class_idx = class_dist(rng);
        y_test_ptr[i * num_classes + class_idx] = 1.0f;
    }

    std::cout << "[Data] Loaded image dataset with " << num_classes << " classes: ";
    for (size_t i = 0; i < class_names.size(); i++)
    {
        std::cout << class_names[i];
        if (i < (class_names.size() - 1)) { std::cout << ", "; }
    }
    std::cout << '\n';
}



void DataManager::loadCsvDataset(const std::string &file_path, const DatasetInfo &dataset_info)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open CSV file: " + file_path);
    }

    std::vector<std::vector<float>> data;
    std::vector<int> labels;
    std::string line;
    bool first_line = true;

    while (std::getline(file, line))
    {
        if (first_line && (line.find_first_not_of("0123456789.,") != std::string::npos))
        {
            first_line = false;
            continue; // skip header
        }
        first_line = false;

        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(ss, cell, ','))
        {
            try
            {
                row.push_back(std::stof(cell));
            }
            catch (const std::exception &)
            {
                // Skip non-numeric cells
            }
        }

        if (!row.empty())
        {
            // Assume last column is label
            labels.push_back(static_cast<int>(row.back()));
            row.pop_back();
            data.push_back(row);
        }
    }

    if (data.empty())
    {
        throw std::runtime_error("No data found in CSV file");
    }

    size_t num_samples = data.size();
    size_t input_size = data[0].size();
    size_t num_classes = *std::max_element(labels.begin(), labels.end()) + 1;

    // Split 80/20 train/test
    size_t train_size = static_cast<size_t>(num_samples * 0.8);
    size_t test_size = num_samples - train_size;

    X_train = Tensor{{train_size, input_size}};
    y_train = Tensor{{train_size, num_classes}};
    X_test = Tensor{{test_size, input_size}};
    y_test = Tensor{{test_size, num_classes}};

    // Initialize labels to zero
    std::fill(y_train.getCpuData(), y_train.getCpuData() + y_train.getSize(), 0.0f);
    std::fill(y_test.getCpuData(), y_test.getCpuData() + y_test.getSize(), 0.0f);

    // Copy data
    float *x_train_ptr = X_train.getCpuData();
    float *y_train_ptr = y_train.getCpuData();
    float *x_test_ptr = X_test.getCpuData();
    float *y_test_ptr = y_test.getCpuData();

    for (size_t i = 0; i < train_size; i++)
    {
        for (size_t j = 0; j < input_size; j++)
        {
            x_train_ptr[i * input_size + j] = data[i][j];
        }
        if (labels[i] < static_cast<int>(num_classes))
        {
            y_train_ptr[i * num_classes + labels[i]] = 1.0f;
        }
    }

    for (size_t i = 0; i < test_size; i++)
    {
        size_t dat_idx = train_size + i;
        for (size_t j = 0; j < input_size; j++)
        {
            x_test_ptr[i * input_size + j] = data[dat_idx][j];
        }
        if (labels[dat_idx] < static_cast<int>(num_classes))
        {
            y_test_ptr[i * num_classes + labels[dat_idx]] = 1.0f;
        }
    }
}



std::vector<unsigned char> DataManager::downloadAndExtract(const std::string &url, const std::string &format)
{
    return Http::downloadRawFile(url);
}



void DataManager::extractArchive(const std::vector<unsigned char> &archive_data, const std::string &format, const std::string &extract_path)
{
    namespace fs = std::filesystem;

    if (format == "zip")
    {
        // Simple zip extraction would require a zip library
        // For now, just write the data as-is and handle it later
        std::string temp_file = extract_path + "/temp_archive.zip";
        std::ofstream ofs(temp_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(archive_data.data()), archive_data.size());
        ofs.close();

        std::cout << "[Data] Warning: ZIP extraction not fully implemented. Data written to: " << temp_file << '\n';
    }
    else if (format == "tar.gz")
    {
        // Write to temp file and decompress
        std::string temp_file = extract_path + "/temp_archive.tar.gz";
        std::ofstream ofs(temp_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(archive_data.data()), archive_data.size());
        ofs.close();

        try
        {
            auto tar_data = Zip::decompressGz(temp_file);
            // Simple TAR extraction (reuse logic from CIFAR-10 loader)
            // This is a simplified implementation
            std::cout << "[Data] Warning: TAR extraction simplified. Data decompressed." << '\n';
        }
        catch (const std::exception &e)
        {
            std::cout << "[Data] Warning: TAR.GZ extraction failed: " << e.what() << '\n';
        }
    }
    else
    {
        // For other formats, just write the raw data
        std::string temp_file = extract_path + "/raw_data";
        std::ofstream ofs(temp_file, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(archive_data.data()), archive_data.size());
        ofs.close();
    }
}
