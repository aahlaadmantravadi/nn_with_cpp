#pragma once



#include "nn/nn_types.h"
#include "utils/Gemini.h"



#include <memory>
#include <string>
#include <vector>



struct LayerConfig
{
    int nodes;
    ActivationType activation;
    bool is_softmax = false;
};



struct DatasetInfo
{
    std::string name;           // e.g., "rice_varieties", "flower_species"
    std::string modality;       // e.g., "image", "text", "tabular"
    std::string url;            // download URL
    std::string format;         // e.g., "zip", "tar.gz", "csv"
    std::string structure;      // e.g., "image_folders", "csv_with_labels", "binary"
    int expected_classes = -1;  // -1 if unknown
    std::vector<int> input_shape; // e.g., {32, 32, 3} for images
};



struct ModelConfig
{
    bool valid = false;
    std::vector<LayerConfig> layers;
    std::string optimizer;
    std::string dataset;        // legacy field, kept for compatibility
    DatasetInfo dataset_info;   // new AI-resolved dataset information
    bool is_classification = false;
    bool use_ai_architecture = false; // if true, ignore layers and infer from data
};



class Parser
{
public:
    Parser();
    [[nodiscard]] ModelConfig parse(const std::string &command);

private:
    [[nodiscard]] ModelConfig parseWithGemini(const std::string &command);
    [[nodiscard]] ModelConfig parseWithRules(const std::string &command);

    std::unique_ptr<Gemini> gemini;
};
