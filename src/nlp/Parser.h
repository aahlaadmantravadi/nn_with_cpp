#pragma once

#include <string>
#include <vector>
#include <memory>
#include "utils/Gemini.h"
#include "nn/nn_types.h"

struct LayerConfig {
    int nodes;
    ActivationType activation;
    bool is_softmax = false;
};

struct ModelConfig {
    bool valid = false;
    std::vector<LayerConfig> layers;
    std::string optimizer;
    std::string dataset;
    bool is_classification = false;
};

class Parser {
public:
    Parser();
    ModelConfig parse(const std::string& command);

private:
    ModelConfig parseWithGemini(const std::string& command);
    ModelConfig parseWithRules(const std::string& command);

    std::unique_ptr<Gemini> gemini;
};
