#include "nlp/Parser.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

// Helper function to trim markdown fences and whitespace from the Gemini response
std::string cleanGeminiResponse(std::string response) {
    size_t start = response.find("```json\n");
    if (start != std::string::npos) {
        response = response.substr(start + 8);
    }
    size_t end = response.find("```");
    if (end != std::string::npos) {
        response = response.substr(0, end);
    }

    start = response.find('{');
    end = response.rfind('}');
    if (start != std::string::npos && end != std::string::npos && end > start) {
        return response.substr(start, end - start + 1);
    }
    
    return response;
}

Parser::Parser() {
    gemini = std::make_unique<Gemini>();
}

ModelConfig Parser::parse(const std::string& command) {
    ModelConfig config = parseWithGemini(command);
    if (config.valid) {
        std::cout << "[Parser] Successfully parsed command with Gemini." << std::endl;
        return config;
    }

    std::cout << "[Parser] Gemini parsing failed. Falling back to rule-based parser." << std::endl;
    return parseWithRules(command);
}

ModelConfig Parser::parseWithGemini(const std::string& command) {
    std::string prompt = 
        "Parse the following user command into a valid JSON object. Do NOT use markdown or any other formatting. The JSON object must have three keys: 'layers', 'optimizer', and 'dataset'.\n"
        "1. 'layers': An array of objects, where each object has two keys: 'nodes' (an integer) and 'activation' (a string which must be one of 'relu', 'sigmoid', or 'softmax').\n"
        "2. 'optimizer': A string representing the optimization algorithm (e.g., 'adam', 'sgd').\n"
        "3. 'dataset': A string for the dataset name (e.g., 'mnist', 'cifar10').\n\n"
        "Example Command: 'build 784-128-relu-64-relu-10-softmax with adam for mnist'\n"
        "Expected JSON output for the example:\n"
        "{\n"
        "  \"layers\": [\n"
        "    {\"nodes\": 784, \"activation\": \"relu\"},\n"
        "    {\"nodes\": 128, \"activation\": \"relu\"},\n"
        "    {\"nodes\": 64, \"activation\": \"relu\"},\n"
        "    {\"nodes\": 10, \"activation\": \"softmax\"}\n"
        "  ],\n"
        "  \"optimizer\": \"adam\",\n"
        "  \"dataset\": \"mnist\"\n"
        "}\n\n"
        "User Command: \"" + command + "\"\n"
        "JSON output:";

    std::string raw_response = gemini->ask(prompt);
    std::string cleaned_response = cleanGeminiResponse(raw_response);
    
    ModelConfig config;
    config.valid = false;

    try {
        auto json = nlohmann::json::parse(cleaned_response);
        
        if (!json.contains("layers") || !json["layers"].is_array() || json["layers"].empty() ||
            !json.contains("optimizer") || !json["optimizer"].is_string() ||
            !json.contains("dataset") || !json["dataset"].is_string()) {
            return config;
        }

        for (const auto& layer : json["layers"]) {
            if (!layer.contains("nodes") || !layer["nodes"].is_number() ||
                !layer.contains("activation") || !layer["activation"].is_string()) {
                return config;
            }

            LayerConfig layer_config;
            layer_config.nodes = layer["nodes"];
            std::string activation = layer["activation"];
            if (activation == "relu") {
                layer_config.activation = ActivationType::ReLU;
            } else if (activation == "sigmoid") {
                layer_config.activation = ActivationType::Sigmoid;
            } else if (activation == "softmax") {
                config.is_classification = true;
                layer_config.is_softmax = true;
            } else {
                return config;
            }
            config.layers.push_back(layer_config);
        }
        
        config.optimizer = json["optimizer"];
        config.dataset = json["dataset"];
        config.valid = true;

    } catch (const nlohmann::json::parse_error& e) {
        // This is now an expected failure case if the API is down
    }

    return config;
}

ModelConfig Parser::parseWithRules(const std::string& command) {
    ModelConfig config;
    config.valid = false;
    std::stringstream ss(command);
    std::string word;

    ss >> word; // "build"
    if (word != "build") return config;

    ss >> word; // "784-128-relu-..."
    std::stringstream layer_ss(word);
    std::string segment;

    while(std::getline(layer_ss, segment, '-')) {
        try {
            int nodes = std::stoi(segment);
            LayerConfig layer_config;
            layer_config.nodes = nodes;
            config.layers.push_back(layer_config);
        } catch (const std::invalid_argument&) {
            if (config.layers.empty()) return config; // Activation before nodes
            if (segment == "relu") {
                config.layers.back().activation = ActivationType::ReLU;
            } else if (segment == "sigmoid") {
                config.layers.back().activation = ActivationType::Sigmoid;
            } else if (segment == "softmax") {
                config.layers.back().is_softmax = true;
                config.is_classification = true;
            } else {
                return config; // Unknown activation
            }
        }
    }

    if (ss >> word && word == "with" && ss >> config.optimizer && ss >> word && word == "for" && ss >> config.dataset) {
        if (config.layers.size() >= 2) {
            config.valid = true;
        }
    }
    
    return config;
}
