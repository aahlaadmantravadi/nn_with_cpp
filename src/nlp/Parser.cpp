#include "nlp/Parser.h"



#include "httplib.h"
#include "nlohmann/json.hpp"



#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>



// Helper function to trim markdown fences and whitespace from the Gemini response
std::string cleanGeminiResponse(std::string response)
{
    size_t start = response.find("```json\n");
    if (start != std::string::npos)
    {
        response = response.substr(start + 8);
    }
    size_t end = response.find("```");
    if (end != std::string::npos)
    {
        response = response.substr(0, end);
    }

    start = response.find('{');
    end = response.rfind('}');
    if ((start != std::string::npos) && (end != std::string::npos) && (end > start))
    {
        return response.substr(start, end - start + 1);
    }

    return response;
}



Parser::Parser()
{
    gemini = std::make_unique<Gemini>();
}



ModelConfig Parser::parse(const std::string &command)
{
    ModelConfig config = parseWithGemini(command);
    if (config.valid)
    {
        std::cout << "[Parser] Successfully parsed command with Gemini." << std::endl;
        return config;
    }
    return config; // invalid -> caller will surface error
}



ModelConfig Parser::parseWithGemini(const std::string &command)
{
    std::string prompt =
        "You are an AI dataset resolver and neural network architect. Analyze this request and return ONLY minified JSON (no markdown).\n"
        "For the task described, determine:\n"
        "1. The most appropriate dataset (infer domain, modality, and structure)\n"
        "2. Suggest realistic download URLs if possible\n"
        "3. Whether to auto-generate architecture or use provided specifications\n\n"
        "Required JSON format:\n"
        "{\n"
        "  \"dataset_info\": {\n"
        "    \"name\": \"descriptive_name\",\n"
        "    \"modality\": \"image|text|tabular|audio\",\n"
        "    \"url\": \"https://example.com/dataset.zip\",\n"
        "    \"format\": \"zip|tar.gz|csv|json\",\n"
        "    \"structure\": \"image_folders|csv_with_labels|binary|custom\",\n"
        "    \"expected_classes\": number_or_-1,\n"
        "    \"input_shape\": [width, height, channels] or [features]\n"
        "  },\n"
        "  \"use_ai_architecture\": true|false,\n"
        "  \"layers\": [{\"nodes\": number, \"activation\": \"relu|sigmoid|softmax\"}] or [],\n"
        "  \"optimizer\": \"adam|sgd\",\n"
        "  \"is_classification\": true|false\n"
        "}\n\n"
        "Guidelines:\n"
        "- For image tasks: suggest appropriate image datasets (flowers, animals, objects, etc.)\n"
        "- For text tasks: suggest text classification/NLP datasets\n"
        "- If no specific architecture mentioned, set use_ai_architecture=true and layers=[]\n"
        "- Try to provide real URLs to common datasets (ImageNet subsets, Kaggle, etc.)\n"
        "- Never default to MNIST/CIFAR unless explicitly requested\n"
        "- If no suitable public dataset exists, set url=\"\" and name=\"custom_needed\"\n\n"
        "User request: \"" + command + "\"";

    std::string raw_response = gemini->ask(prompt);
    std::string cleaned_response = cleanGeminiResponse(raw_response);
    if (cleaned_response.rfind("Error:", 0) == 0)
    {
        return ModelConfig{};
    }

    ModelConfig config;
    config.valid = false;

    try
    {
        auto json = nlohmann::json::parse(cleaned_response);

        // Parse dataset_info
        if (json.contains("dataset_info") && json["dataset_info"].is_object())
        {
            auto &ds_info = json["dataset_info"];
            if (ds_info.contains("name") && ds_info["name"].is_string())
            {
                config.dataset_info.name = ds_info["name"];
            }
            if (ds_info.contains("modality") && ds_info["modality"].is_string())
            {
                config.dataset_info.modality = ds_info["modality"];
            }
            if (ds_info.contains("url") && ds_info["url"].is_string())
            {
                config.dataset_info.url = ds_info["url"];
            }
            if (ds_info.contains("format") && ds_info["format"].is_string())
            {
                config.dataset_info.format = ds_info["format"];
            }
            if (ds_info.contains("structure") && ds_info["structure"].is_string())
            {
                config.dataset_info.structure = ds_info["structure"];
            }
            if (ds_info.contains("expected_classes") && ds_info["expected_classes"].is_number())
            {
                config.dataset_info.expected_classes = ds_info["expected_classes"];
            }
            if (ds_info.contains("input_shape") && ds_info["input_shape"].is_array())
            {
                for (auto &dim : ds_info["input_shape"])
                {
                    if (dim.is_number())
                    {
                        config.dataset_info.input_shape.push_back(dim);
                    }
                }
            }
        }

        // Parse use_ai_architecture flag
        if (json.contains("use_ai_architecture") && json["use_ai_architecture"].is_boolean())
        {
            config.use_ai_architecture = json["use_ai_architecture"];
        }

        // Parse optimizer (required)
        if (json.contains("optimizer") && json["optimizer"].is_string())
        {
            config.optimizer = json["optimizer"];
        }
        else
        {
            config.optimizer = "adam"; // default
        }

        // Parse is_classification flag
        if (json.contains("is_classification") && json["is_classification"].is_boolean())
        {
            config.is_classification = json["is_classification"];
        }

        // Parse layers (optional if use_ai_architecture is true)
        if (json.contains("layers") && json["layers"].is_array())
        {
            for (const auto &layer : json["layers"])
            {
                if ((!layer.contains("nodes")) || (!layer["nodes"].is_number()) ||
                    (!layer.contains("activation")) || (!layer["activation"].is_string()))
                {
                    continue; // skip invalid layers
                }

                LayerConfig layer_config;
                layer_config.nodes = layer["nodes"];
                std::string activation = layer["activation"];
                if (activation == "relu")
                {
                    layer_config.activation = ActivationType::ReLU;
                }
                else if (activation == "sigmoid")
                {
                    layer_config.activation = ActivationType::Sigmoid;
                }
                else if (activation == "softmax")
                {
                    layer_config.is_softmax = true;
                }
                else
                {
                    continue; // skip unknown activation
                }
                config.layers.push_back(layer_config);
            }
        }

        // Set legacy dataset field for compatibility
        config.dataset = config.dataset_info.name;

        // Mark as valid if we have dataset info
        if (!config.dataset_info.name.empty())
        {
            config.valid = true;
        }
    }
    catch (const nlohmann::json::parse_error &e)
    {
        std::cout << "[Parser] JSON parse error: " << e.what() << std::endl;
    }

    return config;
}



ModelConfig Parser::parseWithRules(const std::string &command)
{
    ModelConfig config;
    config.valid = false;
    std::string lower = command;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });

    // Quick path: handle "train ... for <dataset>" without explicit architecture
    if ((lower.find("train") != std::string::npos) && (lower.find("for") != std::string::npos))
    {
        if (lower.find("mnist") != std::string::npos)
        {
            config.dataset = "mnist";
            config.optimizer = "adam";
            config.is_classification = true;
            config.layers = {
                LayerConfig{784, ActivationType::ReLU, false},
                LayerConfig{128, ActivationType::ReLU, false},
                LayerConfig{64, ActivationType::ReLU, false},
                LayerConfig{10, ActivationType::ReLU, true}
            };
            config.valid = true;
            return config;
        }
        if ((lower.find("cifar10") != std::string::npos) || (lower.find("cifar-10") != std::string::npos))
        {
            config.dataset = "cifar10";
            config.optimizer = "adam";
            config.is_classification = true;
            config.layers = {
                LayerConfig{3072, ActivationType::ReLU, false},
                LayerConfig{512, ActivationType::ReLU, false},
                LayerConfig{256, ActivationType::ReLU, false},
                LayerConfig{10, ActivationType::ReLU, true}
            };
            config.valid = true;
            return config;
        }
    }

    std::stringstream ss(command);
    std::string word;

    ss >> word; // "build"
    if (word != "build") return config;

    ss >> word; // "784-128-relu-..."
    std::stringstream layer_ss(word);
    std::string segment;

    while (std::getline(layer_ss, segment, '-'))
    {
        try
        {
            int nodes = std::stoi(segment);
            LayerConfig layer_config;
            layer_config.nodes = nodes;
            config.layers.push_back(layer_config);
        }
        catch (const std::invalid_argument &)
        {
            if (config.layers.empty()) return config; // Activation before nodes
            if (segment == "relu")
            {
                config.layers.back().activation = ActivationType::ReLU;
            }
            else if (segment == "sigmoid")
            {
                config.layers.back().activation = ActivationType::Sigmoid;
            }
            else if (segment == "softmax")
            {
                config.layers.back().is_softmax = true;
                config.is_classification = true;
            }
            else
            {
                return config; // Unknown activation
            }
        }
    }

    if (ss >> word && (word == "with") && (ss >> config.optimizer) && (ss >> word) && (word == "for") && (ss >> config.dataset))
    {
        if (config.layers.size() >= 2)
        {
            config.valid = true;
        }
    }

    return config;
}
