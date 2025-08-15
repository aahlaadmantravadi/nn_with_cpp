#include "utils/Gemini.h"



#include "nlohmann/json.hpp"
#include "httplib.h"



#include "utils/ApiKey.h"



#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>



Gemini::Gemini()
{
}



std::string Gemini::ask(const std::string &prompt, int retry_count)
{
    // Optional offline mode for local testing (disabled by default)
    static bool offline_mode = false;
    if (offline_mode)
    {
        std::cout << "[Gemini] Using offline mode for parsing." << std::endl;

        // AI-style offline responses for common tasks
        std::string query = prompt;
        std::transform(query.begin(), query.end(), query.begin(),
                      [](unsigned char c) { return std::tolower(c); });

        auto build_ai_json = [](const std::string &task_type, const std::string &name, const std::string &url = "")
        {
            std::string json = "{\n";
            json += "  \"dataset_info\": {\n";
            json += "    \"name\": \"" + name + "\",\n";

            if (task_type == "image")
            {
                json += "    \"modality\": \"image\",\n";
                json += "    \"url\": \"" + (url.empty() ? "https://example.com/" + name + ".zip" : url) + "\",\n";
                json += "    \"format\": \"zip\",\n";
                json += "    \"structure\": \"image_folders\",\n";
                json += "    \"expected_classes\": 5,\n";
                json += "    \"input_shape\": [64, 64, 3]\n";
            }
            else if (task_type == "tabular")
            {
                json += "    \"modality\": \"tabular\",\n";
                json += "    \"url\": \"" + (url.empty() ? "https://example.com/" + name + ".csv" : url) + "\",\n";
                json += "    \"format\": \"csv\",\n";
                json += "    \"structure\": \"csv_with_labels\",\n";
                json += "    \"expected_classes\": 3,\n";
                json += "    \"input_shape\": [10]\n";
            }
            else
            {
                json += "    \"modality\": \"image\",\n";
                json += "    \"url\": \"\",\n";
                json += "    \"format\": \"custom\",\n";
                json += "    \"structure\": \"custom\",\n";
                json += "    \"expected_classes\": -1,\n";
                json += "    \"input_shape\": []\n";
            }

            json += "  },\n";
            json += "  \"use_ai_architecture\": true,\n";
            json += "  \"layers\": [],\n";
            json += "  \"optimizer\": \"adam\",\n";
            json += "  \"is_classification\": true\n";
            json += "}";
            return json;
        };

        // Analyze query for task type
        if (query.find("rice") != std::string::npos)
        {
            return build_ai_json("image", "rice_varieties", "https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset");
        }
        else if (query.find("flower") != std::string::npos)
        {
            return build_ai_json("image", "flower_species", "https://www.kaggle.com/datasets/alxmamaev/flowers-recognition");
        }
        else if ((query.find("dog") != std::string::npos) || (query.find("cat") != std::string::npos))
        {
            return build_ai_json("image", "cats_vs_dogs", "https://www.kaggle.com/datasets/tongpython/cat-and-dog");
        }
        else if ((query.find("handwritten") != std::string::npos) || (query.find("mnist") != std::string::npos))
        {
            return build_ai_json("image", "mnist", "");
        }
        else if (query.find("cifar") != std::string::npos)
        {
            return build_ai_json("image", "cifar10", "");
        }
        else if (query.find("animal") != std::string::npos)
        {
            return build_ai_json("image", "animal_classification", "https://www.kaggle.com/datasets/alessiocorrado99/animals10");
        }
        else if (query.find("color") != std::string::npos)
        {
            return build_ai_json("image", "color_classification", "");
        }
        else
        {
            // For unknown tasks, indicate no suitable dataset found
            return "{\n  \"dataset_info\": {\n    \"name\": \"custom_needed\",\n    \"modality\": \"unknown\",\n    \"url\": \"\",\n    \"format\": \"unknown\",\n    \"structure\": \"unknown\",\n    \"expected_classes\": -1,\n    \"input_shape\": []\n  },\n  \"use_ai_architecture\": true,\n  \"layers\": [],\n  \"optimizer\": \"adam\",\n  \"is_classification\": true\n}";
        }
    }

    // Online mode - only used if offline_mode is false
    if (retry_count <= 0)
    {
        return "Error: Maximum retry attempts reached.";
    }
    httplib::Client cli("https://generativelanguage.googleapis.com");
    cli.set_connection_timeout(30);
    cli.set_read_timeout(30);

    nlohmann::json req_body;
    req_body["contents"][0]["parts"][0]["text"] = prompt;
    // Avoid unsupported generationConfig in v1; rely on prompt engineering for JSON-only

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    std::string api_key(ApiKey::kGemini);
    std::string path = "/v1/models/gemini-1.5-flash:generateContent?key=" + api_key; // Using lightweight model for simple instructions

    auto res = cli.Post(path, headers, req_body.dump(), "application/json");

    if (res)
    {
        if (res->status == 200)
        {
            try
            {
                auto json_res = nlohmann::json::parse(res->body);
                if (json_res.contains("candidates") && (!json_res["candidates"].empty()))
                {
                    return json_res["candidates"][0]["content"]["parts"][0]["text"];
                }
                else
                {
                    return "Error: Invalid JSON structure from Gemini.";
                }
            }
            catch (const nlohmann::json::parse_error &e)
            {
                return std::string("Error: JSON parse error - ") + e.what();
            }
        }
        else
        {
            std::cout << "[Gemini DEBUG] Response Status: " << res->status << std::endl;
            std::cout << "[Gemini DEBUG] Response Body: " << res->body << std::endl;

            // If service is overloaded (503), retry with a short delay
            if ((res->status == 503) && (retry_count > 1))
            {
                std::cout << "[Gemini] Service overloaded. Retrying in 2 seconds..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(2));
                return ask(prompt, retry_count - 1);
            }

            // If we still couldn't connect, return an error string (no forced offline fallback)
            return std::string("Error: Gemini API status ") + std::to_string(res->status);
        }
    }
    else
    {
        auto err = res.error();
        std::cout << "[Gemini DEBUG] HTTP Error: " << httplib::to_string(err) << std::endl;
        // Network error, bubble up as error text
        return std::string("Error: Network ") + httplib::to_string(err);
    }
}
