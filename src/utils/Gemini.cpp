#include "utils/Gemini.h"
#include "utils/ApiKey.h"
#include "httplib.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cctype>

Gemini::Gemini() {}

std::string Gemini::ask(const std::string& prompt, int retry_count) {
    // Check for offline mode first - use this for bypass during development/testing
    static bool offline_mode = false;
    
    if (offline_mode) {
        std::cout << "[Gemini] Using offline mode for parsing." << std::endl;
        
        // Very simple rule-based parser that handles common commands
        std::string query = prompt;
        std::transform(query.begin(), query.end(), query.begin(), 
                      [](unsigned char c) { return std::tolower(c); });
        
        if (query.find("mnist") != std::string::npos) {
            if (query.find("train") != std::string::npos || query.find("build") != std::string::npos) {
                return "Command type: build\nDataset: mnist\nArchitecture: 784-128-relu-10-softmax\nOptimizer: adam\nLearning rate: 0.001\nBatch size: 32";
            }
        }
        
        // Default response for unrecognized commands
        return "Command type: unrecognized\nPlease try a standard format like 'build a network for mnist'";
    }
    
    // Online mode - only used if offline_mode is false
    if (retry_count <= 0) {
        return "Error: Maximum retry attempts reached.";
    }
    httplib::Client cli("https://generativelanguage.googleapis.com");
    cli.set_connection_timeout(30);
    cli.set_read_timeout(30);

    nlohmann::json req_body;
    req_body["contents"][0]["parts"][0]["text"] = prompt;

    httplib::Headers headers = {
        {"Content-Type", "application/json"}
    };

    std::string api_key = ApiKey::Gemini;
    std::string path = "/v1/models/gemini-1.5-flash:generateContent?key=" + api_key; // Using lightweight model for simple instructions

    auto res = cli.Post(path, headers, req_body.dump(), "application/json");

    if (res) {
        if (res->status == 200) {
            try {
                auto json_res = nlohmann::json::parse(res->body);
                if (json_res.contains("candidates") && !json_res["candidates"].empty()) {
                    return json_res["candidates"][0]["content"]["parts"][0]["text"];
                } else {
                    return "Error: Invalid JSON structure from Gemini.";
                }
            } catch (const nlohmann::json::parse_error& e) {
                return std::string("Error: JSON parse error - ") + e.what();
            }
        } else {
            std::cout << "[Gemini DEBUG] Response Status: " << res->status << std::endl;
            std::cout << "[Gemini DEBUG] Response Body: " << res->body << std::endl;
            
            // If service is overloaded (503), retry with a short delay
            if (res->status == 503 && retry_count > 1) {
                std::cout << "[Gemini] Service overloaded. Retrying in 2 seconds..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(2));
                return ask(prompt, retry_count - 1);
            }
            
            // If we still couldn't connect, fall back to offline mode
            std::cout << "[Gemini] API connection failed, falling back to offline mode." << std::endl;
            offline_mode = true;
            return ask(prompt, 1); // Retry once in offline mode
        }
    } else {
        auto err = res.error();
        std::cout << "[Gemini DEBUG] HTTP Error: " << httplib::to_string(err) << std::endl;
        
        // Network error, fall back to offline mode
        std::cout << "[Gemini] Network error, falling back to offline mode." << std::endl;
        offline_mode = true;
        return ask(prompt, 1); // Retry once in offline mode
    }
}
