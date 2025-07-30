#pragma once

#include <string>

class Gemini {
public:
    Gemini();
    std::string ask(const std::string& prompt, int retry_count = 3);
};
