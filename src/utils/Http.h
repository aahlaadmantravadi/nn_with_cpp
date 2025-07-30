#pragma once

#include <string>
#include <vector>

namespace Http {
    void downloadFile(const std::string& host, const std::string& path, const std::string& out_path);
    void downloadFileFromUrl(const std::string& url, const std::string& out_path);
    
    // Download and decompress a gzip file directly to avoid file corruption issues
    std::vector<unsigned char> downloadAndDecompress(const std::string& url);
    
    // Download a raw file directly into memory (no decompression)
    std::vector<unsigned char> downloadRawFile(const std::string& url);
}
