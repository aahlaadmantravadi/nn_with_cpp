#include "utils/Http.h"
#include "httplib.h"
#include <fstream>
#include <iostream>
#include <cstring> // For memset

void Http::downloadFile(const std::string& host, const std::string& path, const std::string& out_path) {
    httplib::Client cli(host);
    auto res = cli.Get(path.c_str());

    if (res && res->status == 200) {
        std::ofstream ofs(out_path, std::ios::binary);
        ofs << res->body;
    } else {
        std::string error_message = "[HTTP] Error: Download failed with status code: ";
        if(res) {
            error_message += std::to_string(res->status);
        } else {
            error_message += "unknown";
        }
        throw std::runtime_error(error_message);
    }
}

void Http::downloadFileFromUrl(const std::string& url, const std::string& out_path) {
    std::string host;
    std::string path;
    size_t host_end;

    if (url.rfind("https", 0) == 0) {
        host = url.substr(0, url.find("/", 8));
        path = url.substr(url.find("/", 8));
    } else {
        host = url.substr(0, url.find("/", 7));
        path = url.substr(url.find("/", 7));
    }
    
    downloadFile(host, path, out_path);
}

std::vector<unsigned char> Http::downloadAndDecompress(const std::string& url) {
    std::cout << "[HTTP] Downloading and decompressing " << url << "..." << std::endl;

    // 1. Download the file
    std::string url_path = url;
    std::string host;
    if (url.rfind("http://", 0) == 0) {
        url_path = url.substr(7);
    }
    size_t path_start = url_path.find('/');
    if (path_start != std::string::npos) {
        host = url_path.substr(0, path_start);
        url_path = url_path.substr(path_start);
    } else {
        host = url_path;
        url_path = "/";
    }

    httplib::Client cli(host.c_str());
    cli.set_follow_location(true);
    auto res = cli.Get(url_path.c_str());

    if (!res || res->status != 200) {
        std::cerr << "[HTTP] Error: Failed to download " << url << ". Status: " << (res ? res->status : -1) << std::endl;
        std::string fallback_host = "ossci-datasets.s3.amazonaws.com";
        std::string fallback_path = "/mnist/" + url.substr(url.find_last_of('/') + 1);
        std::cout << "[HTTP] Attempting fallback URL: http://" << fallback_host << fallback_path << std::endl;
        httplib::Client fallback_cli(fallback_host.c_str());
        fallback_cli.set_follow_location(true);
        res = fallback_cli.Get(fallback_path.c_str());
        if (!res || res->status != 200) {
            throw std::runtime_error("Failed to download from both primary and fallback URLs.");
        }
    }

    const std::vector<char> compressed_data(res->body.begin(), res->body.end());
    std::cout << "[HTTP] Download successful. Compressed size: " << compressed_data.size() << " bytes." << std::endl;

    // 2. Decompress the data using zlib
    const size_t CHUNK = 16384;
    std::vector<unsigned char> decompressed_data;
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;

    if (inflateInit2(&strm, 15 + 16) != Z_OK) {
        throw std::runtime_error("zlib inflateInit failed");
    }

    strm.avail_in = compressed_data.size();
    strm.next_in = reinterpret_cast<unsigned char*>(const_cast<char*>(compressed_data.data()));

    int ret;
    unsigned char out[CHUNK];
    do {
        strm.avail_out = CHUNK;
        strm.next_out = out;
        ret = inflate(&strm, Z_NO_FLUSH);
        switch (ret) {
            case Z_NEED_DICT:
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                inflateEnd(&strm);
                throw std::runtime_error("zlib inflation error: " + std::to_string(ret));
        }
        size_t have = CHUNK - strm.avail_out;
        decompressed_data.insert(decompressed_data.end(), out, out + have);
    } while (strm.avail_out == 0);

    inflateEnd(&strm);

    std::cout << "[HTTP] Decompression successful. Unpacked size: " << decompressed_data.size() << " bytes." << std::endl;
    return decompressed_data;
}

std::vector<unsigned char> Http::downloadRawFile(const std::string& url) {
    std::string host;
    std::string path;

    if (url.rfind("https", 0) == 0) {
        host = url.substr(0, url.find("/", 8));
        path = url.substr(url.find("/", 8));
    } else {
        host = url.substr(0, url.find("/", 7));
        path = url.substr(url.find("/", 7));
    }

    httplib::Client cli(host);
    cli.set_read_timeout(60);  // Increase timeout for large files
    cli.set_connection_timeout(15); // Connection timeout
    auto res = cli.Get(path.c_str());

    if (res && res->status == 200) {
        std::cout << "[HTTP] Successfully downloaded " << url << " (" << res->body.size() << " bytes)" << std::endl;
        
        // Convert the response body to a vector of unsigned chars
        const unsigned char* data = reinterpret_cast<const unsigned char*>(res->body.data());
        std::vector<unsigned char> file_data(data, data + res->body.size());
        
        return file_data;
    } else {
        std::string error_message = "[HTTP] Error: Download failed with status code: ";
        if(res) {
            error_message += std::to_string(res->status);
        } else {
            error_message += "unknown (connection error or timeout)";
        }
        throw std::runtime_error(error_message);
    }
}
