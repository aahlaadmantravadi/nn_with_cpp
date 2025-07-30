#include "utils/Http.h"
#include "httplib.h"
#include "miniz.h"
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
    cli.set_read_timeout(30);  // Increase timeout for large files
    auto res = cli.Get(path.c_str());

    if (res && res->status == 200) {
        std::cout << "[HTTP] Successfully downloaded " << url << " (" << res->body.size() << " bytes)" << std::endl;
        
        // Use miniz gzip decompression API directly on the downloaded data
        mz_stream stream;
        memset(&stream, 0, sizeof(stream));
        
        // Initialize the decompressor
        int status = mz_inflateInit2(&stream, -MZ_DEFAULT_WINDOW_BITS); // Negative window bits for gzip format
        if (status != MZ_OK) {
            throw std::runtime_error("Failed to initialize decompressor");
        }
        
        // Set up the input
        stream.next_in = (const unsigned char*)res->body.data();
        stream.avail_in = (unsigned int)res->body.size();
        
        // Prepare for output - start with a reasonable buffer size
        const size_t CHUNK = 16384;
        std::vector<unsigned char> decompressed_data;
        std::vector<unsigned char> outbuf(CHUNK);
        
        // Decompress the data
        do {
            stream.next_out = outbuf.data();
            stream.avail_out = CHUNK;
            
            status = mz_inflate(&stream, MZ_SYNC_FLUSH);
            
            if (status != MZ_OK && status != MZ_STREAM_END) {
                mz_inflateEnd(&stream);
                throw std::runtime_error("Decompression failed: " + std::string(mz_error(status)));
            }
            
            // Copy the decompressed data to our output vector
            size_t bytes_decompressed = CHUNK - stream.avail_out;
            decompressed_data.insert(decompressed_data.end(), outbuf.begin(), outbuf.begin() + bytes_decompressed);
            
        } while (status != MZ_STREAM_END);
        
        // Clean up
        mz_inflateEnd(&stream);
        
        std::cout << "[HTTP] Decompression successful. Unpacked size: " << decompressed_data.size() << " bytes." << std::endl;
        return decompressed_data;
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
