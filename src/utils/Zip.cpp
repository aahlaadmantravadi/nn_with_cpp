// =============================================================================
// File: src/utils/Zip.cpp
// =============================================================================
//
// Description: Implements the decompression utility using miniz.
//
// =============================================================================

#include "utils/Zip.h"
// FIX: Include the correct main header for the miniz library.
#include "miniz.h" 
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>  // For memset

namespace Zip {

std::vector<unsigned char> decompressGz(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "[Zip] Error: Cannot open file " << file_path << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> compressed_data(size);
    if (!file.read(compressed_data.data(), size)) {
        std::cerr << "[Zip] Error: Failed to read file " << file_path << std::endl;
        return {};
    }
    file.close();

    std::cout << "[Zip] Decompressing " << file_path << " (" << size << " bytes)..." << std::endl;

    // Use miniz gzip decompression API which handles gzip headers properly
    mz_stream stream;
    memset(&stream, 0, sizeof(stream));
    
    // Initialize the decompressor
    int status = mz_inflateInit2(&stream, -MZ_DEFAULT_WINDOW_BITS); // Negative window bits for gzip format
    if (status != MZ_OK) {
        std::cerr << "[Zip] Error: Failed to initialize decompressor. miniz error: " << mz_error(status) << std::endl;
        return {};
    }
    
    // Set up the input
    stream.next_in = (const unsigned char*)compressed_data.data();
    stream.avail_in = (unsigned int)compressed_data.size();
    
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
            std::cerr << "[Zip] Error: Decompression failed. miniz error: " << mz_error(status) << std::endl;
            return {};
        }
        
        // Copy the decompressed data to our output vector
        size_t bytes_decompressed = CHUNK - stream.avail_out;
        decompressed_data.insert(decompressed_data.end(), outbuf.begin(), outbuf.begin() + bytes_decompressed);
        
    } while (status != MZ_STREAM_END);
    
    // Clean up
    mz_inflateEnd(&stream);
    
    std::cout << "[Zip] Decompression successful. Unpacked size: " << decompressed_data.size() << " bytes." << std::endl;
    
    return decompressed_data;
}

} // namespace Zip
