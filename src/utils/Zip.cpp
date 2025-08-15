// =============================================================================
// File: src/utils/Zip.cpp
// =============================================================================
//
// Description: Implements the decompression utility using miniz.
//
// =============================================================================

#include "utils/Zip.h"



#include <zlib.h>



#include <cstring>  // For memset
#include <fstream>
#include <iostream>
#include <vector>



namespace Zip
{

// Decompresses a .gz file into a vector of bytes.
// Used for the MNIST dataset which is gzipped.
std::vector<unsigned char> decompressGz(const std::string &file_path)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::cout << "[Zip] Decompressing " << file_path << " (" << size << " bytes) using zlib..." << std::endl;

    std::vector<unsigned char> compressed_data(size);
    if (!file.read(reinterpret_cast<char *>(compressed_data.data()), size))
    {
        throw std::runtime_error("Failed to read file: " + file_path);
    }

    const size_t CHUNK = 16384;
    std::vector<unsigned char> decompressed_data;
    z_stream strm = {};
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;

    if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK)
    {
        throw std::runtime_error("inflateInit2 failed");
    }

    strm.avail_in = compressed_data.size();
    strm.next_in = compressed_data.data();

    int ret;
    unsigned char out[CHUNK];
    do
    {
        strm.avail_out = CHUNK;
        strm.next_out = out;
        ret = inflate(&strm, Z_NO_FLUSH);
        switch (ret)
        {
            case Z_NEED_DICT:
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                inflateEnd(&strm);
                throw std::runtime_error("zlib inflation error: " + std::to_string(ret));
        }
        size_t have = CHUNK - strm.avail_out;
        decompressed_data.insert(decompressed_data.end(), out, out + have);
    }
    while (strm.avail_out == 0);

    inflateEnd(&strm);

    if (ret != Z_STREAM_END)
    {
        throw std::runtime_error("Gzip decompression failed: stream did not end properly.");
    }

    std::cout << "[Zip] Decompression successful. Unpacked size: " << decompressed_data.size() << " bytes." << std::endl;
    return decompressed_data;
}

} // namespace Zip
