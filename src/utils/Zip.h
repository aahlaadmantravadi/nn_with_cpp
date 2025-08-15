// =============================================================================
// File: src/utils/Zip.h
// =============================================================================
//
// Description: Declares a utility class for handling zip and gzip files.
//              This is a wrapper around the miniz and zlib libraries to
//              simplify decompressing datasets.
//
// =============================================================================

#pragma once



#include <string>
#include <vector>



#include <zlib.h>



namespace Zip
{

// Decompresses a .gz file into a vector of bytes.
// Used for the MNIST dataset which is gzipped.
[[nodiscard]] std::vector<unsigned char> decompressGz(const std::string &file_path);

// TODO: Add functionality for standard .zip archives if needed for other datasets.

} // namespace Zip
