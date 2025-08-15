#pragma once



// --- Standard Includes ---
#include <cstdint>



enum class ActivationType : std::uint8_t
{
    ReLU,
    Sigmoid,
};



enum class Dataset : std::uint8_t
{
    None,
    MNIST,
    CIFAR10,
    CIFAR10_CATS_DOGS,
    Custom,
};



enum class Backend : std::uint8_t
{
    CPU,
    GPU,
};
