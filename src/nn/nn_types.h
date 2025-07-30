#pragma once

enum class ActivationType {
    ReLU,
    Sigmoid
};

enum class Dataset {
    None,
    MNIST,
    Custom
};

enum class Backend {
    CPU,
    GPU
};
