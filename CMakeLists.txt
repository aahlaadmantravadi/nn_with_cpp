# =============================================================================
# CMakeLists.txt for TensorFlow from Scratch (MSVC Version)
# =============================================================================
# This file defines the entire build process, including:
# - Project settings (C++23, CUDA).
# - Finding dependencies via vcpkg.
# - Defining the executable target and linking all necessary files.
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# --- Vcpkg Integration ---
# The build.bat script ensures vcpkg is bootstrapped and provides the toolchain file.
# This MUST be declared before the project() command.
if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
elseif(EXISTS "${CMAKE_BINARY_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_BINARY_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
     set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
else()
    message(FATAL_ERROR "vcpkg toolchain file not found. Please run build.bat to bootstrap vcpkg or set VCPKG_ROOT.")
endif()

project(DeepLearningFromScratch LANGUAGES CXX CUDA)

# --- Project Settings ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Find Required Packages (managed by vcpkg) ---
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(miniz REQUIRED)
find_package(httplib REQUIRED)
find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED CONFIG)
find_package(OpenSSL REQUIRED) # FIX: Find OpenSSL

# --- Define Source Files ---
set(APP_SOURCES
    src/main.cpp
    src/backend/cpu/CpuOps.cpp
    src/backend/gpu/GpuOps.cu
    src/data/DataManager.cpp
    src/gui/GuiManager.cpp
    src/gui/Visualizer.cpp
    src/nlp/Parser.cpp
    src/nn/Tensor.cpp
    src/nn/Model.cpp
    src/nn/Loss.cpp
    src/nn/layers/Layer.cpp
    src/nn/layers/Dense.cpp
    src/nn/layers/Activation.cpp
    src/nn/layers/Softmax.cpp
    src/nn/optimizers/Optimizer.cpp
    src/nn/optimizers/SGD.cpp
    src/nn/optimizers/Adam.cpp
    src/utils/Http.cpp
    src/utils/Zip.cpp
    src/utils/Gemini.cpp
)

# --- Define Executable Target ---
add_executable(DeepLearningFromScratch ${APP_SOURCES})

# --- Include Directories ---
target_include_directories(DeepLearningFromScratch PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
)

# --- Link Libraries ---
target_link_libraries(DeepLearningFromScratch PRIVATE
    CUDA::cudart
    OpenGL::GL
    nlohmann_json::nlohmann_json
    miniz::miniz
    httplib::httplib
    glfw
    imgui::imgui
    OpenSSL::SSL      # FIX: Link SSL
    OpenSSL::Crypto   # FIX: Link Crypto
)

# --- Preprocessor Definitions ---
target_compile_definitions(DeepLearningFromScratch PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)

# --- Final Touches ---
set_target_properties(DeepLearningFromScratch PROPERTIES
    CXX_STANDARD 23
    CUDA_STANDARD 20
)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

message(STATUS "Project configured successfully. Dependencies will be downloaded if needed.")
message(STATUS "Run 'cmake --build . --config Release' to compile.")
