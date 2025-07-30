@echo off
setlocal

echo [BUILD] Starting the build process for a standard Visual Studio installation...

rem --- Find Visual Studio Installation ---
echo [BUILD] Searching for Visual Studio or Build Tools installation...
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do (
    set "VS_INSTALL_PATH=%%i"
)

if not defined VS_INSTALL_PATH (
    echo [ERROR] Visual Studio installation not found. Please install Visual Studio 2019 or later with the C++ workload.
    exit /b 1
)
echo [DEBUG] VS_INSTALL_PATH=%VS_INSTALL_PATH%
echo [BUILD] Found Visual Studio installation at: %VS_INSTALL_PATH%

rem --- Set up MSVC Environment ---
echo [BUILD] Setting up MSVC environment...
call "%VS_INSTALL_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 (
    echo [ERROR] Failed to set up the MSVC environment.
    exit /b 1
)
echo [BUILD] MSVC environment configured successfully.

rem --- Bootstrap Vcpkg ---
set "VCPKG_ROOT=%CD%\vcpkg"
if not exist "%VCPKG_ROOT%\.git" (
    echo [BUILD] Vcpkg not found. Cloning repository...
    git clone https://github.com/microsoft/vcpkg.git "%VCPKG_ROOT%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to clone vcpkg.
        exit /b 1
    )
)

if not exist "%VCPKG_ROOT%\vcpkg.exe" (
    echo [BUILD] Bootstrapping vcpkg...
    call "%VCPKG_ROOT%\bootstrap-vcpkg.bat" -disableMetrics
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to bootstrap vcpkg.
        exit /b 1
    )
)
echo [BUILD] Vcpkg is ready.

rem --- Find Required Tools (nvcc, cmake) ---
echo [BUILD] Searching for required tools (nvcc, cmake)...
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] nvcc not found in PATH. Please ensure the NVIDIA CUDA Toolkit is installed and configured correctly.
    exit /b 1
)
echo [BUILD] Found nvcc at:
where nvcc

set "CMAKE_EXE=%~dp0\vendor\cmake\cmake-3.28.1-windows-x86_64\bin\cmake.exe"
if not exist "%CMAKE_EXE%" (
    echo [ERROR] CMake not found at the expected path: %CMAKE_EXE%
    exit /b 1
)
echo [BUILD] Using CMake at: %CMAKE_EXE%

rem --- Configure and Build with CMake ---
set "BUILD_DIR=%~dp0\build"
mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

echo [BUILD] Configuring the project with CMake...
"%CMAKE_EXE%" .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed. Check the output above for errors.
    exit /b 1
)
echo [BUILD] CMake configuration successful.

echo [BUILD] Compiling the source code...
"%CMAKE_EXE%" --build . --config Release
if %errorlevel% neq 0 (
    echo [ERROR] Build failed. Check the MSBuild output above for errors.
    exit /b 1
)

echo [BUILD] Build successful!
echo [BUILD] You can now find the executable in the 'build\Release' directory.
echo [BUILD] Run 'DeepLearningFromScratch.exe' to start the application.

endlocal
