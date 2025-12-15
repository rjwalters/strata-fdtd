cmake_minimum_required(VERSION 3.15...3.28)
project(strata_fdtd_kernels LANGUAGES CXX)

# C++17 required for structured bindings, if constexpr, etc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Find OpenMP (optional but recommended)
# On macOS with Homebrew, we need to help CMake find libomp
if(APPLE)
    # Check for Homebrew libomp
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE BREW_LIBOMP_RESULT
    )
    if(BREW_LIBOMP_RESULT EQUAL 0 AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}")
        message(STATUS "Found Homebrew libomp: ${HOMEBREW_LIBOMP_PREFIX}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)

# Platform-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    # Apple Silicon / ARM - NEON is always available
    message(STATUS "Building for ARM64 - enabling NEON optimizations")
    if(APPLE)
        # Apple Silicon specific
        set(ARCH_FLAGS "-mcpu=apple-m1" CACHE STRING "Architecture flags")
    else()
        # Generic ARM64
        set(ARCH_FLAGS "-mcpu=native" CACHE STRING "Architecture flags")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # x86_64 - use portable baseline that works on most CPUs
    # Avoid -march=native which generates code incompatible with different CPUs
    message(STATUS "Building for x86_64 - using portable optimizations")
    include(CheckCXXCompilerFlag)

    # Default: portable x86-64 baseline (SSE2 only)
    # Use -march=x86-64-v2 for broader compatibility (SSE4.2, POPCNT, etc.)
    # Avoid AVX512 as many VMs/runners don't support it
    check_cxx_compiler_flag("-march=x86-64-v2" COMPILER_SUPPORTS_X86_64_V2)
    if(COMPILER_SUPPORTS_X86_64_V2)
        set(ARCH_FLAGS "-march=x86-64-v2" CACHE STRING "Architecture flags")
    else()
        set(ARCH_FLAGS "" CACHE STRING "Architecture flags")
    endif()
else()
    message(STATUS "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH_FLAGS "" CACHE STRING "Architecture flags")
endif()

# Apply architecture flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")

# Optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Source files
set(FDTD_SOURCES
    src/fdtd_step.cpp
    src/boundaries.cpp
    src/pml.cpp
    src/microphones.cpp
    src/ade.cpp
)

# Create the pybind11 module
pybind11_add_module(_kernels
    kernels.cpp
    ${FDTD_SOURCES}
)

# Include directories
target_include_directories(_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(_kernels PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_kernels PRIVATE FDTD_HAS_OPENMP=1)
    message(STATUS "OpenMP found - parallel kernels enabled")
else()
    message(WARNING "OpenMP not found - kernels will be single-threaded")
    target_compile_definitions(_kernels PRIVATE FDTD_HAS_OPENMP=0)
endif()

# Version info
target_compile_definitions(_kernels PRIVATE
    FDTD_KERNELS_VERSION="0.1.0"
)

# Install target (scikit-build-core handles this automatically)
install(TARGETS _kernels DESTINATION .)
