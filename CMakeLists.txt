cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(xtensor_ml)

# FetchContent module to download xtensor
include(FetchContent)

FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG 0.7.5  # Use a specific tag or version for stability
)
FetchContent_MakeAvailable(xtl)

FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG 0.25.0
)
FetchContent_MakeAvailable(xtensor)

# Fetch xtensor-blas
FetchContent_Declare(
  xtensor_blas
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
  GIT_TAG 0.21.0 
)
FetchContent_MakeAvailable(xtensor_blas)

# Find BLAS library
find_package(BLAS REQUIRED)

# Add your library
add_library(xtensor_ml
    src/core.cpp
    src/models/linear/linear_regression.cpp
    src/matrix.cpp
)

# Include directories for xtensor_ml
target_include_directories(xtensor_ml PUBLIC include)
target_link_libraries(xtensor_ml xtensor xtl)

add_executable(example
   examples/example.cpp
)

target_link_libraries(example xtensor_ml)
