cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(xtensor_ml)
# FetchContent module to download xtensor
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1  # or latest stable version
)
FetchContent_MakeAvailable(googletest)
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
# Find BLAS and LAPACK libraries
find_package(LAPACK REQUIRED)

# Add xtensor library
add_library(xtensor_ml
    src/core.cpp
    src/models/linear/linear_regression.cpp
    src/models/linear/logistic.cpp
    src/models/trees/dt.cpp
    src/matrix.cpp
)
# Include directories for xtensor_ml
target_include_directories(
    xtensor_ml
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PUBLIC ${xtensor_INCLUDE_DIRS}
    PUBLIC ${xtl_INCLUDE_DIRS}
    PUBLIC ${xtensor-blas_INCLUDE_DIRS}
)
target_link_libraries(
    xtensor_ml
    xtensor
    xtl
    xtensor-blas
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

add_executable(example
   examples/example.cpp
)

target_link_libraries(example xtensor_ml)

# GoogleTest setup (removed unnecessary find_package)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# Collect all test files ending with "_test.cpp"
file(GLOB_RECURSE TEST_SOURCES tests/*_test.cpp)

# Add the test executable
add_executable(run_tests ${TEST_SOURCES})

# Link the test executable with GoogleTest, pthread, and xtensor_ml
target_link_libraries(run_tests xtensor_ml gtest gtest_main pthread)

# Include test directory if necessary
target_include_directories(run_tests PUBLIC ${CMAKE_SOURCE_DIR}/tests)

# Enable testing with CTest
enable_testing()

# Register tests
gtest_discover_tests(run_tests)
