cmake_minimum_required(VERSION 3.20)

project(CodedLLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

find_package(CUDAToolkit REQUIRED)

add_library(codedllm
  src/cpu_reference.cpp
  src/coding/codec.cpp
  src/coding/graph.cpp
  kernels/sparse_decoder.cu
)
target_include_directories(codedllm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(codedllm PUBLIC CUDA::cudart)
set_target_properties(codedllm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_STANDARD 17
)

enable_testing()
include(GoogleTest)

add_executable(test_decoder
  tests/test_decoder.cpp
  tests/unit/test_core_types.cpp
  tests/unit/test_codec.cpp
)
target_link_libraries(test_decoder PRIVATE codedllm GTest::gtest_main)
set_target_properties(test_decoder PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)
gtest_discover_tests(test_decoder)
