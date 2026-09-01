cmake_minimum_required(VERSION 3.20)

project(CodedLLM LANGUAGES CXX CUDA)

# The "native" selector was introduced in CMake 3.24. Ubuntu 22.04's
# CMake 3.22 is still used by the CUDA 12.4 development container, so retain
# an sm_75 fallback for the project's current RTX 2080 Ti development GPU.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set(CMAKE_CUDA_ARCHITECTURES "native")
else()
  set(CMAKE_CUDA_ARCHITECTURES "75")
  message(STATUS
    "CMake < 3.24 does not support CUDA architecture 'native'; using sm_75"
  )
endif()

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

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
)
FetchContent_MakeAvailable(googlebenchmark)

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

add_executable(bench_decoder benchmarks/bench_decoder.cpp)
target_include_directories(bench_decoder PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_link_libraries(bench_decoder PRIVATE codedllm benchmark::benchmark)

add_executable(bench_decode_plan benchmarks/bench_decode_plan.cpp)
target_link_libraries(bench_decode_plan PRIVATE codedllm benchmark::benchmark)

add_executable(device_info tools/device_info.cu)
target_include_directories(device_info PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(device_info PRIVATE CUDA::cudart)
