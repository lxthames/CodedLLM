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
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(Threads REQUIRED)

add_library(codedllm_coding
  src/coding/codec.cpp
  src/coding/graph.cpp
  src/coding/graph_generator.cpp
  src/coding/planner.cpp
)
target_include_directories(codedllm_coding PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(codedllm_cuda
  src/cpu_reference.cpp
  kernels/sparse_decoder.cu
)
target_include_directories(codedllm_cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(codedllm_cuda PUBLIC codedllm_coding CUDA::cudart)
set_target_properties(codedllm_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_STANDARD 17
)

# Compatibility target for consumers that require both host and CUDA layers.
add_library(codedllm INTERFACE)
target_link_libraries(codedllm INTERFACE codedllm_coding codedllm_cuda)

add_library(codedllm_runtime
  src/runtime/coordinator.cpp
  src/runtime/policy.cpp
)
target_include_directories(codedllm_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(codedllm_runtime PUBLIC codedllm_coding)

add_library(codedllm_serving_cuda
  src/runtime/cuda_recovery_executor.cpp
)
target_include_directories(codedllm_serving_cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(codedllm_serving_cuda PUBLIC
  codedllm_runtime
  codedllm_cuda
  Threads::Threads
)

add_library(codedllm_simulation
  src/simulation/cluster_simulator.cpp
  src/simulation/recovery_queue.cpp
  src/simulation/simulator.cpp
  src/simulation/workload_generator.cpp
)
target_include_directories(codedllm_simulation PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(codedllm_simulation PUBLIC codedllm_runtime codedllm_coding)

enable_testing()
include(GoogleTest)

add_executable(test_decoder
  tests/test_decoder.cpp
  tests/unit/test_core_types.cpp
  tests/unit/test_codec.cpp
  tests/property/test_peeling_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE codedllm GTest::gtest_main)
set_target_properties(test_decoder PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)
gtest_discover_tests(test_decoder)

add_executable(test_differential_recovery
  tests/integration/test_differential_recovery.cpp
)
target_link_libraries(test_differential_recovery PRIVATE
  codedllm
  GTest::gtest_main
)
set_target_properties(test_differential_recovery PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/integration
)
gtest_discover_tests(test_differential_recovery)

add_executable(test_recovery_policy
  tests/unit/test_recovery_policy.cpp
)
target_link_libraries(test_recovery_policy PRIVATE
  codedllm_runtime
  codedllm_coding
  GTest::gtest_main
)
set_target_properties(test_recovery_policy PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit
)
gtest_discover_tests(test_recovery_policy)

add_executable(test_recovery_coordinator
  tests/integration/test_recovery_coordinator.cpp
)
target_link_libraries(test_recovery_coordinator PRIVATE
  codedllm_runtime
  codedllm_coding
  GTest::gtest_main
)
set_target_properties(test_recovery_coordinator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/integration
)
gtest_discover_tests(test_recovery_coordinator)

add_executable(test_cuda_recovery_coordinator
  tests/integration/test_cuda_recovery_coordinator.cpp
)
target_link_libraries(test_cuda_recovery_coordinator PRIVATE
  codedllm_serving_cuda
  codedllm_runtime
  codedllm_coding
  GTest::gtest_main
)
set_target_properties(test_cuda_recovery_coordinator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/integration
)
gtest_discover_tests(test_cuda_recovery_coordinator)

add_executable(test_simulator
  tests/unit/test_simulator.cpp
)
target_link_libraries(test_simulator PRIVATE
  codedllm_simulation
  codedllm_runtime
  codedllm_coding
  GTest::gtest_main
)
set_target_properties(test_simulator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit
)
gtest_discover_tests(test_simulator)

add_executable(test_cluster_simulator
  tests/unit/test_cluster_simulator.cpp
)
target_link_libraries(test_cluster_simulator PRIVATE
  codedllm_simulation
  codedllm_runtime
  codedllm_coding
  GTest::gtest_main
)
set_target_properties(test_cluster_simulator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit
)
gtest_discover_tests(test_cluster_simulator)

add_executable(test_workload_generator
  tests/unit/test_workload_generator.cpp
)
target_link_libraries(test_workload_generator PRIVATE
  codedllm_simulation
  codedllm_runtime
  codedllm_coding
  GTest::gtest_main
)
set_target_properties(test_workload_generator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit
)
gtest_discover_tests(test_workload_generator)

add_test(
  NAME Phase7AnalysisTest
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_analyze_phase7.py
)
add_test(
  NAME Phase7DecodeCostExtractionTest
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_extract_phase7_decode_costs.py
)

add_executable(phase7_parameter_sweep
  experiments/phase7_parameter_sweep.cpp
)
target_link_libraries(phase7_parameter_sweep PRIVATE
  codedllm_simulation
  codedllm_runtime
  codedllm_coding
)

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
