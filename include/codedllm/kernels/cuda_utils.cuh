#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t cuda_check_error = (call);                                \
    if (cuda_check_error != cudaSuccess) {                                      \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) + ": " +               \
                               cudaGetErrorString(cuda_check_error));           \
    }                                                                           \
  } while (false)
