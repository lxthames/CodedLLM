#pragma once

#include <memory>

#include "cuda_utils.cuh"

struct CudaDeleter {
  template <typename T>
  void operator()(T* ptr) const {
    if (ptr != nullptr) {
      CUDA_CHECK(cudaFree(ptr));
    }
  }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter>;
