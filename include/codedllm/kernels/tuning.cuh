#pragma once

#include "codedllm/kernels/cuda_utils.cuh"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace codedllm::kernels {

struct DeviceTuningConfig {
  int threads_per_block;
  int blocks_per_sm;
  int max_grid_blocks;
  std::size_t vectorization_threshold_words;
};

template <typename KernelFunc>
DeviceTuningConfig GetOptimalLaunchConfig(KernelFunc kernel, int device_id = 0) {
  int previous_device = 0;
  CUDA_CHECK(cudaGetDevice(&previous_device));
  if (previous_device != device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
  }

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));
  if (properties.multiProcessorCount <= 0) {
    throw std::runtime_error("CUDA device reports no streaming multiprocessors");
  }

  int min_grid_size = 0;
  int block_size = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &block_size, kernel, /*dynamic_smem_size=*/0,
      /*block_size_limit=*/0));

  // Ceiling division prevents a fractional recommendation from becoming zero.
  const int potential_blocks_per_sm = std::max(
      1, (min_grid_size + properties.multiProcessorCount - 1) /
             properties.multiProcessorCount);

  // This API gives the actual residency limit for the selected block size.
  int active_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel, block_size, /*dynamic_smem_size=*/0));
  const int blocks_per_sm =
      std::max(1, std::min(potential_blocks_per_sm, active_blocks_per_sm));

  constexpr int kOversubscriptionFactor = 2;
  const long long requested_grid_blocks =
      static_cast<long long>(properties.multiProcessorCount) * blocks_per_sm *
      kOversubscriptionFactor;
  const long long bounded_grid_blocks =
      std::min(requested_grid_blocks,
               static_cast<long long>(properties.maxGridSize[0]));

  if (previous_device != device_id) {
    CUDA_CHECK(cudaSetDevice(previous_device));
  }

  return DeviceTuningConfig{
      block_size,
      blocks_per_sm,
      static_cast<int>(std::min(
          bounded_grid_blocks,
          static_cast<long long>(std::numeric_limits<int>::max()))),
      static_cast<std::size_t>(block_size) * 4};
}

// Cache occupancy results because they depend on the kernel and device, not the
// individual decode request.
template <typename KernelFunc>
DeviceTuningConfig GetCachedOptimalLaunchConfig(KernelFunc kernel,
                                                 int device_id = 0) {
  struct CacheEntry {
    int device_id;
    KernelFunc kernel;
    DeviceTuningConfig config;
  };

  static std::mutex cache_mutex;
  static std::vector<CacheEntry> cache;
  const std::lock_guard<std::mutex> lock(cache_mutex);
  const auto existing = std::find_if(
      cache.begin(), cache.end(), [&](const CacheEntry& entry) {
        return entry.device_id == device_id && entry.kernel == kernel;
      });
  if (existing != cache.end()) {
    return existing->config;
  }

  const DeviceTuningConfig config =
      GetOptimalLaunchConfig(kernel, device_id);
  cache.push_back(CacheEntry{device_id, kernel, config});
  return config;
}

}  // namespace codedllm::kernels
