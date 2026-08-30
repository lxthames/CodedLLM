#include "kernels/sparse_decoder.cuh"

#include "cuda_utils.cuh"

#include <cstddef>
#include <cstdint>

namespace {

__global__ void dummy_reconstruct_kernel(const std::uint32_t* shard_a,
                                         const std::uint32_t* shard_b,
                                         std::uint32_t* reconstructed_c,
                                         std::size_t size) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) {
    reconstructed_c[index] = shard_a[index] ^ shard_b[index];
  }
}

}  // namespace

namespace codedllm {

void run_sparse_decode_cuda(const std::uint32_t* shard_a,
                            const std::uint32_t* shard_b,
                            std::uint32_t* reconstructed_c, std::size_t size) {
  if (size == 0) {
    return;
  }

  const std::size_t bytes = size * sizeof(std::uint32_t);
  std::uint32_t* device_a = nullptr;
  std::uint32_t* device_b = nullptr;
  std::uint32_t* device_c = nullptr;

  try {
    CUDA_CHECK(cudaMalloc(&device_a, bytes));
    CUDA_CHECK(cudaMalloc(&device_b, bytes));
    CUDA_CHECK(cudaMalloc(&device_c, bytes));
    CUDA_CHECK(cudaMemcpy(device_a, shard_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, shard_b, bytes, cudaMemcpyHostToDevice));

    constexpr unsigned int kThreadsPerBlock = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (size + kThreadsPerBlock - 1) / kThreadsPerBlock);
    dummy_reconstruct_kernel<<<blocks, kThreadsPerBlock>>>(device_a, device_b,
                                                            device_c, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(reconstructed_c, device_c, bytes, cudaMemcpyDeviceToHost));
  } catch (...) {
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    throw;
  }

  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));
}

}  // namespace codedllm
