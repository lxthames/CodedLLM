#include "kernels/sparse_decoder.cuh"

namespace codedllm {

void run_sparse_decode_cpu(const std::uint32_t* shard_a,
                           const std::uint32_t* shard_b,
                           std::uint32_t* reconstructed_c, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    reconstructed_c[i] = shard_a[i] ^ shard_b[i];
  }
}

}  // namespace codedllm
