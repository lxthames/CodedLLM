#include "kernels/sparse_decoder.cuh"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

TEST(DecoderTest, CompareCpuAndGpu) {
  constexpr std::size_t kShardSize = 4096;
  std::mt19937 generator(42);
  std::uniform_int_distribution<std::uint32_t> distribution;

  std::vector<std::uint32_t> shard_a(kShardSize);
  std::vector<std::uint32_t> shard_b(kShardSize);
  for (std::size_t i = 0; i < kShardSize; ++i) {
    shard_a[i] = distribution(generator);
    shard_b[i] = distribution(generator);
  }

  std::vector<std::uint32_t> cpu_output(kShardSize);
  std::vector<std::uint32_t> gpu_output(kShardSize);
  codedllm::run_sparse_decode_cpu(shard_a.data(), shard_b.data(),
                                  cpu_output.data(), kShardSize);
  codedllm::run_sparse_decode_cuda(shard_a.data(), shard_b.data(),
                                   gpu_output.data(), kShardSize);

  EXPECT_EQ(cpu_output, gpu_output);
}
