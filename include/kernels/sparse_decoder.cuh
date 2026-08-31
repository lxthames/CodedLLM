#pragma once

#include "codedllm/coding/codec.hpp"

#include <cstddef>
#include <cstdint>

namespace codedllm {

// CPU reference implementation for the dummy shard reconstruction operation.
void run_sparse_decode_cpu(const std::uint32_t* shard_a,
                           const std::uint32_t* shard_b,
                           std::uint32_t* reconstructed_c, std::size_t size);

// CUDA implementation. Memory ownership remains with the caller: all buffers
// passed here are host buffers, while this function manages temporary device memory.
void run_sparse_decode_cuda(const std::uint32_t* shard_a,
                            const std::uint32_t* shard_b,
                            std::uint32_t* reconstructed_c, std::size_t size);

// Executes a recoverable DecodePlan over host-resident word shards.
void run_decode_plan_cuda(const DecodePlan& plan, coding::ShardSlots& shards);

}  // namespace codedllm
