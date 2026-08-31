#include "kernels/sparse_decoder.cuh"

#include "cuda_memory.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaxGridBlocks = 65535;

template <typename T>
cuda_unique_ptr<T> AllocateDeviceBuffer(std::size_t element_count) {
  T* device_buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&device_buffer, element_count * sizeof(T)));
  return cuda_unique_ptr<T>(device_buffer);
}

unsigned int CalculateGridBlocks(std::size_t word_count) {
  const std::size_t required_blocks =
      word_count / kThreadsPerBlock + (word_count % kThreadsPerBlock != 0);
  const std::size_t capped_blocks =
      std::min(required_blocks, static_cast<std::size_t>(kMaxGridBlocks));
  return static_cast<unsigned int>(std::max<std::size_t>(1, capped_blocks));
}

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

template <std::size_t Degree>
__global__ void xor_reconstruct_kernel_optimized(
    std::uint32_t* const* shard_table, const codedllm::ShardId* source_ids,
    std::uint32_t* output, std::size_t word_count) {
  const std::size_t vector_word_count = word_count / 4;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  // All table entries and output buffers originate from cudaMalloc, which
  // guarantees alignment sufficient for uint4 accesses.
  auto* const vector_output = reinterpret_cast<uint4*>(output);
  for (std::size_t vector_index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vector_index < vector_word_count; vector_index += stride) {
    uint4 value{0, 0, 0, 0};
#pragma unroll
    for (std::size_t source_index = 0; source_index < Degree;
         ++source_index) {
      const auto* const source =
          reinterpret_cast<const uint4*>(shard_table[source_ids[source_index]]);
      const uint4 input = source[vector_index];
      value.x ^= input.x;
      value.y ^= input.y;
      value.z ^= input.z;
      value.w ^= input.w;
    }
    vector_output[vector_index] = value;
  }

  const std::size_t tail_start = vector_word_count * 4;
  const std::size_t tail_count = word_count - tail_start;
  if (blockIdx.x == 0 && threadIdx.x < tail_count) {
    const std::size_t word_index = tail_start + threadIdx.x;
    std::uint32_t value = 0;
#pragma unroll
    for (std::size_t source_index = 0; source_index < Degree;
         ++source_index) {
      value ^= shard_table[source_ids[source_index]][word_index];
    }
    output[word_index] = value;
  }
}

__global__ void xor_reconstruct_kernel_dynamic(
    std::uint32_t* const* shard_table, const codedllm::ShardId* source_ids,
    std::size_t source_count, std::uint32_t* output,
    std::size_t word_count) {
  const std::size_t vector_word_count = word_count / 4;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  // All table entries and output buffers originate from cudaMalloc, which
  // guarantees alignment sufficient for uint4 accesses.
  auto* const vector_output = reinterpret_cast<uint4*>(output);
  for (std::size_t vector_index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vector_index < vector_word_count; vector_index += stride) {
    uint4 value{0, 0, 0, 0};
    for (std::size_t source_index = 0; source_index < source_count;
         ++source_index) {
      const auto* const source =
          reinterpret_cast<const uint4*>(shard_table[source_ids[source_index]]);
      const uint4 input = source[vector_index];
      value.x ^= input.x;
      value.y ^= input.y;
      value.z ^= input.z;
      value.w ^= input.w;
    }
    vector_output[vector_index] = value;
  }

  const std::size_t tail_start = vector_word_count * 4;
  const std::size_t tail_count = word_count - tail_start;
  if (blockIdx.x == 0 && threadIdx.x < tail_count) {
    const std::size_t word_index = tail_start + threadIdx.x;
    std::uint32_t value = 0;
    for (std::size_t source_index = 0; source_index < source_count;
         ++source_index) {
      value ^= shard_table[source_ids[source_index]][word_index];
    }
    output[word_index] = value;
  }
}

void ValidatePlanShape(const codedllm::DecodePlan& plan) {
  if (!plan.is_recoverable) {
    throw std::invalid_argument("Cannot execute an unrecoverable decode plan");
  }
  if (plan.dependencies.size() != plan.operations.size()) {
    throw std::invalid_argument(
        "DecodePlan dependencies must match the operation count");
  }
  for (std::size_t operation_index = 0;
       operation_index < plan.dependencies.size(); ++operation_index) {
    for (const std::size_t dependency : plan.dependencies.at(operation_index)) {
      if (dependency >= operation_index) {
        throw std::invalid_argument(
            "DecodePlan dependencies must reference earlier operations");
      }
    }
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
  cuda_unique_ptr<std::uint32_t> device_a = AllocateDeviceBuffer<std::uint32_t>(size);
  cuda_unique_ptr<std::uint32_t> device_b = AllocateDeviceBuffer<std::uint32_t>(size);
  cuda_unique_ptr<std::uint32_t> device_c = AllocateDeviceBuffer<std::uint32_t>(size);

  CUDA_CHECK(cudaMemcpy(device_a.get(), shard_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b.get(), shard_b, bytes, cudaMemcpyHostToDevice));

  const unsigned int blocks = CalculateGridBlocks(size);
  dummy_reconstruct_kernel<<<blocks, kThreadsPerBlock>>>(device_a.get(),
                                                          device_b.get(),
                                                          device_c.get(), size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(reconstructed_c, device_c.get(), bytes,
                        cudaMemcpyDeviceToHost));
}

void run_decode_plan_cuda(const DecodePlan& plan, coding::ShardSlots& shards) {
  ValidatePlanShape(plan);
  if (plan.operations.empty()) {
    return;
  }

  const std::size_t shard_count = shards.size();
  std::vector<std::optional<std::size_t>> word_counts(shard_count);
  std::vector<bool> required_shards(shard_count, false);
  std::vector<bool> available_shards(shard_count, false);

  for (std::size_t shard_id = 0; shard_id < shard_count; ++shard_id) {
    if (!shards.at(shard_id).has_value()) {
      continue;
    }
    if (shards.at(shard_id)->empty()) {
      throw std::invalid_argument("DecodePlan source shards must be non-empty");
    }
    word_counts.at(shard_id) = shards.at(shard_id)->size();
    available_shards.at(shard_id) = true;
  }

  std::vector<std::size_t> source_offsets;
  std::vector<ShardId> flattened_sources;
  source_offsets.reserve(plan.operations.size());
  for (const XorOperation& operation : plan.operations) {
    if (operation.output >= shard_count) {
      throw std::invalid_argument("DecodePlan output shard is out of bounds");
    }
    if (operation.sources.empty()) {
      throw std::invalid_argument("DecodePlan XOR operation has no sources");
    }
    if (available_shards.at(operation.output)) {
      throw std::invalid_argument("DecodePlan output shard is already present");
    }

    source_offsets.push_back(flattened_sources.size());
    std::optional<std::size_t> operation_word_count;
    for (const ShardId source : operation.sources) {
      if (source >= shard_count || !available_shards.at(source) ||
          !word_counts.at(source).has_value()) {
        throw std::invalid_argument("DecodePlan source shard is unavailable");
      }
      if (!operation_word_count.has_value()) {
        operation_word_count = word_counts.at(source);
      } else if (operation_word_count != word_counts.at(source)) {
        throw std::invalid_argument(
            "DecodePlan sources must be available and equally sized");
      }
      required_shards.at(source) = true;
      flattened_sources.push_back(source);
    }

    word_counts.at(operation.output) = operation_word_count;
    available_shards.at(operation.output) = true;
    required_shards.at(operation.output) = true;
  }

  std::vector<cuda_unique_ptr<std::uint32_t>> device_shards(shard_count);
  std::vector<std::uint32_t*> host_shard_table(shard_count, nullptr);
  for (std::size_t shard_id = 0; shard_id < shard_count; ++shard_id) {
    if (!required_shards.at(shard_id)) {
      continue;
    }
    device_shards.at(shard_id) =
        AllocateDeviceBuffer<std::uint32_t>(*word_counts.at(shard_id));
    host_shard_table.at(shard_id) = device_shards.at(shard_id).get();

    if (shards.at(shard_id).has_value()) {
      const std::size_t bytes =
          *word_counts.at(shard_id) * sizeof(std::uint32_t);
      CUDA_CHECK(cudaMemcpy(device_shards.at(shard_id).get(),
                            shards.at(shard_id)->data(), bytes,
                            cudaMemcpyHostToDevice));
    }
  }

  cuda_unique_ptr<std::uint32_t*> device_shard_table =
      AllocateDeviceBuffer<std::uint32_t*>(host_shard_table.size());
  CUDA_CHECK(cudaMemcpy(device_shard_table.get(), host_shard_table.data(),
                        host_shard_table.size() * sizeof(std::uint32_t*),
                        cudaMemcpyHostToDevice));

  cuda_unique_ptr<ShardId> device_source_ids =
      AllocateDeviceBuffer<ShardId>(flattened_sources.size());
  CUDA_CHECK(cudaMemcpy(device_source_ids.get(), flattened_sources.data(),
                        flattened_sources.size() * sizeof(ShardId),
                        cudaMemcpyHostToDevice));

  for (std::size_t operation_index = 0;
       operation_index < plan.operations.size(); ++operation_index) {
    const XorOperation& operation = plan.operations.at(operation_index);
    const std::size_t word_count = *word_counts.at(operation.output);
    const unsigned int blocks = CalculateGridBlocks(word_count);
    const ShardId* const source_ids =
        device_source_ids.get() + source_offsets.at(operation_index);
    switch (operation.sources.size()) {
      case 2:
        xor_reconstruct_kernel_optimized<2><<<blocks, kThreadsPerBlock>>>(
            device_shard_table.get(), source_ids,
            device_shards.at(operation.output).get(), word_count);
        break;
      case 3:
        xor_reconstruct_kernel_optimized<3><<<blocks, kThreadsPerBlock>>>(
            device_shard_table.get(), source_ids,
            device_shards.at(operation.output).get(), word_count);
        break;
      case 4:
        xor_reconstruct_kernel_optimized<4><<<blocks, kThreadsPerBlock>>>(
            device_shard_table.get(), source_ids,
            device_shards.at(operation.output).get(), word_count);
        break;
      case 5:
        xor_reconstruct_kernel_optimized<5><<<blocks, kThreadsPerBlock>>>(
            device_shard_table.get(), source_ids,
            device_shards.at(operation.output).get(), word_count);
        break;
      case 6:
        xor_reconstruct_kernel_optimized<6><<<blocks, kThreadsPerBlock>>>(
            device_shard_table.get(), source_ids,
            device_shards.at(operation.output).get(), word_count);
        break;
      default:
        xor_reconstruct_kernel_dynamic<<<blocks, kThreadsPerBlock>>>(
            device_shard_table.get(), source_ids, operation.sources.size(),
            device_shards.at(operation.output).get(), word_count);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  for (const XorOperation& operation : plan.operations) {
    const std::size_t word_count = *word_counts.at(operation.output);
    coding::WordShard output(word_count);
    CUDA_CHECK(cudaMemcpy(output.data(), device_shards.at(operation.output).get(),
                          word_count * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    shards.at(operation.output) = std::move(output);
  }
}

}  // namespace codedllm
