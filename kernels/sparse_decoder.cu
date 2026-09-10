#include "kernels/sparse_decoder.cuh"

#include "cuda_memory.cuh"
#include "codedllm/kernels/tuning.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

template <typename T>
cuda_unique_ptr<T> AllocateDeviceBuffer(std::size_t element_count) {
  T* device_buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&device_buffer, element_count * sizeof(T)));
  return cuda_unique_ptr<T>(device_buffer);
}

class CudaEvent {
 public:
  CudaEvent() = default;
  ~CudaEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  CudaEvent(CudaEvent&& other) noexcept
      : event_(std::exchange(other.event_, nullptr)) {}
  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (event_ != nullptr) {
        cudaEventDestroy(event_);
      }
      event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  void Create() {
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreate(&event));
    event_ = event;
  }

  void Record(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(event_, stream));
  }

  [[nodiscard]] cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

std::size_t CalculateGridBlocks(
    std::size_t word_count, std::size_t words_per_thread,
    const codedllm::kernels::DeviceTuningConfig& config) {
  const std::size_t words_per_block =
      static_cast<std::size_t>(config.threads_per_block) * words_per_thread;
  const std::size_t required_blocks =
      word_count / words_per_block + (word_count % words_per_block != 0);
  return std::max<std::size_t>(
      1, std::min<std::size_t>(
             required_blocks,
             static_cast<std::size_t>(config.max_grid_blocks)));
}

__global__ void dummy_reconstruct_kernel(const std::uint32_t* shard_a,
                                         const std::uint32_t* shard_b,
                                         std::uint32_t* reconstructed_c,
                                         std::size_t size) {
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < size; index += stride) {
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

void LaunchXorOperationCuda(
    const codedllm::XorOperation& operation, std::size_t word_count,
    std::uint32_t* const* device_shard_table,
    const codedllm::ShardId* device_source_ids, std::uint32_t* device_output,
    int device_id, cudaStream_t stream) {
  switch (operation.sources.size()) {
    case 2: {
      const codedllm::kernels::DeviceTuningConfig config =
          codedllm::kernels::GetCachedOptimalLaunchConfig(
              xor_reconstruct_kernel_optimized<2>, device_id);
      const std::size_t grid_blocks =
          CalculateGridBlocks(word_count, 4, config);
      xor_reconstruct_kernel_optimized<2>
          <<<grid_blocks, config.threads_per_block, 0, stream>>>(
              device_shard_table, device_source_ids, device_output, word_count);
      break;
    }
    case 3: {
      const codedllm::kernels::DeviceTuningConfig config =
          codedllm::kernels::GetCachedOptimalLaunchConfig(
              xor_reconstruct_kernel_optimized<3>, device_id);
      const std::size_t grid_blocks =
          CalculateGridBlocks(word_count, 4, config);
      xor_reconstruct_kernel_optimized<3>
          <<<grid_blocks, config.threads_per_block, 0, stream>>>(
              device_shard_table, device_source_ids, device_output, word_count);
      break;
    }
    case 4: {
      const codedllm::kernels::DeviceTuningConfig config =
          codedllm::kernels::GetCachedOptimalLaunchConfig(
              xor_reconstruct_kernel_optimized<4>, device_id);
      const std::size_t grid_blocks =
          CalculateGridBlocks(word_count, 4, config);
      xor_reconstruct_kernel_optimized<4>
          <<<grid_blocks, config.threads_per_block, 0, stream>>>(
              device_shard_table, device_source_ids, device_output, word_count);
      break;
    }
    case 5: {
      const codedllm::kernels::DeviceTuningConfig config =
          codedllm::kernels::GetCachedOptimalLaunchConfig(
              xor_reconstruct_kernel_optimized<5>, device_id);
      const std::size_t grid_blocks =
          CalculateGridBlocks(word_count, 4, config);
      xor_reconstruct_kernel_optimized<5>
          <<<grid_blocks, config.threads_per_block, 0, stream>>>(
              device_shard_table, device_source_ids, device_output, word_count);
      break;
    }
    case 6: {
      const codedllm::kernels::DeviceTuningConfig config =
          codedllm::kernels::GetCachedOptimalLaunchConfig(
              xor_reconstruct_kernel_optimized<6>, device_id);
      const std::size_t grid_blocks =
          CalculateGridBlocks(word_count, 4, config);
      xor_reconstruct_kernel_optimized<6>
          <<<grid_blocks, config.threads_per_block, 0, stream>>>(
              device_shard_table, device_source_ids, device_output, word_count);
      break;
    }
    default: {
      const codedllm::kernels::DeviceTuningConfig config =
          codedllm::kernels::GetCachedOptimalLaunchConfig(
              xor_reconstruct_kernel_dynamic, device_id);
      const std::size_t grid_blocks =
          CalculateGridBlocks(word_count, 4, config);
      xor_reconstruct_kernel_dynamic
          <<<grid_blocks, config.threads_per_block, 0, stream>>>(
              device_shard_table, device_source_ids, operation.sources.size(),
              device_output, word_count);
      break;
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

namespace codedllm {

struct DeviceDecodeContext::Impl {
  DecodePlan plan;
  std::vector<std::optional<std::size_t>> word_counts;
  std::vector<std::size_t> source_offsets;
  std::vector<cuda_unique_ptr<std::uint32_t>> device_shards;
  cuda_unique_ptr<std::uint32_t*> device_shard_table;
  cuda_unique_ptr<ShardId> device_source_ids;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  int device_id = 0;

  ~Impl() {
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

DeviceDecodeContext::DeviceDecodeContext() : impl_(std::make_unique<Impl>()) {}
DeviceDecodeContext::~DeviceDecodeContext() = default;
DeviceDecodeContext::DeviceDecodeContext(DeviceDecodeContext&&) noexcept =
    default;
DeviceDecodeContext& DeviceDecodeContext::operator=(
    DeviceDecodeContext&&) noexcept = default;

std::unique_ptr<DeviceDecodeContext> prepare_decode_plan_cuda(
    const DecodePlan& plan, const coding::ShardSlots& shards) {
  ValidatePlanShape(plan);

  std::unique_ptr<DeviceDecodeContext> context(new DeviceDecodeContext());
  DeviceDecodeContext::Impl& impl = *context->impl_;
  impl.plan = plan;
  CUDA_CHECK(cudaGetDevice(&impl.device_id));
  CUDA_CHECK(cudaStreamCreateWithFlags(&impl.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&impl.start));
  CUDA_CHECK(cudaEventCreate(&impl.stop));

  const std::size_t shard_count = shards.size();
  impl.word_counts.resize(shard_count);
  std::vector<bool> required_shards(shard_count, false);
  std::vector<bool> available_shards(shard_count, false);

  for (std::size_t shard_id = 0; shard_id < shard_count; ++shard_id) {
    if (!shards.at(shard_id).has_value()) {
      continue;
    }
    if (shards.at(shard_id)->empty()) {
      throw std::invalid_argument("DecodePlan source shards must be non-empty");
    }
    impl.word_counts.at(shard_id) = shards.at(shard_id)->size();
    available_shards.at(shard_id) = true;
  }

  std::vector<ShardId> flattened_sources;
  impl.source_offsets.reserve(plan.operations.size());
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

    impl.source_offsets.push_back(flattened_sources.size());
    std::optional<std::size_t> operation_word_count;
    for (const ShardId source : operation.sources) {
      if (source >= shard_count || !available_shards.at(source) ||
          !impl.word_counts.at(source).has_value()) {
        throw std::invalid_argument("DecodePlan source shard is unavailable");
      }
      if (!operation_word_count.has_value()) {
        operation_word_count = impl.word_counts.at(source);
      } else if (operation_word_count != impl.word_counts.at(source)) {
        throw std::invalid_argument(
            "DecodePlan sources must be available and equally sized");
      }
      required_shards.at(source) = true;
      flattened_sources.push_back(source);
    }

    impl.word_counts.at(operation.output) = operation_word_count;
    available_shards.at(operation.output) = true;
    required_shards.at(operation.output) = true;
  }

  impl.device_shards.resize(shard_count);
  std::vector<std::uint32_t*> host_shard_table(shard_count, nullptr);
  for (std::size_t shard_id = 0; shard_id < shard_count; ++shard_id) {
    if (!required_shards.at(shard_id)) {
      continue;
    }
    impl.device_shards.at(shard_id) =
        AllocateDeviceBuffer<std::uint32_t>(*impl.word_counts.at(shard_id));
    host_shard_table.at(shard_id) = impl.device_shards.at(shard_id).get();
    if (shards.at(shard_id).has_value()) {
      CUDA_CHECK(cudaMemcpyAsync(
          impl.device_shards.at(shard_id).get(), shards.at(shard_id)->data(),
          *impl.word_counts.at(shard_id) * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, impl.stream));
    }
  }

  if (!plan.operations.empty()) {
    impl.device_shard_table =
        AllocateDeviceBuffer<std::uint32_t*>(host_shard_table.size());
    CUDA_CHECK(cudaMemcpyAsync(
        impl.device_shard_table.get(), host_shard_table.data(),
        host_shard_table.size() * sizeof(std::uint32_t*),
        cudaMemcpyHostToDevice, impl.stream));

    impl.device_source_ids =
        AllocateDeviceBuffer<ShardId>(flattened_sources.size());
    CUDA_CHECK(cudaMemcpyAsync(
        impl.device_source_ids.get(), flattened_sources.data(),
        flattened_sources.size() * sizeof(ShardId), cudaMemcpyHostToDevice,
        impl.stream));
  }

  CUDA_CHECK(cudaStreamSynchronize(impl.stream));
  return context;
}

void launch_decode_plan_cuda(DeviceDecodeContext& context) {
  DeviceDecodeContext::Impl& impl = *context.impl_;
  CUDA_CHECK(cudaSetDevice(impl.device_id));
  for (std::size_t operation_index = 0;
       operation_index < impl.plan.operations.size(); ++operation_index) {
    const XorOperation& operation = impl.plan.operations.at(operation_index);
    LaunchXorOperationCuda(
        operation, *impl.word_counts.at(operation.output),
        impl.device_shard_table.get(),
        impl.device_source_ids.get() + impl.source_offsets.at(operation_index),
        impl.device_shards.at(operation.output).get(), impl.device_id,
        impl.stream);
  }
}

float launch_decode_plan_cuda_timed(DeviceDecodeContext& context) {
  DeviceDecodeContext::Impl& impl = *context.impl_;
  CUDA_CHECK(cudaSetDevice(impl.device_id));
  CUDA_CHECK(cudaEventRecord(impl.start, impl.stream));
  launch_decode_plan_cuda(context);
  CUDA_CHECK(cudaEventRecord(impl.stop, impl.stream));
  CUDA_CHECK(cudaEventSynchronize(impl.stop));

  float elapsed_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, impl.start, impl.stop));
  return elapsed_ms;
}

void synchronize_decode_plan_cuda(DeviceDecodeContext& context) {
  DeviceDecodeContext::Impl& impl = *context.impl_;
  CUDA_CHECK(cudaSetDevice(impl.device_id));
  CUDA_CHECK(cudaStreamSynchronize(impl.stream));
}

void download_decode_results_cuda(DeviceDecodeContext& context,
                                  coding::ShardSlots& shards) {
  DeviceDecodeContext::Impl& impl = *context.impl_;
  if (shards.size() != impl.word_counts.size()) {
    throw std::invalid_argument(
        "Destination shard slots must match the prepared context");
  }

  CUDA_CHECK(cudaSetDevice(impl.device_id));
  std::vector<coding::WordShard> outputs;
  outputs.reserve(impl.plan.operations.size());
  for (const XorOperation& operation : impl.plan.operations) {
    const std::size_t word_count = *impl.word_counts.at(operation.output);
    outputs.emplace_back(word_count);
    CUDA_CHECK(cudaMemcpyAsync(
        outputs.back().data(), impl.device_shards.at(operation.output).get(),
        word_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
        impl.stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(impl.stream));

  for (std::size_t operation_index = 0;
       operation_index < impl.plan.operations.size(); ++operation_index) {
    shards.at(impl.plan.operations.at(operation_index).output) =
        std::move(outputs.at(operation_index));
  }
}

struct DeviceShardStagingContext::Impl {
  struct StagedTransfer {
    CudaEvent start;
    CudaEvent stop;
  };

  std::vector<std::optional<std::size_t>> word_counts;
  std::vector<std::optional<coding::WordShard>> staged_shards;
  std::vector<cuda_unique_ptr<std::uint32_t>> device_shards;
  std::vector<StagedTransfer> staged_transfers;
  DecodePlan plan;
  std::vector<std::size_t> source_offsets;
  cuda_unique_ptr<std::uint32_t*> device_shard_table;
  cuda_unique_ptr<ShardId> device_source_ids;
  cudaStream_t stream = nullptr;
  int device_id = 0;
  bool finalized = false;

  ~Impl() {
    cudaSetDevice(device_id);
    if (stream != nullptr) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }
};

DeviceShardStagingContext::DeviceShardStagingContext()
    : impl_(std::make_unique<Impl>()) {}
DeviceShardStagingContext::~DeviceShardStagingContext() = default;
DeviceShardStagingContext::DeviceShardStagingContext(
    DeviceShardStagingContext&&) noexcept = default;
DeviceShardStagingContext& DeviceShardStagingContext::operator=(
    DeviceShardStagingContext&&) noexcept = default;

std::unique_ptr<DeviceShardStagingContext>
create_device_shard_staging_context_cuda(std::size_t shard_count) {
  if (shard_count == 0) {
    throw std::invalid_argument("Staged decode context requires shard slots");
  }

  std::unique_ptr<DeviceShardStagingContext> context(
      new DeviceShardStagingContext());
  DeviceShardStagingContext::Impl& impl = *context->impl_;
  impl.word_counts.resize(shard_count);
  impl.staged_shards.resize(shard_count);
  impl.device_shards.resize(shard_count);
  CUDA_CHECK(cudaGetDevice(&impl.device_id));
  CUDA_CHECK(cudaStreamCreateWithFlags(&impl.stream, cudaStreamNonBlocking));
  return context;
}

void stage_decode_shard_cuda(DeviceShardStagingContext& context,
                             ShardId shard_id,
                             const coding::WordShard& shard) {
  DeviceShardStagingContext::Impl& impl = *context.impl_;
  if (impl.finalized) {
    throw std::logic_error("Cannot stage shards after decode plan finalization");
  }
  if (shard_id >= impl.device_shards.size()) {
    throw std::invalid_argument("Staged shard ID is out of bounds");
  }
  if (shard.empty()) {
    throw std::invalid_argument("Staged shards must not be empty");
  }
  if (impl.device_shards.at(shard_id) != nullptr) {
    throw std::invalid_argument("Staged shard is already present");
  }

  CUDA_CHECK(cudaSetDevice(impl.device_id));
  impl.staged_shards.at(shard_id) = shard;
  impl.word_counts.at(shard_id) = shard.size();
  impl.device_shards.at(shard_id) =
      AllocateDeviceBuffer<std::uint32_t>(shard.size());

  DeviceShardStagingContext::Impl::StagedTransfer transfer;
  transfer.start.Create();
  transfer.stop.Create();
  transfer.start.Record(impl.stream);
  CUDA_CHECK(cudaMemcpyAsync(impl.device_shards.at(shard_id).get(),
                             impl.staged_shards.at(shard_id)->data(),
                             shard.size() * sizeof(std::uint32_t),
                             cudaMemcpyHostToDevice, impl.stream));
  transfer.stop.Record(impl.stream);
  impl.staged_transfers.push_back(std::move(transfer));
}

void synchronize_staged_shard_uploads_cuda(DeviceShardStagingContext& context) {
  DeviceShardStagingContext::Impl& impl = *context.impl_;
  CUDA_CHECK(cudaSetDevice(impl.device_id));
  CUDA_CHECK(cudaStreamSynchronize(impl.stream));
}

void run_staged_decode_plan_cuda(DeviceShardStagingContext& context,
                                 const DecodePlan& plan,
                                 coding::ShardSlots& shards,
                                 CudaDecodeTimings& timings) {
  DeviceShardStagingContext::Impl& impl = *context.impl_;
  if (impl.finalized) {
    throw std::logic_error("Staged decode context has already executed a plan");
  }
  if (shards.size() != impl.device_shards.size()) {
    throw std::invalid_argument(
        "Destination shard slots must match the staged context");
  }
  ValidatePlanShape(plan);
  CUDA_CHECK(cudaSetDevice(impl.device_id));

  for (std::size_t shard_id = 0; shard_id < impl.device_shards.size();
       ++shard_id) {
    if (impl.staged_shards.at(shard_id) != shards.at(shard_id)) {
      throw std::invalid_argument(
          "Staged shards must match the supplied host shard slots");
    }
  }

  std::vector<std::optional<std::size_t>> finalized_word_counts =
      impl.word_counts;
  std::vector<bool> required_shards(impl.device_shards.size(), false);
  std::vector<bool> available_shards(impl.device_shards.size(), false);
  for (std::size_t shard_id = 0; shard_id < impl.device_shards.size();
       ++shard_id) {
    available_shards.at(shard_id) =
        finalized_word_counts.at(shard_id).has_value();
  }

  std::vector<ShardId> flattened_sources;
  std::vector<std::size_t> source_offsets;
  source_offsets.reserve(plan.operations.size());
  for (const XorOperation& operation : plan.operations) {
    if (operation.output >= impl.device_shards.size()) {
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
      if (source >= impl.device_shards.size() ||
          !available_shards.at(source) ||
          !finalized_word_counts.at(source).has_value()) {
        throw std::invalid_argument("DecodePlan source shard is unavailable");
      }
      if (!operation_word_count.has_value()) {
        operation_word_count = finalized_word_counts.at(source);
      } else if (operation_word_count != finalized_word_counts.at(source)) {
        throw std::invalid_argument(
            "DecodePlan sources must be available and equally sized");
      }
      required_shards.at(source) = true;
      flattened_sources.push_back(source);
    }

    finalized_word_counts.at(operation.output) = operation_word_count;
    available_shards.at(operation.output) = true;
    required_shards.at(operation.output) = true;
  }

  std::vector<std::uint32_t*> host_shard_table(impl.device_shards.size(),
                                                nullptr);
  for (std::size_t shard_id = 0; shard_id < impl.device_shards.size();
       ++shard_id) {
    if (!required_shards.at(shard_id)) {
      continue;
    }
    if (impl.device_shards.at(shard_id) == nullptr) {
      impl.device_shards.at(shard_id) = AllocateDeviceBuffer<std::uint32_t>(
          *finalized_word_counts.at(shard_id));
    }
    host_shard_table.at(shard_id) = impl.device_shards.at(shard_id).get();
  }

  if (!plan.operations.empty()) {
    impl.device_shard_table =
        AllocateDeviceBuffer<std::uint32_t*>(host_shard_table.size());
    CUDA_CHECK(cudaMemcpyAsync(
        impl.device_shard_table.get(), host_shard_table.data(),
        host_shard_table.size() * sizeof(std::uint32_t*), cudaMemcpyHostToDevice,
        impl.stream));
    impl.device_source_ids =
        AllocateDeviceBuffer<ShardId>(flattened_sources.size());
    CUDA_CHECK(cudaMemcpyAsync(impl.device_source_ids.get(),
                               flattened_sources.data(),
                               flattened_sources.size() * sizeof(ShardId),
                               cudaMemcpyHostToDevice, impl.stream));
  }

  impl.word_counts = std::move(finalized_word_counts);
  impl.plan = plan;
  impl.source_offsets = std::move(source_offsets);
  impl.finalized = true;

  CudaEvent kernel_start;
  CudaEvent kernel_stop;
  CudaEvent d2h_start;
  CudaEvent d2h_stop;
  kernel_start.Create();
  kernel_stop.Create();
  d2h_start.Create();
  d2h_stop.Create();

  kernel_start.Record(impl.stream);
  for (std::size_t operation_index = 0;
       operation_index < impl.plan.operations.size(); ++operation_index) {
    const XorOperation& operation = impl.plan.operations.at(operation_index);
    LaunchXorOperationCuda(
        operation, *impl.word_counts.at(operation.output),
        impl.device_shard_table.get(),
        impl.device_source_ids.get() + impl.source_offsets.at(operation_index),
        impl.device_shards.at(operation.output).get(), impl.device_id,
        impl.stream);
  }
  kernel_stop.Record(impl.stream);

  std::vector<coding::WordShard> outputs;
  outputs.reserve(impl.plan.operations.size());
  d2h_start.Record(impl.stream);
  for (const XorOperation& operation : impl.plan.operations) {
    const std::size_t word_count = *impl.word_counts.at(operation.output);
    outputs.emplace_back(word_count);
    CUDA_CHECK(cudaMemcpyAsync(outputs.back().data(),
                               impl.device_shards.at(operation.output).get(),
                               word_count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, impl.stream));
  }
  d2h_stop.Record(impl.stream);
  CUDA_CHECK(cudaStreamSynchronize(impl.stream));

  timings = {};
  for (const DeviceShardStagingContext::Impl::StagedTransfer& transfer :
       impl.staged_transfers) {
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, transfer.start.get(),
                                    transfer.stop.get()));
    timings.h2d_ms += elapsed_ms;
  }
  impl.staged_transfers.clear();
  CUDA_CHECK(cudaEventElapsedTime(&timings.kernel_ms, kernel_start.get(),
                                  kernel_stop.get()));
  CUDA_CHECK(cudaEventElapsedTime(&timings.d2h_ms, d2h_start.get(),
                                  d2h_stop.get()));

  for (std::size_t operation_index = 0;
       operation_index < impl.plan.operations.size(); ++operation_index) {
    shards.at(impl.plan.operations.at(operation_index).output) =
        std::move(outputs.at(operation_index));
  }
  impl.staged_shards.clear();
}

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

  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  const kernels::DeviceTuningConfig config =
      kernels::GetCachedOptimalLaunchConfig(dummy_reconstruct_kernel, device_id);
  const std::size_t grid_blocks = CalculateGridBlocks(size, 1, config);
  dummy_reconstruct_kernel<<<grid_blocks, config.threads_per_block>>>(
      device_a.get(), device_b.get(), device_c.get(), size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(reconstructed_c, device_c.get(), bytes,
                        cudaMemcpyDeviceToHost));
}

void RunDecodePlanCuda(const DecodePlan& plan, coding::ShardSlots& shards,
                       CudaDecodeTimings* timings) {
  ValidatePlanShape(plan);
  if (plan.operations.empty()) {
    return;
  }

  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));

  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  cudaEvent_t d2h_start = nullptr;
  cudaEvent_t d2h_stop = nullptr;
  if (timings != nullptr) {
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));
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
  }

  if (timings != nullptr) {
    CUDA_CHECK(cudaEventRecord(h2d_start));
  }
  for (std::size_t shard_id = 0; shard_id < shard_count; ++shard_id) {
    if (!required_shards.at(shard_id)) {
      continue;
    }
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

  if (timings != nullptr) {
    CUDA_CHECK(cudaEventRecord(h2d_stop));
    CUDA_CHECK(cudaEventRecord(kernel_start));
  }

  for (std::size_t operation_index = 0;
       operation_index < plan.operations.size(); ++operation_index) {
    const XorOperation& operation = plan.operations.at(operation_index);
    const std::size_t word_count = *word_counts.at(operation.output);
    const ShardId* const source_ids =
        device_source_ids.get() + source_offsets.at(operation_index);
    switch (operation.sources.size()) {
      case 2: {
        const kernels::DeviceTuningConfig config =
            kernels::GetCachedOptimalLaunchConfig(
                xor_reconstruct_kernel_optimized<2>, device_id);
        const std::size_t grid_blocks =
            CalculateGridBlocks(word_count, 4, config);
        xor_reconstruct_kernel_optimized<2>
            <<<grid_blocks, config.threads_per_block>>>(
            device_shard_table.get(), source_ids,
            device_shards.at(operation.output).get(), word_count);
        break;
      }
      case 3: {
        const kernels::DeviceTuningConfig config =
            kernels::GetCachedOptimalLaunchConfig(
                xor_reconstruct_kernel_optimized<3>, device_id);
        const std::size_t grid_blocks =
            CalculateGridBlocks(word_count, 4, config);
        xor_reconstruct_kernel_optimized<3>
            <<<grid_blocks, config.threads_per_block>>>(
            device_shard_table.get(), source_ids,
            device_shards.at(operation.output).get(), word_count);
        break;
      }
      case 4: {
        const kernels::DeviceTuningConfig config =
            kernels::GetCachedOptimalLaunchConfig(
                xor_reconstruct_kernel_optimized<4>, device_id);
        const std::size_t grid_blocks =
            CalculateGridBlocks(word_count, 4, config);
        xor_reconstruct_kernel_optimized<4>
            <<<grid_blocks, config.threads_per_block>>>(
            device_shard_table.get(), source_ids,
            device_shards.at(operation.output).get(), word_count);
        break;
      }
      case 5: {
        const kernels::DeviceTuningConfig config =
            kernels::GetCachedOptimalLaunchConfig(
                xor_reconstruct_kernel_optimized<5>, device_id);
        const std::size_t grid_blocks =
            CalculateGridBlocks(word_count, 4, config);
        xor_reconstruct_kernel_optimized<5>
            <<<grid_blocks, config.threads_per_block>>>(
            device_shard_table.get(), source_ids,
            device_shards.at(operation.output).get(), word_count);
        break;
      }
      case 6: {
        const kernels::DeviceTuningConfig config =
            kernels::GetCachedOptimalLaunchConfig(
                xor_reconstruct_kernel_optimized<6>, device_id);
        const std::size_t grid_blocks =
            CalculateGridBlocks(word_count, 4, config);
        xor_reconstruct_kernel_optimized<6>
            <<<grid_blocks, config.threads_per_block>>>(
            device_shard_table.get(), source_ids,
            device_shards.at(operation.output).get(), word_count);
        break;
      }
      default: {
        const kernels::DeviceTuningConfig config =
            kernels::GetCachedOptimalLaunchConfig(
                xor_reconstruct_kernel_dynamic, device_id);
        const std::size_t grid_blocks =
            CalculateGridBlocks(word_count, 4, config);
        xor_reconstruct_kernel_dynamic
            <<<grid_blocks, config.threads_per_block>>>(
            device_shard_table.get(), source_ids, operation.sources.size(),
            device_shards.at(operation.output).get(), word_count);
        break;
      }
    }
    CUDA_CHECK(cudaGetLastError());
  }

  if (timings != nullptr) {
    CUDA_CHECK(cudaEventRecord(kernel_stop));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  if (timings != nullptr) {
    CUDA_CHECK(cudaEventRecord(d2h_start));
  }
  for (const XorOperation& operation : plan.operations) {
    const std::size_t word_count = *word_counts.at(operation.output);
    coding::WordShard output(word_count);
    CUDA_CHECK(cudaMemcpy(output.data(), device_shards.at(operation.output).get(),
                          word_count * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    shards.at(operation.output) = std::move(output);
  }

  if (timings != nullptr) {
    CUDA_CHECK(cudaEventRecord(d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(d2h_stop));
    CUDA_CHECK(cudaEventElapsedTime(&timings->h2d_ms, h2d_start, h2d_stop));
    CUDA_CHECK(
        cudaEventElapsedTime(&timings->kernel_ms, kernel_start, kernel_stop));
    CUDA_CHECK(cudaEventElapsedTime(&timings->d2h_ms, d2h_start, d2h_stop));
    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));
  }
}

void run_decode_plan_cuda(const DecodePlan& plan, coding::ShardSlots& shards) {
  RunDecodePlanCuda(plan, shards, nullptr);
}

void run_decode_plan_cuda_profiled(const DecodePlan& plan,
                                   coding::ShardSlots& shards,
                                   CudaDecodeTimings& timings) {
  timings = {};
  RunDecodePlanCuda(plan, shards, &timings);
}

}  // namespace codedllm
