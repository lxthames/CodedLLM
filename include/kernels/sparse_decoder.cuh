#pragma once

#include "codedllm/coding/codec.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace codedllm {

struct CudaDecodeTimings {
  float h2d_ms = 0.0F;
  float kernel_ms = 0.0F;
  float d2h_ms = 0.0F;
};

// Owns device-resident shards and decode metadata. The implementation is
// opaque so callers do not need CUDA headers to hold the context.
class DeviceDecodeContext {
 public:
  ~DeviceDecodeContext();
  DeviceDecodeContext(DeviceDecodeContext&&) noexcept;
  DeviceDecodeContext& operator=(DeviceDecodeContext&&) noexcept;

  DeviceDecodeContext(const DeviceDecodeContext&) = delete;
  DeviceDecodeContext& operator=(const DeviceDecodeContext&) = delete;

 private:
  DeviceDecodeContext();

  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend std::unique_ptr<DeviceDecodeContext> prepare_decode_plan_cuda(
      const DecodePlan&, const coding::ShardSlots&);
  friend void launch_decode_plan_cuda(DeviceDecodeContext&);
  friend float launch_decode_plan_cuda_timed(DeviceDecodeContext&);
  friend void synchronize_decode_plan_cuda(DeviceDecodeContext&);
  friend void download_decode_results_cuda(DeviceDecodeContext&,
                                           coding::ShardSlots&);
};

class DeviceShardStagingContext {
 public:
  ~DeviceShardStagingContext();
  DeviceShardStagingContext(DeviceShardStagingContext&&) noexcept;
  DeviceShardStagingContext& operator=(DeviceShardStagingContext&&) noexcept;

  DeviceShardStagingContext(const DeviceShardStagingContext&) = delete;
  DeviceShardStagingContext& operator=(const DeviceShardStagingContext&) = delete;

 private:
  DeviceShardStagingContext();

  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend std::unique_ptr<DeviceShardStagingContext>
  create_device_shard_staging_context_cuda(std::size_t);
  friend void stage_decode_shard_cuda(DeviceShardStagingContext&, ShardId,
                                      const coding::WordShard&);
  friend void synchronize_staged_shard_uploads_cuda(
      DeviceShardStagingContext&);
  friend void run_staged_decode_plan_cuda(DeviceShardStagingContext&,
                                          const DecodePlan&,
                                          coding::ShardSlots&,
                                          CudaDecodeTimings&);
};

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

// Executes the same path while reporting device-side phase timings.
void run_decode_plan_cuda_profiled(const DecodePlan& plan,
                                   coding::ShardSlots& shards,
                                   CudaDecodeTimings& timings);

// Prepares all allocations and H2D metadata transfers once. Subsequent launch
// calls enqueue only the decode kernels on the context's private CUDA stream.
[[nodiscard]] std::unique_ptr<DeviceDecodeContext> prepare_decode_plan_cuda(
    const DecodePlan& plan, const coding::ShardSlots& shards);

void launch_decode_plan_cuda(DeviceDecodeContext& context);
[[nodiscard]] float launch_decode_plan_cuda_timed(
    DeviceDecodeContext& context);
void synchronize_decode_plan_cuda(DeviceDecodeContext& context);
void download_decode_results_cuda(DeviceDecodeContext& context,
                                  coding::ShardSlots& shards);

[[nodiscard]] std::unique_ptr<DeviceShardStagingContext>
create_device_shard_staging_context_cuda(std::size_t shard_count);
void stage_decode_shard_cuda(DeviceShardStagingContext& context, ShardId shard_id,
                             const coding::WordShard& shard);
void synchronize_staged_shard_uploads_cuda(DeviceShardStagingContext& context);
void run_staged_decode_plan_cuda(DeviceShardStagingContext& context,
                                 const DecodePlan& plan,
                                 coding::ShardSlots& shards,
                                 CudaDecodeTimings& timings);

}  // namespace codedllm
