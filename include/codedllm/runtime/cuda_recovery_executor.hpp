#pragma once

#include "codedllm/runtime/serving.hpp"

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>

namespace codedllm::runtime {

// Bounded asynchronous adapter around the frozen CUDA decode backend. This
// public interface deliberately exposes no CUDA types or headers.
class CudaRecoveryExecutor final : public RecoveryExecutor {
public:
  CudaRecoveryExecutor(std::size_t concurrency_limit, std::size_t max_queue_depth,
                       std::chrono::microseconds estimated_service_cost);
  ~CudaRecoveryExecutor() override;

  CudaRecoveryExecutor(const CudaRecoveryExecutor&) = delete;
  CudaRecoveryExecutor& operator=(const CudaRecoveryExecutor&) = delete;

  [[nodiscard]] std::optional<std::chrono::microseconds>
  EstimateRecoveryCost() const override;
  bool Submit(RequestId request_id, DecodePlan plan, coding::ShardSlots shards,
              CompletionHandler completion) override;
  bool Cancel(RequestId request_id) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace codedllm::runtime
