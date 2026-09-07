#pragma once

#include "codedllm/coding/codec.hpp"
#include "codedllm/coding/graph_generator.hpp"
#include "codedllm/coding/planner.hpp"
#include "codedllm/runtime/policy.hpp"

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace codedllm::runtime {

using RequestId = std::size_t;

struct RecoveryExecutionMetrics {
  std::chrono::microseconds planning{};
  std::chrono::microseconds queue_wait{};
  std::chrono::microseconds h2d{};
  std::chrono::microseconds kernel{};
  std::chrono::microseconds d2h{};
  std::chrono::microseconds handoff{};
  std::chrono::microseconds total{};
};

struct RecoveryTaskResult {
  bool success = false;
  coding::ShardSlots shards;
  RecoveryExecutionMetrics metrics;
  std::string error;
};

class RecoveryExecutor {
public:
  using CompletionHandler = std::function<void(RecoveryTaskResult)>;

  virtual ~RecoveryExecutor() = default;
  [[nodiscard]] virtual std::optional<std::chrono::microseconds>
  EstimateRecoveryCost() const = 0;
  virtual bool Submit(RequestId request_id, DecodePlan plan, coding::ShardSlots shards,
                      CompletionHandler completion) = 0;
  virtual bool Cancel(RequestId request_id) = 0;
};

class ShardTransport {
public:
  using ArrivalHandler = std::function<void(
      RequestId request_id, ShardId shard_id, coding::WordShard payload,
      std::chrono::steady_clock::time_point arrival_time)>;

  virtual ~ShardTransport() = default;
  virtual void SetArrivalHandler(ArrivalHandler handler) = 0;
  virtual void StartRequest(RequestId request_id) = 0;
};

enum class RequestStatus { Pending, NaturalComplete, Recovered, Failed };

struct RequestSnapshot {
  RequestStatus status = RequestStatus::Pending;
  bool recovery_submitted = false;
  bool recovery_rejected = false;
  std::size_t late_arrivals = 0;
  RecoveryExecutionMetrics metrics;
  std::string error;
};

class RecoveryCoordinator {
public:
  using WaitCostOracle = std::function<std::chrono::microseconds(
      RequestId request_id, std::chrono::steady_clock::time_point now)>;

  RecoveryCoordinator(coding::BipartiteGraph graph, coding::DecodePlanner planner,
                      std::shared_ptr<RecoveryExecutor> executor);
  ~RecoveryCoordinator();

  RecoveryCoordinator(const RecoveryCoordinator&) = delete;
  RecoveryCoordinator& operator=(const RecoveryCoordinator&) = delete;

  void AttachTransport(ShardTransport& transport);
  void StartRequest(RequestId request_id, WaitCostOracle wait_cost_oracle);

  [[nodiscard]] RequestSnapshot GetSnapshot(RequestId request_id) const;
  [[nodiscard]] std::optional<coding::WordShard> GetShard(RequestId request_id,
                                                          ShardId shard_id) const;
  [[nodiscard]] bool WaitForCompletion(RequestId request_id,
                                       std::chrono::milliseconds timeout) const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace codedllm::runtime
