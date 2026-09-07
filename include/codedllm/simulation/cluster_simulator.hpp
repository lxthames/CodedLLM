#pragma once

#include "codedllm/coding/graph_generator.hpp"
#include "codedllm/coding/planner.hpp"
#include "codedllm/core/types.hpp"
#include "codedllm/runtime/policy.hpp"
#include "codedllm/simulation/recovery_queue.hpp"

#include <chrono>
#include <cstddef>
#include <functional>
#include <map>
#include <vector>

namespace codedllm::simulation {

struct GlobalArrivalEvent {
  std::chrono::microseconds arrival_time;
  std::size_t request_id;
  ShardId shard_id;
  std::chrono::microseconds request_start_time{};

  [[nodiscard]] bool operator==(const GlobalArrivalEvent& other) const noexcept {
    return arrival_time == other.arrival_time && request_id == other.request_id &&
           shard_id == other.shard_id && request_start_time == other.request_start_time;
  }
};

struct RequestMetrics {
  std::chrono::microseconds latency{};
  bool recovery_won = false;
  bool recovery_admitted = false;
  bool recovery_rejected = false;

  [[nodiscard]] bool operator==(const RequestMetrics& other) const noexcept {
    return latency == other.latency && recovery_won == other.recovery_won &&
           recovery_admitted == other.recovery_admitted &&
           recovery_rejected == other.recovery_rejected;
  }
};

class ClusterSimulator {
public:
  using WaitCostOracle = std::function<std::chrono::microseconds(
      std::size_t request_id, std::chrono::microseconds current_time)>;

  ClusterSimulator(coding::BipartiteGraph graph, coding::DecodePlanner planner,
                   RecoveryQueue recovery_queue);

  [[nodiscard]] std::map<std::size_t, RequestMetrics>
  Run(std::vector<GlobalArrivalEvent> events,
      WaitCostOracle get_expected_wait_remaining);

private:
  const std::size_t systematic_shard_count_;
  const std::size_t parity_shard_count_;
  RecoveryQueue recovery_queue_;
  std::map<std::size_t, runtime::ShardArrivalTracker> trackers_;
  const runtime::RecoveryPolicy policy_;
};

} // namespace codedllm::simulation
