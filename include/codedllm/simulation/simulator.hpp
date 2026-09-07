#pragma once

#include "codedllm/coding/graph_generator.hpp"
#include "codedllm/coding/planner.hpp"
#include "codedllm/core/types.hpp"
#include "codedllm/runtime/policy.hpp"

#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace codedllm::simulation {

using runtime::CostEstimates;

struct ShardArrivalEvent {
  ShardId shard_id;
  std::chrono::microseconds arrival_time;
};

struct SimulationMetrics {
  std::chrono::microseconds natural_completion_time{};
  std::optional<std::chrono::microseconds> recovery_start_time;
  std::optional<std::chrono::microseconds> recovery_completion_time;
  std::chrono::microseconds final_latency{};
  bool recovery_won = false;
};

class RequestSimulation {
public:
  using CostOracle =
      std::function<CostEstimates(std::chrono::microseconds current_time)>;

  RequestSimulation(coding::BipartiteGraph graph, coding::DecodePlanner planner,
                    std::vector<ShardArrivalEvent> events, CostOracle get_costs);

  [[nodiscard]] SimulationMetrics Run() const;

private:
  const std::size_t systematic_shard_count_;
  const std::size_t parity_shard_count_;
  const runtime::RecoveryPolicy policy_;
  const std::vector<ShardArrivalEvent> events_;
  const CostOracle get_costs_;
};

} // namespace codedllm::simulation
