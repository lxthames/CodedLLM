#include "codedllm/simulation/simulator.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace codedllm::simulation {

RequestSimulation::RequestSimulation(coding::BipartiteGraph graph,
                                     coding::DecodePlanner planner,
                                     std::vector<ShardArrivalEvent> events,
                                     CostOracle get_costs)
    : systematic_shard_count_(graph.systematic_shard_count),
      parity_shard_count_(graph.parity_shard_count),
      policy_(std::move(graph), std::move(planner)), events_(std::move(events)),
      get_costs_(std::move(get_costs)) {
  if (!get_costs_) {
    throw std::invalid_argument("RequestSimulation requires a cost oracle");
  }
}

SimulationMetrics RequestSimulation::Run() const {
  std::vector<ShardArrivalEvent> ordered_events = events_;
  std::stable_sort(ordered_events.begin(), ordered_events.end(),
                   [](const ShardArrivalEvent& left, const ShardArrivalEvent& right) {
                     return left.arrival_time < right.arrival_time;
                   });

  runtime::ShardArrivalTracker tracker(systematic_shard_count_, parity_shard_count_);
  SimulationMetrics metrics;
  std::optional<std::chrono::microseconds> natural_completion_time;

  std::size_t event_index = 0;
  while (event_index < ordered_events.size()) {
    const std::chrono::microseconds current_time =
        ordered_events.at(event_index).arrival_time;
    if (current_time < std::chrono::microseconds::zero()) {
      throw std::invalid_argument("Shard arrival times must not be negative");
    }

    const auto logical_time = std::chrono::steady_clock::time_point{} + current_time;
    while (event_index < ordered_events.size() &&
           ordered_events.at(event_index).arrival_time == current_time) {
      tracker.MarkArrived(ordered_events.at(event_index).shard_id, logical_time);
      ++event_index;
    }

    if (tracker.IsSystematicComplete()) {
      natural_completion_time = current_time;
      break;
    }

    if (!metrics.recovery_start_time.has_value()) {
      const CostEstimates costs = get_costs_(current_time);
      const runtime::PolicyDecision decision = policy_.Evaluate(tracker, costs);
      if (decision == runtime::PolicyDecision::Recover) {
        if (costs.expected_recovery_cost < std::chrono::microseconds::zero()) {
          throw std::invalid_argument("Expected recovery cost must not be negative");
        }
        metrics.recovery_start_time = current_time;
        metrics.recovery_completion_time = current_time + costs.expected_recovery_cost;
      }
    }
  }

  if (!natural_completion_time.has_value()) {
    throw std::runtime_error(
        "Simulation requires a natural arrival for every systematic shard");
  }

  metrics.natural_completion_time = *natural_completion_time;
  metrics.final_latency = metrics.natural_completion_time;
  if (metrics.recovery_completion_time.has_value() &&
      *metrics.recovery_completion_time < metrics.natural_completion_time) {
    metrics.final_latency = *metrics.recovery_completion_time;
    metrics.recovery_won = true;
  }
  return metrics;
}

} // namespace codedllm::simulation
