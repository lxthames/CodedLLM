#include "codedllm/simulation/cluster_simulator.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace codedllm::simulation {

ClusterSimulator::ClusterSimulator(
    coding::BipartiteGraph graph, coding::DecodePlanner planner,
    RecoveryQueue recovery_queue,
    RecoveryServiceTimeEstimator estimate_recovery_service_time)
    : systematic_shard_count_(graph.systematic_shard_count),
      parity_shard_count_(graph.parity_shard_count),
      recovery_queue_(std::move(recovery_queue)),
      policy_(std::move(graph), std::move(planner)),
      estimate_recovery_service_time_(std::move(estimate_recovery_service_time)) {
  if (!estimate_recovery_service_time_) {
    throw std::invalid_argument("ClusterSimulator requires a recovery-cost estimator");
  }
}

std::map<std::size_t, RequestMetrics>
ClusterSimulator::Run(std::vector<GlobalArrivalEvent> events,
                      WaitCostOracle get_expected_wait_remaining) {
  if (!get_expected_wait_remaining) {
    throw std::invalid_argument("ClusterSimulator requires a wait-cost oracle");
  }

  std::sort(events.begin(), events.end(),
            [](const GlobalArrivalEvent& left, const GlobalArrivalEvent& right) {
              return std::tie(left.arrival_time, left.request_id, left.shard_id) <
                     std::tie(right.arrival_time, right.request_id, right.shard_id);
            });

  trackers_.clear();
  std::map<std::size_t, std::chrono::microseconds> request_start_times;
  for (const GlobalArrivalEvent& event : events) {
    trackers_.try_emplace(event.request_id, systematic_shard_count_,
                          parity_shard_count_);
    const auto [start, inserted] =
        request_start_times.emplace(event.request_id, event.request_start_time);
    if (!inserted && start->second != event.request_start_time) {
      throw std::invalid_argument(
          "Events for one request must share a request start time");
    }
    if (event.arrival_time < event.request_start_time) {
      throw std::invalid_argument("Shard arrival must not precede its request start");
    }
  }

  std::map<std::size_t, std::chrono::microseconds> natural_completion_times;
  std::map<std::size_t, std::chrono::microseconds> recovery_completion_times;
  std::map<std::size_t, bool> recovery_rejected;
  std::map<std::size_t, bool> recovery_waited;
  std::map<std::size_t, bool> recovery_unrecoverable;

  std::size_t event_index = 0;
  while (event_index < events.size()) {
    const std::chrono::microseconds current_time = events.at(event_index).arrival_time;
    if (current_time < std::chrono::microseconds::zero()) {
      throw std::invalid_argument("Shard arrival times must not be negative");
    }
    recovery_queue_.AdvanceTime(current_time);

    while (event_index < events.size() &&
           events.at(event_index).arrival_time == current_time) {
      const std::size_t request_id = events.at(event_index).request_id;
      runtime::ShardArrivalTracker& tracker = trackers_.at(request_id);
      const auto logical_time = std::chrono::steady_clock::time_point{} + current_time;

      while (event_index < events.size() &&
             events.at(event_index).arrival_time == current_time &&
             events.at(event_index).request_id == request_id) {
        tracker.MarkArrived(events.at(event_index).shard_id, logical_time);
        ++event_index;
      }

      if (tracker.IsSystematicComplete()) {
        natural_completion_times.try_emplace(request_id, current_time);
        continue;
      }
      if (recovery_completion_times.count(request_id) != 0) {
        continue;
      }

      const DecodePlan plan = policy_.CreatePlan(tracker);
      if (!plan.is_recoverable) {
        recovery_unrecoverable[request_id] = true;
        continue;
      }

      const std::chrono::microseconds service_time =
          estimate_recovery_service_time_(plan);
      const std::optional<std::chrono::microseconds> recovery_cost =
          recovery_queue_.GetExpectedRecoveryCost(current_time, service_time);
      if (!recovery_cost.has_value()) {
        recovery_rejected[request_id] = true;
        continue;
      }

      const std::chrono::microseconds wait_cost =
          get_expected_wait_remaining(request_id, current_time);
      if (wait_cost < std::chrono::microseconds::zero()) {
        throw std::invalid_argument("Expected wait cost must not be negative");
      }
      const runtime::PolicyDecision decision =
          policy_.Evaluate(plan, runtime::CostEstimates{wait_cost, *recovery_cost});
      if (decision == runtime::PolicyDecision::Recover) {
        if (!recovery_queue_.SubmitJob(current_time, service_time)) {
          recovery_rejected[request_id] = true;
          continue;
        }
        recovery_completion_times.emplace(request_id, current_time + *recovery_cost);
      } else if (decision == runtime::PolicyDecision::Wait) {
        recovery_waited[request_id] = true;
      } else {
        recovery_unrecoverable[request_id] = true;
      }
    }
  }

  std::map<std::size_t, RequestMetrics> metrics;
  for (const auto& [request_id, tracker] : trackers_) {
    static_cast<void>(tracker);
    const auto natural = natural_completion_times.find(request_id);
    if (natural == natural_completion_times.end()) {
      throw std::runtime_error(
          "Simulation requires a natural arrival for every systematic shard");
    }

    RequestMetrics request_metrics;
    const std::chrono::microseconds request_start = request_start_times.at(request_id);
    request_metrics.latency = natural->second - request_start;
    request_metrics.recovery_admitted =
        recovery_completion_times.count(request_id) != 0;
    request_metrics.recovery_rejected = recovery_rejected[request_id];
    request_metrics.recovery_waited = recovery_waited[request_id];
    request_metrics.recovery_unrecoverable = recovery_unrecoverable[request_id];
    const auto recovery = recovery_completion_times.find(request_id);
    if (recovery != recovery_completion_times.end() &&
        recovery->second < natural->second) {
      request_metrics.latency = recovery->second - request_start;
      request_metrics.recovery_won = true;
    }
    metrics.emplace(request_id, request_metrics);
  }
  return metrics;
}

} // namespace codedllm::simulation
