#include "codedllm/runtime/policy.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace codedllm::runtime {

ShardArrivalTracker::ShardArrivalTracker(std::size_t systematic_shard_count,
                                         std::size_t parity_shard_count)
    : systematic_shard_count_(systematic_shard_count),
      parity_shard_count_(parity_shard_count) {
  if (systematic_shard_count == 0 || parity_shard_count == 0) {
    throw std::invalid_argument("ShardArrivalTracker requires non-zero shard counts");
  }

  const std::uint64_t total_shard_count =
      static_cast<std::uint64_t>(systematic_shard_count) +
      static_cast<std::uint64_t>(parity_shard_count);
  if (systematic_shard_count > std::numeric_limits<ShardId>::max() ||
      parity_shard_count > std::numeric_limits<ShardId>::max() ||
      total_shard_count >
          static_cast<std::uint64_t>(std::numeric_limits<ShardId>::max()) + 1U) {
    throw std::invalid_argument("ShardArrivalTracker shard IDs overflow ShardId");
  }

  arrivals_.resize(static_cast<std::size_t>(total_shard_count));
}

void ShardArrivalTracker::MarkArrived(ShardId id,
                                      std::chrono::steady_clock::time_point now) {
  if (id >= arrivals_.size()) {
    throw std::invalid_argument("Arrived shard ID is out of bounds");
  }

  std::optional<std::chrono::steady_clock::time_point>& recorded = arrivals_.at(id);
  if (!recorded.has_value() || now < *recorded) {
    recorded = now;
  }
}

bool ShardArrivalTracker::IsSystematicComplete() const {
  for (std::size_t id = 0; id < systematic_shard_count_; ++id) {
    if (!arrivals_.at(id).has_value()) {
      return false;
    }
  }
  return true;
}

std::unordered_set<ShardId> ShardArrivalTracker::GetMissingSystematic() const {
  std::unordered_set<ShardId> missing;
  for (std::size_t id = 0; id < systematic_shard_count_; ++id) {
    if (!arrivals_.at(id).has_value()) {
      missing.insert(static_cast<ShardId>(id));
    }
  }
  return missing;
}

bool ShardArrivalTracker::HasArrived(ShardId id) const {
  if (id >= arrivals_.size()) {
    throw std::invalid_argument("Shard ID is out of bounds");
  }
  return arrivals_.at(id).has_value();
}

RecoveryPolicy::RecoveryPolicy(coding::BipartiteGraph graph,
                               coding::DecodePlanner planner)
    : graph_(std::move(graph)), planner_(std::move(planner)) {
  static_cast<void>(planner_.CreatePlan(graph_, {}));
}

DecodePlan RecoveryPolicy::CreatePlan(const ShardArrivalTracker& tracker) const {
  if (tracker.systematic_shard_count() != graph_.systematic_shard_count ||
      tracker.parity_shard_count() != graph_.parity_shard_count) {
    throw std::invalid_argument(
        "ShardArrivalTracker dimensions do not match the recovery graph");
  }

  std::unordered_set<ShardId> available_parity;
  for (const auto& [parity, systematic] : graph_.parity_to_systematic) {
    static_cast<void>(systematic);
    if (tracker.HasArrived(parity)) {
      available_parity.insert(parity);
    }
  }
  return planner_.CreatePlan(graph_, tracker.GetMissingSystematic(), available_parity);
}

PolicyDecision RecoveryPolicy::Evaluate(const DecodePlan& plan,
                                        const CostEstimates& costs) const {
  if (!plan.is_recoverable) {
    return PolicyDecision::Unrecoverable;
  }
  return costs.expected_recovery_cost < costs.expected_wait_remaining
             ? PolicyDecision::Recover
             : PolicyDecision::Wait;
}

PolicyDecision RecoveryPolicy::Evaluate(const ShardArrivalTracker& tracker,
                                        const CostEstimates& costs) const {
  if (tracker.IsSystematicComplete()) {
    return PolicyDecision::Wait;
  }
  return Evaluate(CreatePlan(tracker), costs);
}

} // namespace codedllm::runtime
