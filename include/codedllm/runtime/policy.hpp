#pragma once

#include "codedllm/coding/graph_generator.hpp"
#include "codedllm/coding/planner.hpp"
#include "codedllm/core/types.hpp"

#include <chrono>
#include <cstddef>
#include <optional>
#include <unordered_set>
#include <vector>

namespace codedllm::runtime {

enum class PolicyDecision { Wait, Recover, Unrecoverable };

struct CostEstimates {
  std::chrono::microseconds expected_wait_remaining;
  std::chrono::microseconds expected_recovery_cost;
};

class ShardArrivalTracker {
public:
  ShardArrivalTracker(std::size_t systematic_shard_count,
                      std::size_t parity_shard_count);

  void MarkArrived(ShardId id, std::chrono::steady_clock::time_point now);

  [[nodiscard]] bool IsSystematicComplete() const;
  [[nodiscard]] std::unordered_set<ShardId> GetMissingSystematic() const;
  [[nodiscard]] bool HasArrived(ShardId id) const;

  [[nodiscard]] std::size_t systematic_shard_count() const noexcept {
    return systematic_shard_count_;
  }

  [[nodiscard]] std::size_t parity_shard_count() const noexcept {
    return parity_shard_count_;
  }

private:
  std::size_t systematic_shard_count_;
  std::size_t parity_shard_count_;
  std::vector<std::optional<std::chrono::steady_clock::time_point>> arrivals_;
};

class RecoveryPolicy {
public:
  RecoveryPolicy(coding::BipartiteGraph graph, coding::DecodePlanner planner);

  [[nodiscard]] PolicyDecision Evaluate(const ShardArrivalTracker& tracker,
                                        const CostEstimates& costs) const;

private:
  const coding::BipartiteGraph graph_;
  const coding::DecodePlanner planner_;
};

} // namespace codedllm::runtime
