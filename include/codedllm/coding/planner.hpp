#pragma once

#include "codedllm/coding/graph_generator.hpp"
#include "codedllm/coding/plan.hpp"

#include <unordered_set>

namespace codedllm::coding {

class DecodePlanner {
public:
  [[nodiscard]] DecodePlan
  CreatePlan(const BipartiteGraph& graph,
             const std::unordered_set<ShardId>& missing_systematic_shards) const;

  [[nodiscard]] DecodePlan
  CreatePlan(const BipartiteGraph& graph,
             const std::unordered_set<ShardId>& missing_systematic_shards,
             const std::unordered_set<ShardId>& available_parity_shards) const;
};

} // namespace codedllm::coding
