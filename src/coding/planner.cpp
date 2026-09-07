#include "codedllm/coding/planner.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace codedllm::coding {
namespace {

void ValidateGraph(const BipartiteGraph& graph) {
  if (graph.systematic_shard_count == 0 || graph.parity_shard_count == 0) {
    throw std::invalid_argument("BipartiteGraph requires non-zero k and m");
  }

  const std::size_t max_shard_count =
      static_cast<std::size_t>(std::numeric_limits<ShardId>::max()) + 1ULL;
  if (graph.systematic_shard_count > std::numeric_limits<ShardId>::max() ||
      graph.parity_shard_count > max_shard_count - graph.systematic_shard_count) {
    throw std::invalid_argument("BipartiteGraph shard IDs overflow ShardId");
  }
  if (graph.parity_to_systematic.size() != graph.parity_shard_count) {
    throw std::invalid_argument(
        "BipartiteGraph requires one equation per parity shard");
  }

  for (std::size_t parity_index = 0; parity_index < graph.parity_shard_count;
       ++parity_index) {
    const ShardId expected_parity =
        static_cast<ShardId>(graph.systematic_shard_count + parity_index);
    const auto equation = graph.parity_to_systematic.find(expected_parity);
    if (equation == graph.parity_to_systematic.end()) {
      throw std::invalid_argument("BipartiteGraph parity shard IDs must be contiguous");
    }
    if (equation->second.empty()) {
      throw std::invalid_argument("BipartiteGraph parity equations must not be empty");
    }

    std::unordered_set<ShardId> unique_neighbors;
    for (const ShardId systematic : equation->second) {
      if (systematic >= graph.systematic_shard_count) {
        throw std::invalid_argument(
            "BipartiteGraph neighbor is not a systematic shard");
      }
      if (!unique_neighbors.insert(systematic).second) {
        throw std::invalid_argument("BipartiteGraph contains a duplicate parity edge");
      }
    }
  }
}

} // namespace

DecodePlan DecodePlanner::CreatePlan(
    const BipartiteGraph& graph,
    const std::unordered_set<ShardId>& missing_systematic_shards) const {
  std::unordered_set<ShardId> available_parity_shards;
  for (const auto& equation : graph.parity_to_systematic) {
    available_parity_shards.insert(equation.first);
  }
  return CreatePlan(graph, missing_systematic_shards, available_parity_shards);
}

DecodePlan DecodePlanner::CreatePlan(
    const BipartiteGraph& graph,
    const std::unordered_set<ShardId>& missing_systematic_shards,
    const std::unordered_set<ShardId>& available_parity_shards) const {
  ValidateGraph(graph);
  for (const ShardId missing : missing_systematic_shards) {
    if (missing >= graph.systematic_shard_count) {
      throw std::invalid_argument(
          "Missing shard set must contain only systematic shards");
    }
  }
  const std::size_t total_shard_count =
      graph.systematic_shard_count + graph.parity_shard_count;
  for (const ShardId parity : available_parity_shards) {
    if (parity < graph.systematic_shard_count || parity >= total_shard_count) {
      throw std::invalid_argument("Available parity set contains a non-parity shard");
    }
  }

  std::unordered_set<ShardId> remaining = missing_systematic_shards;
  DecodePlan plan;
  std::unordered_map<ShardId, std::size_t> producing_operation;

  while (!remaining.empty()) {
    bool made_progress = false;
    for (const auto& equation : graph.parity_to_systematic) {
      if (available_parity_shards.count(equation.first) == 0) {
        continue;
      }
      ShardId missing_neighbor = 0;
      std::size_t missing_neighbor_count = 0;
      for (const ShardId systematic : equation.second) {
        if (remaining.count(systematic) != 0) {
          missing_neighbor = systematic;
          ++missing_neighbor_count;
        }
      }
      if (missing_neighbor_count != 1) {
        continue;
      }

      XorOperation operation;
      operation.output = missing_neighbor;
      operation.sources.reserve(equation.second.size());
      operation.sources.push_back(equation.first);
      std::vector<std::size_t> dependencies;
      for (const ShardId systematic : equation.second) {
        if (systematic == missing_neighbor) {
          continue;
        }
        operation.sources.push_back(systematic);
        const auto producer = producing_operation.find(systematic);
        if (producer != producing_operation.end()) {
          dependencies.push_back(producer->second);
        }
      }
      std::sort(dependencies.begin(), dependencies.end());
      dependencies.erase(std::unique(dependencies.begin(), dependencies.end()),
                         dependencies.end());

      const std::size_t operation_index = plan.operations.size();
      plan.operations.push_back(std::move(operation));
      plan.dependencies.push_back(std::move(dependencies));
      producing_operation.emplace(missing_neighbor, operation_index);
      remaining.erase(missing_neighbor);
      made_progress = true;

      if (remaining.empty()) {
        break;
      }
    }

    if (!made_progress) {
      return DecodePlan::Unrecoverable();
    }
  }

  return plan;
}

} // namespace codedllm::coding
