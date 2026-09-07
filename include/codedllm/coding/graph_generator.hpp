#pragma once

#include "codedllm/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace codedllm::coding {

// Parity shard IDs occupy [systematic_shard_count,
// systematic_shard_count + parity_shard_count). A sorted map makes graph
// traversal and generated decode plans deterministic.
struct BipartiteGraph {
  std::size_t systematic_shard_count;
  std::size_t parity_shard_count;
  std::map<ShardId, std::vector<ShardId>> parity_to_systematic;
};

// Generates a right-regular bipartite graph: every parity shard has exactly
// degree distinct systematic neighbors.
[[nodiscard]] BipartiteGraph GenerateRegularGraph(std::size_t k, std::size_t m,
                                                  std::size_t degree,
                                                  std::uint32_t seed);

} // namespace codedllm::coding
