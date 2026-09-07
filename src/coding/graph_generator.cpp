#include "codedllm/coding/graph_generator.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace codedllm::coding {

BipartiteGraph GenerateRegularGraph(std::size_t k, std::size_t m, std::size_t degree,
                                    std::uint32_t seed) {
  if (k == 0 || m == 0 || degree == 0) {
    throw std::invalid_argument("Regular graph dimensions and degree must be non-zero");
  }
  if (degree > k) {
    throw std::invalid_argument(
        "Regular graph degree cannot exceed systematic shard count");
  }

  const std::size_t max_shard_count =
      static_cast<std::size_t>(std::numeric_limits<ShardId>::max()) + 1ULL;
  if (k > std::numeric_limits<ShardId>::max() || m > max_shard_count - k) {
    throw std::invalid_argument("Regular graph shard IDs overflow ShardId");
  }

  BipartiteGraph graph{k, m, {}};
  std::vector<ShardId> candidates(k);
  std::iota(candidates.begin(), candidates.end(), 0U);
  std::mt19937 generator(seed);

  for (std::size_t parity_index = 0; parity_index < m; ++parity_index) {
    std::shuffle(candidates.begin(), candidates.end(), generator);
    std::vector<ShardId> neighbors(candidates.begin(), candidates.begin() + degree);
    const ShardId parity_id = static_cast<ShardId>(k + parity_index);
    graph.parity_to_systematic.emplace(parity_id, std::move(neighbors));
  }

  return graph;
}

} // namespace codedllm::coding
