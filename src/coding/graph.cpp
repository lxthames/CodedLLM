#include "codedllm/coding/graph.hpp"

#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace codedllm::coding {

CodeGraph::CodeGraph(ShardId systematic_shard_count,
                     ShardId parity_shard_count,
                     std::vector<ParityEquation> parity_equations)
    : systematic_shard_count_(systematic_shard_count),
      parity_shard_count_(parity_shard_count),
      parity_equations_(std::move(parity_equations)) {
  Validate();
}

void CodeGraph::Validate() const {
  if (systematic_shard_count_ == 0 || parity_shard_count_ == 0) {
    throw std::invalid_argument("CodeGraph requires non-zero k and m");
  }
  if (systematic_shard_count_ >
      std::numeric_limits<ShardId>::max() - parity_shard_count_) {
    throw std::invalid_argument("CodeGraph shard count overflows ShardId");
  }
  if (parity_equations_.size() != parity_shard_count_) {
    throw std::invalid_argument(
        "CodeGraph requires exactly one equation per parity shard");
  }

  const ShardId total_shards = total_shard_count();
  std::unordered_set<ShardId> outputs;
  for (const ParityEquation& equation : parity_equations_) {
    if (equation.output < systematic_shard_count_ ||
        equation.output >= total_shards) {
      throw std::invalid_argument("CodeGraph parity output is out of bounds");
    }
    if (!outputs.insert(equation.output).second) {
      throw std::invalid_argument("CodeGraph contains duplicate parity outputs");
    }
    if (equation.sources.empty()) {
      throw std::invalid_argument("CodeGraph parity equations must not be empty");
    }

    std::unordered_set<ShardId> sources;
    for (const ShardId source : equation.sources) {
      if (source >= total_shards) {
        throw std::invalid_argument("CodeGraph source shard is out of bounds");
      }
      if (!sources.insert(source).second) {
        throw std::invalid_argument("CodeGraph contains duplicate edges");
      }
    }
  }
}

}  // namespace codedllm::coding
