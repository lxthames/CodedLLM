#pragma once

#include "codedllm/core/types.hpp"

#include <vector>

namespace codedllm::coding {

struct ParityEquation {
  ShardId output;
  std::vector<ShardId> sources;
};

class CodeGraph {
 public:
  CodeGraph(ShardId systematic_shard_count, ShardId parity_shard_count,
            std::vector<ParityEquation> parity_equations);

  [[nodiscard]] ShardId systematic_shard_count() const noexcept {
    return systematic_shard_count_;
  }

  [[nodiscard]] ShardId parity_shard_count() const noexcept {
    return parity_shard_count_;
  }

  [[nodiscard]] ShardId total_shard_count() const noexcept {
    return systematic_shard_count_ + parity_shard_count_;
  }

  [[nodiscard]] const std::vector<ParityEquation>& parity_equations() const
      noexcept {
    return parity_equations_;
  }

 private:
  void Validate() const;

  const ShardId systematic_shard_count_;
  const ShardId parity_shard_count_;
  const std::vector<ParityEquation> parity_equations_;
};

}  // namespace codedllm::coding
