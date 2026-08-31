#pragma once

#include "codedllm/core/types.hpp"

#include <cstddef>
#include <vector>

namespace codedllm {

struct XorOperation {
  ShardId output;
  std::vector<ShardId> sources;
};

struct DecodePlan {
  // dependencies[i] lists operation indices that must complete before operations[i].
  std::vector<XorOperation> operations;
  std::vector<std::vector<std::size_t>> dependencies;
  bool is_recoverable = true;

  [[nodiscard]] static DecodePlan Unrecoverable() {
    return DecodePlan{{}, {}, false};
  }
};

}  // namespace codedllm
