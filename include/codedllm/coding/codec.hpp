#pragma once

#include "codedllm/coding/graph.hpp"
#include "codedllm/coding/plan.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace codedllm::coding {

using ShardWord = std::uint32_t;
using WordShard = std::vector<ShardWord>;
using ShardSlots = std::vector<std::optional<WordShard>>;

// Produces one parity shard per graph equation, ordered by parity shard ID.
[[nodiscard]] std::vector<WordShard> encode_parity_cpu(
    const CodeGraph& graph, const std::vector<WordShard>& systematic_shards);

// Executes operations in plan order, materializing each missing output shard.
void execute_decode_plan_cpu(const DecodePlan& plan, ShardSlots& shards);

}  // namespace codedllm::coding
