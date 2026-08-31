#include "codedllm/coding/codec.hpp"

#include <cstddef>
#include <stdexcept>

namespace codedllm::coding {
namespace {

std::size_t ValidateShardWords(const std::vector<WordShard>& shards) {
  if (shards.empty() || shards.front().empty()) {
    throw std::invalid_argument("Shards must be non-empty");
  }

  const std::size_t word_count = shards.front().size();
  for (const WordShard& shard : shards) {
    if (shard.size() != word_count) {
      throw std::invalid_argument("All shards must have the same word count");
    }
  }
  return word_count;
}

void ValidatePlanShape(const DecodePlan& plan) {
  if (!plan.is_recoverable) {
    throw std::invalid_argument("Cannot execute an unrecoverable decode plan");
  }
  if (plan.dependencies.size() != plan.operations.size()) {
    throw std::invalid_argument(
        "DecodePlan dependencies must match the operation count");
  }
  for (std::size_t operation_index = 0;
       operation_index < plan.dependencies.size(); ++operation_index) {
    for (const std::size_t dependency : plan.dependencies.at(operation_index)) {
      if (dependency >= operation_index) {
        throw std::invalid_argument(
            "DecodePlan dependencies must reference earlier operations");
      }
    }
  }
}

}  // namespace

std::vector<WordShard> encode_parity_cpu(
    const CodeGraph& graph, const std::vector<WordShard>& systematic_shards) {
  if (systematic_shards.size() != graph.systematic_shard_count()) {
    throw std::invalid_argument(
        "Systematic shard count does not match the code graph");
  }

  const std::size_t word_count = ValidateShardWords(systematic_shards);
  std::vector<WordShard> parity_shards(
      graph.parity_shard_count(), WordShard(word_count, 0));

  for (const ParityEquation& equation : graph.parity_equations()) {
    const std::size_t parity_index =
        equation.output - graph.systematic_shard_count();
    WordShard& parity = parity_shards.at(parity_index);
    for (const ShardId source : equation.sources) {
      if (source >= graph.systematic_shard_count()) {
        throw std::invalid_argument(
            "Parity encoding currently supports systematic source shards only");
      }
      const WordShard& input = systematic_shards.at(source);
      for (std::size_t word = 0; word < word_count; ++word) {
        parity.at(word) ^= input.at(word);
      }
    }
  }

  return parity_shards;
}

void execute_decode_plan_cpu(const DecodePlan& plan, ShardSlots& shards) {
  ValidatePlanShape(plan);

  for (const XorOperation& operation : plan.operations) {
    if (operation.output >= shards.size()) {
      throw std::invalid_argument("DecodePlan output shard is out of bounds");
    }
    if (operation.sources.empty()) {
      throw std::invalid_argument("DecodePlan XOR operation has no sources");
    }
    if (shards.at(operation.output).has_value()) {
      throw std::invalid_argument("DecodePlan output shard is already present");
    }

    const ShardId first_source = operation.sources.front();
    if (first_source >= shards.size() || !shards.at(first_source).has_value()) {
      throw std::invalid_argument("DecodePlan source shard is unavailable");
    }
    const std::size_t word_count = shards.at(first_source)->size();
    if (word_count == 0) {
      throw std::invalid_argument("DecodePlan source shards must be non-empty");
    }

    WordShard output(word_count, 0);
    for (const ShardId source : operation.sources) {
      if (source >= shards.size() || !shards.at(source).has_value() ||
          shards.at(source)->size() != word_count) {
        throw std::invalid_argument(
            "DecodePlan sources must be available and equally sized");
      }
      const WordShard& input = *shards.at(source);
      for (std::size_t word = 0; word < word_count; ++word) {
        output.at(word) ^= input.at(word);
      }
    }
    shards.at(operation.output) = std::move(output);
  }
}

}  // namespace codedllm::coding
