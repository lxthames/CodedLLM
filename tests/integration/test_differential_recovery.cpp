#include "codedllm/coding/codec.hpp"
#include "codedllm/coding/graph.hpp"
#include "codedllm/coding/graph_generator.hpp"
#include "codedllm/coding/planner.hpp"
#include "kernels/sparse_decoder.cuh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace codedllm::coding {
namespace {

struct DifferentialOutcome {
  bool is_recoverable = false;
  bool has_dependency_chain = false;
};

CodeGraph ToCodeGraph(const BipartiteGraph& graph) {
  std::vector<ParityEquation> equations;
  equations.reserve(graph.parity_to_systematic.size());
  for (const auto& [parity, systematic] : graph.parity_to_systematic) {
    equations.push_back(ParityEquation{parity, systematic});
  }

  return CodeGraph(static_cast<ShardId>(graph.systematic_shard_count),
                   static_cast<ShardId>(graph.parity_shard_count),
                   std::move(equations));
}

bool HasDependencyChain(const DecodePlan& plan) {
  return std::any_of(plan.dependencies.begin(), plan.dependencies.end(),
                     [](const std::vector<std::size_t>& dependencies) {
                       return !dependencies.empty();
                     });
}

std::unordered_set<ShardId>
SelectMissingShards(std::size_t k, std::size_t erasure_count, std::mt19937& generator) {
  std::vector<ShardId> candidates(k);
  std::iota(candidates.begin(), candidates.end(), ShardId{0});
  std::shuffle(candidates.begin(), candidates.end(), generator);
  return {candidates.begin(), candidates.begin() + erasure_count};
}

bool FindDependencyErasureSet(const BipartiteGraph& graph, std::size_t erasure_count,
                              std::unordered_set<ShardId>& missing, DecodePlan& plan) {
  if (erasure_count != 2) {
    return false;
  }

  const DecodePlanner planner;
  for (ShardId first = 0; first < graph.systematic_shard_count; ++first) {
    for (ShardId second = first + 1; second < graph.systematic_shard_count; ++second) {
      std::unordered_set<ShardId> candidate{first, second};
      DecodePlan candidate_plan = planner.CreatePlan(graph, candidate);
      if (candidate_plan.is_recoverable && HasDependencyChain(candidate_plan)) {
        missing = std::move(candidate);
        plan = std::move(candidate_plan);
        return true;
      }
    }
  }
  return false;
}

void RunDifferentialTest(std::size_t k, std::size_t m, std::size_t degree,
                         std::size_t words_per_shard, std::uint32_t seed,
                         std::size_t erasure_count,
                         bool require_dependency_chain = false,
                         DifferentialOutcome* outcome = nullptr) {
  ASSERT_GT(k, 0U);
  ASSERT_GT(words_per_shard, 0U);
  ASSERT_GT(erasure_count, 0U);
  ASSERT_LE(erasure_count, k);

  const BipartiteGraph graph = GenerateRegularGraph(k, m, degree, seed);
  const CodeGraph code_graph = ToCodeGraph(graph);

  std::mt19937 data_generator(seed ^ 0x9E3779B9U);
  std::vector<WordShard> original_systematic(k, WordShard(words_per_shard, 0));
  for (WordShard& shard : original_systematic) {
    std::generate(shard.begin(), shard.end(), [&data_generator]() {
      return static_cast<ShardWord>(data_generator());
    });
  }
  const std::vector<WordShard> parity =
      encode_parity_cpu(code_graph, original_systematic);

  std::mt19937 erasure_generator(seed ^ 0x85EBCA6BU);
  std::unordered_set<ShardId> missing =
      SelectMissingShards(k, erasure_count, erasure_generator);
  const DecodePlanner planner;
  DecodePlan plan = planner.CreatePlan(graph, missing);

  if (require_dependency_chain && (!plan.is_recoverable || !HasDependencyChain(plan))) {
    ASSERT_TRUE(FindDependencyErasureSet(graph, erasure_count, missing, plan))
        << "No recoverable dependency chain found for seed " << seed;
  }

  if (outcome != nullptr) {
    outcome->is_recoverable = plan.is_recoverable;
    outcome->has_dependency_chain = HasDependencyChain(plan);
  }
  if (!plan.is_recoverable) {
    return;
  }

  ShardSlots initial_shards(k + m);
  for (std::size_t shard = 0; shard < k; ++shard) {
    initial_shards.at(shard) = original_systematic.at(shard);
  }
  for (std::size_t parity_index = 0; parity_index < m; ++parity_index) {
    initial_shards.at(k + parity_index) = parity.at(parity_index);
  }
  for (const ShardId erased : missing) {
    initial_shards.at(erased) = std::nullopt;
  }

  ShardSlots shards_cpu = initial_shards;
  ShardSlots shards_gpu = initial_shards;
  ASSERT_NO_THROW(execute_decode_plan_cpu(plan, shards_cpu));
  ASSERT_NO_THROW(run_decode_plan_cuda(plan, shards_gpu));

  for (const ShardId erased : missing) {
    ASSERT_TRUE(shards_cpu.at(erased).has_value());
    ASSERT_TRUE(shards_gpu.at(erased).has_value());
    ASSERT_EQ(*shards_cpu.at(erased), original_systematic.at(erased));
    ASSERT_EQ(*shards_gpu.at(erased), *shards_cpu.at(erased));
    ASSERT_EQ(*shards_gpu.at(erased), original_systematic.at(erased));
  }
}

TEST(DifferentialRecoveryTest, TestSingleErasure) {
  RunDifferentialTest(/*k=*/8, /*m=*/4, /*degree=*/3,
                      /*words_per_shard=*/1024, /*seed=*/11,
                      /*erasure_count=*/1);
}

TEST(DifferentialRecoveryTest, TestMultiErasureDependency) {
  DifferentialOutcome outcome;
  RunDifferentialTest(/*k=*/8, /*m=*/8, /*degree=*/3,
                      /*words_per_shard=*/4096, /*seed=*/29,
                      /*erasure_count=*/2,
                      /*require_dependency_chain=*/true, &outcome);
  ASSERT_TRUE(outcome.is_recoverable);
  ASSERT_TRUE(outcome.has_dependency_chain);
}

TEST(DifferentialRecoveryTest, TestTailSizes) {
  RunDifferentialTest(/*k=*/8, /*m=*/4, /*degree=*/3,
                      /*words_per_shard=*/1'000'001, /*seed=*/47,
                      /*erasure_count=*/1);
}

TEST(DifferentialRecoveryTest, TestFuzzManySeeds) {
  std::size_t recoverable_cases = 0;
  for (std::uint32_t seed = 0; seed < 50; ++seed) {
    DifferentialOutcome outcome;
    RunDifferentialTest(/*k=*/16, /*m=*/8, /*degree=*/3,
                        /*words_per_shard=*/257, seed,
                        /*erasure_count=*/3,
                        /*require_dependency_chain=*/false, &outcome);
    recoverable_cases += outcome.is_recoverable ? 1U : 0U;
  }
  ASSERT_GT(recoverable_cases, 0U);
}

} // namespace
} // namespace codedllm::coding
