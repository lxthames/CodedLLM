#include "codedllm/coding/codec.hpp"
#include "codedllm/coding/graph_generator.hpp"
#include "codedllm/coding/planner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace codedllm::coding {
namespace {

BipartiteGraph MakeThreeDataTwoParityGraph() {
  return BipartiteGraph{
      /*systematic_shard_count=*/3,
      /*parity_shard_count=*/2,
      {{/*parity=*/3, {/*systematic=*/0, 1}}, {/*parity=*/4, {/*systematic=*/1, 2}}}};
}

void ExpectTopologicallyValid(const BipartiteGraph& graph,
                              const std::unordered_set<ShardId>& initially_missing,
                              const DecodePlan& plan) {
  ASSERT_TRUE(plan.is_recoverable);
  ASSERT_EQ(plan.operations.size(), plan.dependencies.size());

  std::unordered_set<ShardId> available;
  for (ShardId shard = 0; shard < graph.systematic_shard_count; ++shard) {
    if (initially_missing.count(shard) == 0) {
      available.insert(shard);
    }
  }
  for (const auto& entry : graph.parity_to_systematic) {
    available.insert(entry.first);
  }

  std::unordered_map<ShardId, std::size_t> producer;
  for (std::size_t operation_index = 0; operation_index < plan.operations.size();
       ++operation_index) {
    const XorOperation& operation = plan.operations.at(operation_index);
    for (const std::size_t dependency : plan.dependencies.at(operation_index)) {
      EXPECT_LT(dependency, operation_index);
    }
    for (const ShardId source : operation.sources) {
      EXPECT_EQ(available.count(source), 1U);
      const auto producing_operation = producer.find(source);
      if (producing_operation != producer.end()) {
        EXPECT_NE(std::find(plan.dependencies.at(operation_index).begin(),
                            plan.dependencies.at(operation_index).end(),
                            producing_operation->second),
                  plan.dependencies.at(operation_index).end());
      }
    }
    EXPECT_EQ(initially_missing.count(operation.output), 1U);
    EXPECT_EQ(available.count(operation.output), 0U);
    available.insert(operation.output);
    producer.emplace(operation.output, operation_index);
  }

  for (const ShardId missing : initially_missing) {
    EXPECT_EQ(available.count(missing), 1U);
  }
}

TEST(PeelingDecoderTest, SimpleRecoverable) {
  const BipartiteGraph graph = MakeThreeDataTwoParityGraph();
  const std::unordered_set<ShardId> missing = {1};

  const DecodePlan plan = DecodePlanner{}.CreatePlan(graph, missing);

  ASSERT_TRUE(plan.is_recoverable);
  ASSERT_EQ(plan.operations.size(), 1U);
  EXPECT_EQ(plan.operations.front().output, 1U);
  EXPECT_EQ(plan.operations.front().sources, (std::vector<ShardId>{3, 0}));
  ExpectTopologicallyValid(graph, missing, plan);
}

TEST(PeelingDecoderTest, Unrecoverable) {
  const BipartiteGraph graph = MakeThreeDataTwoParityGraph();
  const std::unordered_set<ShardId> missing = {0, 1, 2};

  const DecodePlan plan = DecodePlanner{}.CreatePlan(graph, missing);

  EXPECT_FALSE(plan.is_recoverable);
  EXPECT_TRUE(plan.operations.empty());
  EXPECT_TRUE(plan.dependencies.empty());
}

TEST(PeelingDecoderTest, UsesOnlyAvailableParityAndPreservesShardIds) {
  const BipartiteGraph graph = MakeThreeDataTwoParityGraph();
  const std::unordered_set<ShardId> missing{1};
  const std::unordered_set<ShardId> available_parity{/*parity=*/4};

  const DecodePlan plan = DecodePlanner{}.CreatePlan(graph, missing, available_parity);

  ASSERT_TRUE(plan.is_recoverable);
  ASSERT_EQ(plan.operations.size(), 1U);
  EXPECT_EQ(plan.operations.front().output, 1U);
  EXPECT_EQ(plan.operations.front().sources,
            (std::vector<ShardId>{/*parity=*/4, /*systematic=*/2}));
}

TEST(PeelingDecoderTest, RandomizedPropertyTest) {
  constexpr std::size_t kIterations = 100;
  constexpr std::size_t kSystematicShards = 16;
  constexpr std::size_t kParityShards = 8;
  constexpr std::size_t kDegree = 3;
  constexpr std::size_t kWordsPerShard = 8;
  std::mt19937 erasure_generator(0xC0DE1234U);
  std::mt19937 data_generator(0x1234C0DEU);
  std::size_t recoverable_cases = 0;

  for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
    const BipartiteGraph graph =
        GenerateRegularGraph(kSystematicShards, kParityShards, kDegree,
                             static_cast<std::uint32_t>(iteration));
    std::vector<ShardId> candidates(kSystematicShards);
    std::iota(candidates.begin(), candidates.end(), 0U);
    std::shuffle(candidates.begin(), candidates.end(), erasure_generator);
    const std::unordered_set<ShardId> missing = {candidates.at(0), candidates.at(1)};

    coding::ShardSlots shards(kSystematicShards + kParityShards);
    std::vector<coding::WordShard> original_systematic(kSystematicShards);
    for (std::size_t systematic = 0; systematic < kSystematicShards; ++systematic) {
      coding::WordShard data(kWordsPerShard);
      std::generate(data.begin(), data.end(),
                    [&]() { return static_cast<std::uint32_t>(data_generator()); });
      original_systematic.at(systematic) = data;
      shards.at(systematic) = std::move(data);
    }
    for (const auto& equation : graph.parity_to_systematic) {
      coding::WordShard parity(kWordsPerShard, 0U);
      for (const ShardId source : equation.second) {
        for (std::size_t word = 0; word < kWordsPerShard; ++word) {
          parity.at(word) ^= original_systematic.at(source).at(word);
        }
      }
      shards.at(equation.first) = std::move(parity);
    }
    for (const ShardId erased : missing) {
      shards.at(erased) = std::nullopt;
    }

    const DecodePlan plan = DecodePlanner{}.CreatePlan(graph, missing);
    if (plan.is_recoverable) {
      ++recoverable_cases;
      ExpectTopologicallyValid(graph, missing, plan);
      execute_decode_plan_cpu(plan, shards);
      for (const ShardId erased : missing) {
        ASSERT_TRUE(shards.at(erased).has_value());
        EXPECT_EQ(*shards.at(erased), original_systematic.at(erased));
      }
    }
  }

  EXPECT_GT(recoverable_cases, 0U);
}

} // namespace
} // namespace codedllm::coding
