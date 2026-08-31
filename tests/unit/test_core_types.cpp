#include "codedllm/coding/graph.hpp"
#include "codedllm/coding/plan.hpp"
#include "codedllm/core/types.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace codedllm {
namespace {

TEST(ShardLayoutTest, CalculatesTotalBytes) {
  const ShardLayout layout(/*word_width=*/4, /*element_count=*/256);

  EXPECT_EQ(layout.word_width, 4U);
  EXPECT_EQ(layout.element_count, 256U);
  EXPECT_EQ(layout.byte_length, 1024U);
}

TEST(ShardLayoutTest, RejectsZeroSizedDimensions) {
  EXPECT_THROW(ShardLayout(/*word_width=*/0, /*element_count=*/1),
               std::invalid_argument);
  EXPECT_THROW(ShardLayout(/*word_width=*/1, /*element_count=*/0),
               std::invalid_argument);
}

TEST(CodeGraphTest, AcceptsValidSystematicParityGraph) {
  const coding::CodeGraph graph(
      /*systematic_shard_count=*/3, /*parity_shard_count=*/2,
      {{/*output=*/3, /*sources=*/{0, 1}},
       {/*output=*/4, /*sources=*/{1, 2}}});

  EXPECT_EQ(graph.systematic_shard_count(), 3U);
  EXPECT_EQ(graph.parity_shard_count(), 2U);
  ASSERT_EQ(graph.parity_equations().size(), 2U);
  EXPECT_EQ(graph.parity_equations().at(0).output, 3U);
  EXPECT_EQ(graph.parity_equations().at(0).sources,
            (std::vector<ShardId>{0, 1}));
}

TEST(CodeGraphTest, RejectsDuplicateEdges) {
  EXPECT_THROW((coding::CodeGraph(
                   /*systematic_shard_count=*/3, /*parity_shard_count=*/1,
                   {{/*output=*/3, /*sources=*/{0, 0}}})),
               std::invalid_argument);
}

TEST(CodeGraphTest, RejectsOutOfBoundsShardIndices) {
  EXPECT_THROW((coding::CodeGraph(
                   /*systematic_shard_count=*/3, /*parity_shard_count=*/1,
                   {{/*output=*/3, /*sources=*/{0, 4}}})),
               std::invalid_argument);
}

TEST(CodeGraphTest, RejectsEmptyParityEquations) {
  EXPECT_THROW((coding::CodeGraph(
                   /*systematic_shard_count=*/3, /*parity_shard_count=*/1,
                   {{/*output=*/3, /*sources=*/{}}})),
               std::invalid_argument);
}

TEST(DecodePlanTest, StoresXorOperationsAndDependencies) {
  DecodePlan plan;
  plan.operations = {{/*output=*/2, /*sources=*/{0, 1}}};
  plan.dependencies = {{}};

  ASSERT_TRUE(plan.is_recoverable);
  ASSERT_EQ(plan.operations.size(), 1U);
  EXPECT_EQ(plan.operations.at(0).output, 2U);
  EXPECT_EQ(plan.operations.at(0).sources, (std::vector<ShardId>{0, 1}));
  EXPECT_EQ(plan.dependencies, (std::vector<std::vector<std::size_t>>{{}}));
}

TEST(DecodePlanTest, RepresentsUnrecoverableResult) {
  const DecodePlan plan = DecodePlan::Unrecoverable();

  EXPECT_FALSE(plan.is_recoverable);
  EXPECT_TRUE(plan.operations.empty());
  EXPECT_TRUE(plan.dependencies.empty());
}

}  // namespace
}  // namespace codedllm
