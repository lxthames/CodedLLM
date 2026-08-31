#include "codedllm/coding/codec.hpp"
#include "codedllm/coding/graph.hpp"
#include "codedllm/coding/plan.hpp"
#include "kernels/sparse_decoder.cuh"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace codedllm {
namespace {

using WordShard = std::vector<std::uint32_t>;
using ShardSlots = std::vector<std::optional<WordShard>>;

coding::CodeGraph MakeThreeDataOneParityGraph() {
  return coding::CodeGraph(
      /*systematic_shard_count=*/3, /*parity_shard_count=*/1,
      {{/*output=*/3, /*sources=*/{0, 1, 2}}});
}

std::vector<WordShard> MakeSystematicShards() {
  return {{0x01020304U, 0x11121314U, 0x21222324U},
          {0xA0B0C0D0U, 0x01010101U, 0xFFFFFFFFU},
          {0x00000001U, 0x22222222U, 0x12345678U}};
}

DecodePlan MakeRecoverFirstShardPlan() {
  return DecodePlan{{{/*output=*/0, /*sources=*/{3, 1, 2}}}, {{}}, true};
}

TEST(CodecTest, EncodesParityForHandAuthoredSystematicGraph) {
  const std::vector<WordShard> systematic_shards = MakeSystematicShards();
  const std::vector<WordShard> parity_shards =
      coding::encode_parity_cpu(MakeThreeDataOneParityGraph(), systematic_shards);

  ASSERT_EQ(parity_shards.size(), 1U);
  EXPECT_EQ(parity_shards.at(0),
            (WordShard{0xA1B2C3D5U, 0x32313037U, 0xCCE98AA3U}));
}

TEST(CodecTest, RecoversMissingShardWithSameDecodePlan) {
  const std::vector<WordShard> systematic_shards = MakeSystematicShards();
  const std::vector<WordShard> parity_shards =
      coding::encode_parity_cpu(MakeThreeDataOneParityGraph(), systematic_shards);
  ShardSlots shards = {std::nullopt, systematic_shards.at(1),
                       systematic_shards.at(2), parity_shards.at(0)};

  coding::execute_decode_plan_cpu(MakeRecoverFirstShardPlan(), shards);

  ASSERT_TRUE(shards.at(0).has_value());
  EXPECT_EQ(*shards.at(0), systematic_shards.at(0));
}

TEST(CodecTest, CudaRecoversMissingShardWithSameDecodePlan) {
  const std::vector<WordShard> systematic_shards = MakeSystematicShards();
  const std::vector<WordShard> parity_shards =
      coding::encode_parity_cpu(MakeThreeDataOneParityGraph(), systematic_shards);
  ShardSlots shards = {std::nullopt, systematic_shards.at(1),
                       systematic_shards.at(2), parity_shards.at(0)};

  run_decode_plan_cuda(MakeRecoverFirstShardPlan(), shards);

  ASSERT_TRUE(shards.at(0).has_value());
  EXPECT_EQ(*shards.at(0), systematic_shards.at(0));
}

TEST(CodecTest, CudaExecutesDependentOperationsInOnePlan) {
  ShardSlots shards = {WordShard{0x01020304U, 0x11121314U},
                       WordShard{0xA0B0C0D0U, 0x01010101U}, std::nullopt,
                       std::nullopt};
  const DecodePlan plan = {
      {{/*output=*/2, /*sources=*/{0, 1}},
       {/*output=*/3, /*sources=*/{2, 0}}},
      {{}, {0}}, true};

  run_decode_plan_cuda(plan, shards);

  ASSERT_TRUE(shards.at(2).has_value());
  ASSERT_TRUE(shards.at(3).has_value());
  EXPECT_EQ(*shards.at(2), (WordShard{0xA1B2C3D4U, 0x10131215U}));
  EXPECT_EQ(*shards.at(3), (WordShard{0xA0B0C0D0U, 0x01010101U}));
}

TEST(CodecTest, CudaUsesDynamicDegreePathWithVectorTail) {
  ShardSlots shards = {
      WordShard{0x01020304U, 0x11121314U, 0x21222324U, 0x31323334U,
                0x41424344U},
      WordShard{0xA0B0C0D0U, 0x01010101U, 0xFFFFFFFFU, 0x12345678U,
                0x87654321U},
      WordShard{0x00000001U, 0x22222222U, 0x12345678U, 0x87654321U,
                0xABCDEF01U},
      WordShard{0x11111111U, 0x33333333U, 0x44444444U, 0x55555555U,
                0x66666666U},
      WordShard{0x77777777U, 0x88888888U, 0x99999999U, 0xAAAAAAAAU,
                0xBBBBBBBBU},
      WordShard{0xCCCCCCCCU, 0xDDDDDDDDU, 0xEEEEEEEEU, 0xFFFFFFFFU,
                0x13579BDFU},
      WordShard{0x2468ACE0U, 0x0F0F0F0FU, 0xF0F0F0F0U, 0x13579BDFU,
                0x2468ACE0U},
      std::nullopt};
  const DecodePlan plan = {{{/*output=*/7, /*sources=*/{0, 1, 2, 3, 4, 5, 6}}},
                           {{}}, true};

  WordShard expected = *shards.at(0);
  for (std::size_t source = 1; source < 7; ++source) {
    for (std::size_t word = 0; word < expected.size(); ++word) {
      expected.at(word) ^= shards.at(source)->at(word);
    }
  }

  run_decode_plan_cuda(plan, shards);

  ASSERT_TRUE(shards.at(7).has_value());
  EXPECT_EQ(*shards.at(7), expected);
}

}  // namespace
}  // namespace codedllm
