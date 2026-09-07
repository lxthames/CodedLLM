#include "codedllm/runtime/policy.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <utility>

namespace codedllm::runtime {
namespace {

using namespace std::chrono_literals;

coding::BipartiteGraph MakeTestGraph() {
  return coding::BipartiteGraph{
      /*systematic_shard_count=*/3,
      /*parity_shard_count=*/2,
      {{/*parity=*/3, {/*systematic=*/0, 1}}, {/*parity=*/4, {/*systematic=*/1, 2}}}};
}

RecoveryPolicy MakePolicy() {
  return RecoveryPolicy(MakeTestGraph(), coding::DecodePlanner{});
}

TEST(RecoveryPolicyTest, TestFastArrival) {
  ShardArrivalTracker tracker(/*systematic_shard_count=*/3,
                              /*parity_shard_count=*/2);
  const auto now = std::chrono::steady_clock::time_point{} + 100ms;
  tracker.MarkArrived(0, now);
  tracker.MarkArrived(1, now + 1us);
  tracker.MarkArrived(2, now + 2us);

  EXPECT_EQ(MakePolicy().Evaluate(tracker, CostEstimates{10ms, 2ms}),
            PolicyDecision::Wait);
}

TEST(RecoveryPolicyTest, TestStragglerTriggersRecovery) {
  ShardArrivalTracker tracker(/*systematic_shard_count=*/3,
                              /*parity_shard_count=*/2);
  const auto now = std::chrono::steady_clock::time_point{} + 200ms;
  tracker.MarkArrived(0, now);
  tracker.MarkArrived(2, now);
  tracker.MarkArrived(3, now + 1us);

  EXPECT_EQ(MakePolicy().Evaluate(tracker, CostEstimates{10ms, 2ms}),
            PolicyDecision::Recover);
}

TEST(RecoveryPolicyTest, TestWaitIsCheaper) {
  ShardArrivalTracker tracker(/*systematic_shard_count=*/3,
                              /*parity_shard_count=*/2);
  const auto now = std::chrono::steady_clock::time_point{} + 300ms;
  tracker.MarkArrived(0, now);
  tracker.MarkArrived(2, now);
  tracker.MarkArrived(3, now + 1us);

  EXPECT_EQ(MakePolicy().Evaluate(tracker, CostEstimates{1ms, 5ms}),
            PolicyDecision::Wait);
}

TEST(RecoveryPolicyTest, TestUnrecoverablePattern) {
  ShardArrivalTracker tracker(/*systematic_shard_count=*/3,
                              /*parity_shard_count=*/2);
  const auto now = std::chrono::steady_clock::time_point{} + 400ms;
  tracker.MarkArrived(2, now);
  tracker.MarkArrived(4, now + 1us);

  EXPECT_EQ(MakePolicy().Evaluate(tracker, CostEstimates{10ms, 2ms}),
            PolicyDecision::Unrecoverable);
}

} // namespace
} // namespace codedllm::runtime
