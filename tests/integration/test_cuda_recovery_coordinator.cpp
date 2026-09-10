#include "codedllm/runtime/cuda_recovery_executor.hpp"
#include "codedllm/runtime/serving.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <utility>

namespace codedllm::runtime {
namespace {

using namespace std::chrono_literals;

class ManualTransport final : public ShardTransport {
public:
  void SetArrivalHandler(ArrivalHandler handler) override {
    handler_ = std::move(handler);
  }
  void StartRequest(RequestId request_id) override { static_cast<void>(request_id); }
  void Emit(RequestId request_id, ShardId shard_id, coding::WordShard payload,
            std::chrono::steady_clock::time_point now) {
    handler_(request_id, shard_id, std::move(payload), now);
  }

private:
  ArrivalHandler handler_;
};

TEST(CudaRecoveryCoordinatorTest, DiscardsPreStagedContextOnNaturalCompletion) {
  auto executor = std::make_shared<CudaRecoveryExecutor>(
      /*concurrency_limit=*/1, /*max_queue_depth=*/1,
      /*estimated_service_cost=*/2ms);
  const coding::BipartiteGraph graph{/*systematic_shard_count=*/2,
                                     /*parity_shard_count=*/1,
                                     {{/*parity=*/2, {/*systematic=*/0, 1}}}};
  RecoveryCoordinator coordinator(graph, coding::DecodePlanner{}, executor);
  ManualTransport transport;
  coordinator.AttachTransport(transport);
  coordinator.StartRequest(
      41, [](RequestId, std::chrono::steady_clock::time_point) { return 100ms; });
  const auto now = std::chrono::steady_clock::now();

  transport.Emit(41, 0, coding::WordShard{1, 2, 3}, now);
  transport.Emit(41, 1, coding::WordShard{4, 5, 6}, now + 1us);

  ASSERT_TRUE(coordinator.WaitForCompletion(41, 5s));
  EXPECT_EQ(coordinator.GetSnapshot(41).status, RequestStatus::NaturalComplete);
  EXPECT_FALSE(executor->Cancel(41));
}

TEST(CudaRecoveryCoordinatorTest, RecoversEndToEndThroughBoundedExecutor) {
  constexpr std::size_t kWords = 4097;
  coding::WordShard shard_a(kWords);
  coding::WordShard shard_b(kWords);
  coding::WordShard parity(kWords);
  for (std::size_t word = 0; word < kWords; ++word) {
    shard_a.at(word) = static_cast<std::uint32_t>(word * 17U + 3U);
    shard_b.at(word) = static_cast<std::uint32_t>(word * 29U + 5U);
    parity.at(word) = shard_a.at(word) ^ shard_b.at(word);
  }

  auto executor = std::make_shared<CudaRecoveryExecutor>(
      /*concurrency_limit=*/1, /*max_queue_depth=*/2,
      /*estimated_service_cost=*/2ms);
  const coding::BipartiteGraph graph{/*systematic_shard_count=*/2,
                                     /*parity_shard_count=*/1,
                                     {{/*parity=*/2, {/*systematic=*/0, 1}}}};
  RecoveryCoordinator coordinator(graph, coding::DecodePlanner{}, executor);
  ManualTransport transport;
  coordinator.AttachTransport(transport);
  coordinator.StartRequest(
      42, [](RequestId, std::chrono::steady_clock::time_point) { return 100ms; });
  const auto now = std::chrono::steady_clock::now();

  transport.Emit(42, 0, shard_a, now);
  transport.Emit(42, 2, parity, now + 1us);

  ASSERT_TRUE(coordinator.WaitForCompletion(42, 5s));
  const RequestSnapshot snapshot = coordinator.GetSnapshot(42);
  ASSERT_EQ(snapshot.status, RequestStatus::Recovered) << snapshot.error;
  EXPECT_TRUE(snapshot.recovery_submitted);
  EXPECT_GT(snapshot.metrics.h2d, 0us);
  EXPECT_GT(snapshot.metrics.kernel, 0us);
  ASSERT_TRUE(coordinator.GetShard(42, 1).has_value());
  EXPECT_EQ(*coordinator.GetShard(42, 1), shard_b);
}

} // namespace
} // namespace codedllm::runtime
