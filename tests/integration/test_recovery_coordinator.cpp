#include "codedllm/coding/codec.hpp"
#include "codedllm/runtime/serving.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>

namespace codedllm::runtime {
namespace {

using namespace std::chrono_literals;

coding::BipartiteGraph MakeGraph() {
  return coding::BipartiteGraph{/*systematic_shard_count=*/2,
                                /*parity_shard_count=*/1,
                                {{/*parity=*/2, {/*systematic=*/0, 1}}}};
}

coding::WordShard Xor(const coding::WordShard& left, const coding::WordShard& right) {
  coding::WordShard output(left.size());
  for (std::size_t word = 0; word < left.size(); ++word) {
    output.at(word) = left.at(word) ^ right.at(word);
  }
  return output;
}

class FakeTransport final : public ShardTransport {
public:
  void SetArrivalHandler(ArrivalHandler handler) override {
    handler_ = std::move(handler);
  }

  void StartRequest(RequestId request_id) override { started_ = request_id; }

  void Emit(RequestId request_id, ShardId shard_id, coding::WordShard payload,
            std::chrono::steady_clock::time_point now) {
    ASSERT_TRUE(static_cast<bool>(handler_));
    handler_(request_id, shard_id, std::move(payload), now);
  }

  [[nodiscard]] std::optional<RequestId> started() const { return started_; }

private:
  ArrivalHandler handler_;
  std::optional<RequestId> started_;
};

class FakeExecutor final : public RecoveryExecutor {
public:
  struct Task {
    DecodePlan plan;
    coding::ShardSlots shards;
    CompletionHandler completion;
  };

  explicit FakeExecutor(std::optional<std::chrono::microseconds> estimated_cost = 2ms)
      : estimated_cost_(estimated_cost) {}

  std::optional<std::chrono::microseconds> EstimateRecoveryCost() const override {
    return estimated_cost_;
  }

  bool Submit(RequestId request_id, DecodePlan plan, coding::ShardSlots shards,
              CompletionHandler completion) override {
    return tasks_
        .emplace(request_id,
                 Task{std::move(plan), std::move(shards), std::move(completion)})
        .second;
  }

  bool Cancel(RequestId request_id) override {
    cancelled_ = tasks_.erase(request_id) != 0;
    return cancelled_;
  }

  void Complete(RequestId request_id) {
    auto task = tasks_.find(request_id);
    ASSERT_NE(task, tasks_.end());
    Task work = std::move(task->second);
    tasks_.erase(task);
    coding::execute_decode_plan_cpu(work.plan, work.shards);
    RecoveryTaskResult result;
    result.success = true;
    result.shards = std::move(work.shards);
    result.metrics.total = 2ms;
    work.completion(std::move(result));
  }

  [[nodiscard]] bool HasTask(RequestId request_id) const {
    return tasks_.count(request_id) != 0;
  }

  [[nodiscard]] bool cancelled() const { return cancelled_; }

private:
  std::optional<std::chrono::microseconds> estimated_cost_;
  std::unordered_map<RequestId, Task> tasks_;
  bool cancelled_ = false;
};

RecoveryCoordinator MakeCoordinator(const std::shared_ptr<RecoveryExecutor>& executor) {
  return RecoveryCoordinator(MakeGraph(), coding::DecodePlanner{}, executor);
}

TEST(RecoveryCoordinatorTest, RecoversThroughInjectedExecutor) {
  const coding::WordShard shard_a{1, 2, 3, 4, 5};
  const coding::WordShard shard_b{9, 8, 7, 6, 5};
  const coding::WordShard parity = Xor(shard_a, shard_b);
  auto executor = std::make_shared<FakeExecutor>();
  RecoveryCoordinator coordinator = MakeCoordinator(executor);
  FakeTransport transport;
  coordinator.AttachTransport(transport);
  coordinator.StartRequest(
      7, [](RequestId, std::chrono::steady_clock::time_point) { return 10ms; });
  const auto now = std::chrono::steady_clock::time_point{} + 1ms;

  transport.Emit(7, 0, shard_a, now);
  transport.Emit(7, 2, parity, now + 1ms);

  EXPECT_TRUE(executor->HasTask(7));
  EXPECT_TRUE(coordinator.GetSnapshot(7).recovery_submitted);
  executor->Complete(7);

  ASSERT_TRUE(coordinator.WaitForCompletion(7, 10ms));
  EXPECT_EQ(coordinator.GetSnapshot(7).status, RequestStatus::Recovered);
  ASSERT_TRUE(coordinator.GetShard(7, 1).has_value());
  EXPECT_EQ(*coordinator.GetShard(7, 1), shard_b);
}

TEST(RecoveryCoordinatorTest, CompletesNaturallyWithoutRecovery) {
  auto executor = std::make_shared<FakeExecutor>();
  RecoveryCoordinator coordinator = MakeCoordinator(executor);
  FakeTransport transport;
  coordinator.AttachTransport(transport);
  coordinator.StartRequest(
      8, [](RequestId, std::chrono::steady_clock::time_point) { return 1ms; });
  const auto now = std::chrono::steady_clock::time_point{} + 1ms;

  transport.Emit(8, 0, coding::WordShard{1, 2}, now);
  transport.Emit(8, 1, coding::WordShard{3, 4}, now + 1ms);

  EXPECT_EQ(coordinator.GetSnapshot(8).status, RequestStatus::NaturalComplete);
  EXPECT_FALSE(coordinator.GetSnapshot(8).recovery_submitted);
}

TEST(RecoveryCoordinatorTest, NaturalCompletionCancelsQueuedRecovery) {
  const coding::WordShard shard_a{1, 2};
  const coding::WordShard shard_b{3, 4};
  auto executor = std::make_shared<FakeExecutor>();
  RecoveryCoordinator coordinator = MakeCoordinator(executor);
  FakeTransport transport;
  coordinator.AttachTransport(transport);
  coordinator.StartRequest(
      9, [](RequestId, std::chrono::steady_clock::time_point) { return 10ms; });
  const auto now = std::chrono::steady_clock::time_point{} + 1ms;

  transport.Emit(9, 0, shard_a, now);
  transport.Emit(9, 2, Xor(shard_a, shard_b), now + 1ms);
  ASSERT_TRUE(executor->HasTask(9));
  transport.Emit(9, 1, shard_b, now + 2ms);

  EXPECT_EQ(coordinator.GetSnapshot(9).status, RequestStatus::NaturalComplete);
  EXPECT_TRUE(executor->cancelled());
  EXPECT_FALSE(executor->HasTask(9));
  transport.Emit(9, 2, Xor(shard_a, shard_b), now + 3ms);
  EXPECT_EQ(coordinator.GetSnapshot(9).late_arrivals, 1U);
}

TEST(RecoveryCoordinatorTest, QueueRejectionFallsBackToNaturalCompletion) {
  auto executor = std::make_shared<FakeExecutor>(std::nullopt);
  RecoveryCoordinator coordinator = MakeCoordinator(executor);
  FakeTransport transport;
  coordinator.AttachTransport(transport);
  coordinator.StartRequest(
      10, [](RequestId, std::chrono::steady_clock::time_point) { return 10ms; });
  const auto now = std::chrono::steady_clock::time_point{} + 1ms;
  const coding::WordShard shard_a{1, 2};
  const coding::WordShard shard_b{3, 4};

  transport.Emit(10, 0, shard_a, now);
  transport.Emit(10, 2, Xor(shard_a, shard_b), now + 1ms);
  EXPECT_TRUE(coordinator.GetSnapshot(10).recovery_rejected);
  transport.Emit(10, 1, shard_b, now + 2ms);

  EXPECT_EQ(coordinator.GetSnapshot(10).status, RequestStatus::NaturalComplete);
}

} // namespace
} // namespace codedllm::runtime
