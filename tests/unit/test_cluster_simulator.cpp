#include "codedllm/simulation/cluster_simulator.hpp"
#include "codedllm/simulation/recovery_queue.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace codedllm::simulation {
namespace {

using namespace std::chrono_literals;

coding::BipartiteGraph MakeTestGraph() {
  return coding::BipartiteGraph{/*systematic_shard_count=*/2,
                                /*parity_shard_count=*/1,
                                {{/*parity=*/2, {/*systematic=*/0, 1}}}};
}

ClusterSimulator::RecoveryServiceTimeEstimator
ConstantServiceTime(std::chrono::microseconds service_time) {
  return [service_time](const DecodePlan&) { return service_time; };
}

void AddRecoverableRequest(std::vector<GlobalArrivalEvent>& events,
                           std::size_t request_id,
                           std::chrono::microseconds natural_completion) {
  events.push_back({0us, request_id, /*systematic=*/0});
  events.push_back({0us, request_id, /*parity=*/2});
  events.push_back({natural_completion, request_id, /*systematic=*/1});
}

TEST(RecoveryQueueTest, TestAdvanceTimeReleasesCompletedCapacity) {
  RecoveryQueue queue(/*concurrency_limit=*/1, /*max_queue_depth=*/2);
  ASSERT_TRUE(queue.SubmitJob(0us, 2ms));
  ASSERT_TRUE(queue.SubmitJob(0us, 2ms));
  EXPECT_FALSE(queue.GetExpectedRecoveryCost(0us, 2ms).has_value());

  queue.AdvanceTime(2ms);

  const auto cost = queue.GetExpectedRecoveryCost(2ms, 2ms);
  ASSERT_TRUE(cost.has_value());
  EXPECT_EQ(*cost, 4ms);
  EXPECT_TRUE(queue.SubmitJob(2ms, 2ms));
}

TEST(RecoveryQueueTest, TestDurationSpecificCosts) {
  RecoveryQueue queue(/*concurrency_limit=*/1, /*max_queue_depth=*/3);
  ASSERT_TRUE(queue.SubmitJob(0us, 10ms));

  const auto short_cost = queue.GetExpectedRecoveryCost(0us, 2ms);
  const auto long_cost = queue.GetExpectedRecoveryCost(0us, 7ms);
  ASSERT_TRUE(short_cost.has_value());
  ASSERT_TRUE(long_cost.has_value());
  EXPECT_EQ(*short_cost, 12ms);
  EXPECT_EQ(*long_cost, 17ms);
}

TEST(RecoveryQueueTest, TestZeroDepthDisablesRecovery) {
  RecoveryQueue queue(/*concurrency_limit=*/1, /*max_queue_depth=*/0);

  EXPECT_FALSE(queue.GetExpectedRecoveryCost(0us, 2ms).has_value());
  EXPECT_FALSE(queue.SubmitJob(0us, 2ms));
}

TEST(ClusterSimulatorTest, TestImmediateAdmission) {
  std::vector<GlobalArrivalEvent> events;
  AddRecoverableRequest(events, /*request_id=*/0, /*natural_completion=*/20ms);
  ClusterSimulator simulator(MakeTestGraph(), coding::DecodePlanner{},
                             RecoveryQueue(/*concurrency_limit=*/1,
                                           /*max_queue_depth=*/4),
                             ConstantServiceTime(2ms));

  const auto metrics =
      simulator.Run(events, [](std::size_t, std::chrono::microseconds current_time) {
        return 20ms - current_time;
      });

  ASSERT_EQ(metrics.size(), 1U);
  EXPECT_EQ(metrics.at(0).latency, 2ms);
  EXPECT_TRUE(metrics.at(0).recovery_won);
  EXPECT_TRUE(metrics.at(0).recovery_admitted);
  EXPECT_FALSE(metrics.at(0).recovery_rejected);
}

TEST(ClusterSimulatorTest, TestLaterParityArrivalReplansAfterUnrecoverablePlan) {
  const std::vector<GlobalArrivalEvent> events{
      {0us, /*request_id=*/0, /*systematic=*/0},
      {1ms, /*request_id=*/0, /*parity=*/2},
      {20ms, /*request_id=*/0, /*systematic=*/1},
  };
  ClusterSimulator simulator(MakeTestGraph(), coding::DecodePlanner{},
                             RecoveryQueue(/*concurrency_limit=*/1,
                                           /*max_queue_depth=*/4),
                             ConstantServiceTime(2ms));

  const auto metrics =
      simulator.Run(events, [](std::size_t, std::chrono::microseconds current_time) {
        return 20ms - current_time;
      });

  EXPECT_EQ(metrics.at(0).latency, 3ms);
  EXPECT_TRUE(metrics.at(0).recovery_unrecoverable);
  EXPECT_TRUE(metrics.at(0).recovery_admitted);
  EXPECT_TRUE(metrics.at(0).recovery_won);
}

TEST(ClusterSimulatorTest, TestLatencyIsRelativeToRequestStart) {
  const std::vector<GlobalArrivalEvent> events{
      {101ms, /*request_id=*/0, /*systematic=*/0, /*request_start=*/100ms},
      {105ms, /*request_id=*/0, /*systematic=*/1, /*request_start=*/100ms}};
  ClusterSimulator simulator(MakeTestGraph(), coding::DecodePlanner{},
                             RecoveryQueue(/*concurrency_limit=*/1,
                                           /*max_queue_depth=*/0),
                             ConstantServiceTime(2ms));

  const auto metrics =
      simulator.Run(events, [](std::size_t, std::chrono::microseconds) { return 4ms; });

  EXPECT_EQ(metrics.at(0).latency, 5ms);
  EXPECT_FALSE(metrics.at(0).recovery_won);
}

TEST(ClusterSimulatorTest, TestMultiSlotConcurrency) {
  std::vector<GlobalArrivalEvent> events;
  AddRecoverableRequest(events, /*request_id=*/0, /*natural_completion=*/20ms);
  AddRecoverableRequest(events, /*request_id=*/1, /*natural_completion=*/20ms);
  ClusterSimulator simulator(MakeTestGraph(), coding::DecodePlanner{},
                             RecoveryQueue(/*concurrency_limit=*/2,
                                           /*max_queue_depth=*/4),
                             ConstantServiceTime(2ms));

  const auto metrics =
      simulator.Run(events, [](std::size_t, std::chrono::microseconds current_time) {
        return 20ms - current_time;
      });

  EXPECT_EQ(metrics.at(0).latency, 2ms);
  EXPECT_EQ(metrics.at(1).latency, 2ms);
  EXPECT_TRUE(metrics.at(0).recovery_won);
  EXPECT_TRUE(metrics.at(1).recovery_won);
}

TEST(ClusterSimulatorTest, TestQueueDelayForcesWait) {
  RecoveryQueue queue(/*concurrency_limit=*/1, /*max_queue_depth=*/4);
  ASSERT_TRUE(queue.SubmitJob(0us, 10ms));

  std::vector<GlobalArrivalEvent> events;
  AddRecoverableRequest(events, /*request_id=*/0, /*natural_completion=*/5ms);
  ClusterSimulator simulator(MakeTestGraph(), coding::DecodePlanner{}, std::move(queue),
                             ConstantServiceTime(10ms));

  const auto metrics =
      simulator.Run(events, [](std::size_t, std::chrono::microseconds current_time) {
        return 5ms - current_time;
      });

  EXPECT_EQ(metrics.at(0).latency, 5ms);
  EXPECT_FALSE(metrics.at(0).recovery_won);
  EXPECT_FALSE(metrics.at(0).recovery_admitted);
  EXPECT_FALSE(metrics.at(0).recovery_rejected);
  EXPECT_TRUE(metrics.at(0).recovery_waited);
}

TEST(ClusterSimulatorTest, TestQueueFullRejection) {
  RecoveryQueue queue(/*concurrency_limit=*/1, /*max_queue_depth=*/1);
  ASSERT_TRUE(queue.SubmitJob(0us, 10ms));

  std::vector<GlobalArrivalEvent> events;
  AddRecoverableRequest(events, /*request_id=*/0, /*natural_completion=*/20ms);
  ClusterSimulator simulator(MakeTestGraph(), coding::DecodePlanner{}, std::move(queue),
                             ConstantServiceTime(10ms));

  const auto metrics =
      simulator.Run(events, [](std::size_t, std::chrono::microseconds current_time) {
        return 20ms - current_time;
      });

  EXPECT_EQ(metrics.at(0).latency, 20ms);
  EXPECT_FALSE(metrics.at(0).recovery_won);
  EXPECT_FALSE(metrics.at(0).recovery_admitted);
  EXPECT_TRUE(metrics.at(0).recovery_rejected);
}

TEST(ClusterSimulatorTest, TestPlanSpecificCostChangesAdmission) {
  std::vector<GlobalArrivalEvent> events;
  AddRecoverableRequest(events, /*request_id=*/0, /*natural_completion=*/5ms);

  ClusterSimulator fast(MakeTestGraph(), coding::DecodePlanner{},
                        RecoveryQueue(/*concurrency_limit=*/1,
                                      /*max_queue_depth=*/4),
                        ConstantServiceTime(2ms));
  ClusterSimulator slow(MakeTestGraph(), coding::DecodePlanner{},
                        RecoveryQueue(/*concurrency_limit=*/1,
                                      /*max_queue_depth=*/4),
                        ConstantServiceTime(6ms));
  const auto wait = [](std::size_t, std::chrono::microseconds current_time) {
    return 5ms - current_time;
  };

  EXPECT_TRUE(fast.Run(events, wait).at(0).recovery_won);
  const RequestMetrics slow_metrics = slow.Run(events, wait).at(0);
  EXPECT_FALSE(slow_metrics.recovery_admitted);
  EXPECT_TRUE(slow_metrics.recovery_waited);
}

TEST(ClusterSimulatorTest, TestDeterministicTieBreaking) {
  std::vector<GlobalArrivalEvent> canonical_events;
  for (std::size_t request_id = 0; request_id < 3; ++request_id) {
    AddRecoverableRequest(canonical_events, request_id,
                          /*natural_completion=*/20ms);
  }

  std::map<std::size_t, RequestMetrics> reference;
  for (std::uint32_t seed = 0; seed < 10; ++seed) {
    std::vector<GlobalArrivalEvent> shuffled_events = canonical_events;
    std::mt19937 generator(seed);
    std::shuffle(shuffled_events.begin(), shuffled_events.end(), generator);

    ClusterSimulator simulator(MakeTestGraph(), coding::DecodePlanner{},
                               RecoveryQueue(/*concurrency_limit=*/1,
                                             /*max_queue_depth=*/3),
                               ConstantServiceTime(2ms));
    const auto metrics = simulator.Run(
        shuffled_events, [](std::size_t, std::chrono::microseconds current_time) {
          return 20ms - current_time;
        });

    if (seed == 0) {
      reference = metrics;
    } else {
      EXPECT_EQ(metrics, reference);
    }
  }

  ASSERT_EQ(reference.size(), 3U);
  EXPECT_EQ(reference.at(0).latency, 2ms);
  EXPECT_EQ(reference.at(1).latency, 4ms);
  EXPECT_EQ(reference.at(2).latency, 6ms);
}

} // namespace
} // namespace codedllm::simulation
