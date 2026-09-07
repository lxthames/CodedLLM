#include "codedllm/simulation/simulator.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <vector>

namespace codedllm::simulation {
namespace {

using namespace std::chrono_literals;

coding::BipartiteGraph MakeTestGraph() {
  return coding::BipartiteGraph{/*systematic_shard_count=*/2,
                                /*parity_shard_count=*/1,
                                {{/*parity=*/2, {/*systematic=*/0, 1}}}};
}

TEST(RequestSimulationTest, TestWaitWinsFastArrival) {
  const std::vector<ShardArrivalEvent> events{{0, 1ms}, {/*parity=*/2, 1ms}, {1, 5ms}};
  RequestSimulation simulation(MakeTestGraph(), coding::DecodePlanner{}, events,
                               [](std::chrono::microseconds current_time) {
                                 return CostEstimates{5ms - current_time, 10ms};
                               });

  const SimulationMetrics metrics = simulation.Run();

  EXPECT_EQ(metrics.natural_completion_time, 5ms);
  EXPECT_FALSE(metrics.recovery_start_time.has_value());
  EXPECT_EQ(metrics.final_latency, 5ms);
  EXPECT_FALSE(metrics.recovery_won);
}

TEST(RequestSimulationTest, TestRecoveryWins) {
  const std::vector<ShardArrivalEvent> events{{0, 1ms}, {/*parity=*/2, 5ms}, {1, 20ms}};
  RequestSimulation simulation(MakeTestGraph(), coding::DecodePlanner{}, events,
                               [](std::chrono::microseconds current_time) {
                                 return CostEstimates{20ms - current_time, 2ms};
                               });

  const SimulationMetrics metrics = simulation.Run();

  ASSERT_TRUE(metrics.recovery_start_time.has_value());
  ASSERT_TRUE(metrics.recovery_completion_time.has_value());
  EXPECT_EQ(*metrics.recovery_start_time, 5ms);
  EXPECT_EQ(*metrics.recovery_completion_time, 7ms);
  EXPECT_EQ(metrics.natural_completion_time, 20ms);
  EXPECT_EQ(metrics.final_latency, 7ms);
  EXPECT_TRUE(metrics.recovery_won);
}

TEST(RequestSimulationTest, TestWaitWinsStraggler) {
  const std::vector<ShardArrivalEvent> events{{0, 1ms}, {/*parity=*/2, 5ms}, {1, 10ms}};
  RequestSimulation simulation(MakeTestGraph(), coding::DecodePlanner{}, events,
                               [](std::chrono::microseconds current_time) {
                                 return CostEstimates{10ms - current_time, 8ms};
                               });

  const SimulationMetrics metrics = simulation.Run();

  EXPECT_EQ(metrics.natural_completion_time, 10ms);
  EXPECT_FALSE(metrics.recovery_start_time.has_value());
  EXPECT_EQ(metrics.final_latency, 10ms);
  EXPECT_FALSE(metrics.recovery_won);
}

TEST(RequestSimulationTest, TestOutOfOrderEvents) {
  const std::vector<ShardArrivalEvent> events{{1, 5ms}, {0, 1ms}};
  RequestSimulation simulation(MakeTestGraph(), coding::DecodePlanner{}, events,
                               [](std::chrono::microseconds) {
                                 return CostEstimates{1ms, 2ms};
                               });

  const SimulationMetrics metrics = simulation.Run();

  EXPECT_EQ(metrics.natural_completion_time, 5ms);
  EXPECT_EQ(metrics.final_latency, 5ms);
  EXPECT_FALSE(metrics.recovery_won);
}

} // namespace
} // namespace codedllm::simulation
