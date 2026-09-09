#include "codedllm/simulation/workload_generator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace codedllm::simulation {
namespace {

using namespace std::chrono_literals;

WorkloadConfig MakeConfig() {
  return WorkloadConfig{/*num_requests=*/32,
                        /*request_arrival_rate_hz=*/500.0,
                        /*straggler_probability=*/0.1,
                        /*normal_mean=*/2ms,
                        /*normal_stddev=*/250us,
                        /*straggler_mean=*/20ms,
                        /*straggler_stddev=*/2ms,
                        /*seed=*/12345};
}

TEST(WorkloadGeneratorTest, SameSeedProducesIdenticalEvents) {
  const WorkloadConfig config = MakeConfig();
  const WorkloadGenerator first(config, /*systematic_shard_count=*/16,
                                /*parity_shard_count=*/8);
  const WorkloadGenerator second(config, /*systematic_shard_count=*/16,
                                 /*parity_shard_count=*/8);

  EXPECT_EQ(first.Generate(), second.Generate());
}

TEST(WorkloadGeneratorTest, FullReplicationOnlyAcceleratesSystematicShards) {
  const WorkloadConfig config = MakeConfig();
  const WorkloadGenerator generator(config, /*systematic_shard_count=*/16,
                                    /*parity_shard_count=*/8);
  const std::vector<GlobalArrivalEvent> original = generator.Generate();
  const std::vector<GlobalArrivalEvent> replicated =
      ApplyReplication(original, config, /*systematic_shard_count=*/16,
                       /*replication_seed=*/98765);

  ASSERT_EQ(replicated.size(), original.size());
  for (std::size_t index = 0; index < original.size(); ++index) {
    EXPECT_EQ(replicated.at(index).request_id, original.at(index).request_id);
    EXPECT_EQ(replicated.at(index).shard_id, original.at(index).shard_id);
    if (original.at(index).shard_id < 16) {
      EXPECT_LE(replicated.at(index).arrival_time, original.at(index).arrival_time);
    } else {
      EXPECT_EQ(replicated.at(index).arrival_time, original.at(index).arrival_time);
    }
  }
}

TEST(WorkloadGeneratorTest, CapacityNormalizedSelectionIsDeterministicAndExact) {
  constexpr std::size_t kSystematic = 16;
  constexpr std::size_t kReplicas = 4;
  std::vector<std::size_t> protections(kSystematic);
  for (std::size_t request = 0; request < kSystematic; ++request) {
    const std::vector<ShardId> first = SelectCapacityNormalizedReplicas(
        request, kSystematic, kReplicas, /*replication_seed=*/98765);
    const std::vector<ShardId> second = SelectCapacityNormalizedReplicas(
        request, kSystematic, kReplicas, /*replication_seed=*/98765);
    EXPECT_EQ(first, second);
    EXPECT_EQ(first.size(), kReplicas);
    for (const ShardId shard : first) {
      EXPECT_LT(shard, kSystematic);
      EXPECT_EQ(std::count(first.begin(), first.end(), shard), 1);
      ++protections.at(shard);
    }
  }
  for (const std::size_t count : protections) {
    EXPECT_EQ(count, kReplicas);
  }
}

TEST(WorkloadGeneratorTest, CapacityNormalizedReplicationOnlyProtectsSelectedShards) {
  const WorkloadConfig config = MakeConfig();
  const WorkloadGenerator generator(config, /*systematic_shard_count=*/16,
                                    /*parity_shard_count=*/8);
  const std::vector<GlobalArrivalEvent> original = generator.Generate();
  const std::vector<GlobalArrivalEvent> replicated =
      ApplyCapacityNormalizedReplication(original, config,
                                         /*systematic_shard_count=*/16,
                                         /*replica_count=*/4,
                                         /*replication_seed=*/98765);

  ASSERT_EQ(replicated.size(), original.size());
  for (std::size_t index = 0; index < original.size(); ++index) {
    const GlobalArrivalEvent& before = original.at(index);
    const GlobalArrivalEvent& after = replicated.at(index);
    const std::vector<ShardId> selected = SelectCapacityNormalizedReplicas(
        before.request_id, /*systematic_shard_count=*/16, /*replica_count=*/4,
        /*replication_seed=*/98765);
    const bool is_selected =
        std::find(selected.begin(), selected.end(), before.shard_id) != selected.end();
    if (is_selected) {
      EXPECT_LE(after.arrival_time, before.arrival_time);
    } else {
      EXPECT_EQ(after.arrival_time, before.arrival_time);
    }
  }
}

} // namespace
} // namespace codedllm::simulation
