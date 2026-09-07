#include "codedllm/simulation/workload_generator.hpp"

#include <gtest/gtest.h>

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

TEST(WorkloadGeneratorTest, ReplicationNeverIncreasesArrivalTime) {
  const WorkloadConfig config = MakeConfig();
  const WorkloadGenerator generator(config, /*systematic_shard_count=*/16,
                                    /*parity_shard_count=*/8);
  const std::vector<GlobalArrivalEvent> original = generator.Generate();
  const std::vector<GlobalArrivalEvent> replicated =
      ApplyReplication(original, config, /*replication_seed=*/98765);

  ASSERT_EQ(replicated.size(), original.size());
  for (std::size_t index = 0; index < original.size(); ++index) {
    EXPECT_EQ(replicated.at(index).request_id, original.at(index).request_id);
    EXPECT_EQ(replicated.at(index).shard_id, original.at(index).shard_id);
    EXPECT_LE(replicated.at(index).arrival_time, original.at(index).arrival_time);
  }
}

} // namespace
} // namespace codedllm::simulation
