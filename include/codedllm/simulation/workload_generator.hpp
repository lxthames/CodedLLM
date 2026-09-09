#pragma once

#include "codedllm/simulation/cluster_simulator.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace codedllm::simulation {

struct WorkloadConfig {
  std::size_t num_requests;
  double request_arrival_rate_hz;
  double straggler_probability;
  std::chrono::microseconds normal_mean;
  std::chrono::microseconds normal_stddev;
  std::chrono::microseconds straggler_mean;
  std::chrono::microseconds straggler_stddev;
  std::uint32_t seed;
};

class WorkloadGenerator {
public:
  WorkloadGenerator(WorkloadConfig config, std::size_t systematic_shard_count,
                    std::size_t parity_shard_count = 0);

  [[nodiscard]] std::vector<GlobalArrivalEvent> Generate() const;

private:
  const WorkloadConfig config_;
  const std::size_t systematic_shard_count_;
  const std::size_t parity_shard_count_;
};

[[nodiscard]] std::vector<ShardId> SelectCapacityNormalizedReplicas(
    std::size_t request_id, std::size_t systematic_shard_count,
    std::size_t replica_count, std::uint32_t replication_seed);

[[nodiscard]] std::vector<GlobalArrivalEvent>
ApplyReplication(const std::vector<GlobalArrivalEvent>& events,
                 const WorkloadConfig& config, std::size_t systematic_shard_count,
                 std::uint32_t replication_seed);

[[nodiscard]] std::vector<GlobalArrivalEvent> ApplyCapacityNormalizedReplication(
    const std::vector<GlobalArrivalEvent>& events, const WorkloadConfig& config,
    std::size_t systematic_shard_count, std::size_t replica_count,
    std::uint32_t replication_seed);

} // namespace codedllm::simulation
