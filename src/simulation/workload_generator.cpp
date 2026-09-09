#include "codedllm/simulation/workload_generator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace codedllm::simulation {
namespace {

constexpr std::uint32_t kArrivalSeedSalt = 0x9E3779B9U;
constexpr std::uint32_t kDelaySeedSalt = 0x85EBCA6BU;

void ValidateConfig(const WorkloadConfig& config) {
  if (config.num_requests == 0) {
    throw std::invalid_argument("Workload must contain at least one request");
  }
  if (!std::isfinite(config.request_arrival_rate_hz) ||
      config.request_arrival_rate_hz <= 0.0) {
    throw std::invalid_argument("Request arrival rate must be positive");
  }
  if (!std::isfinite(config.straggler_probability) ||
      config.straggler_probability < 0.0 || config.straggler_probability > 1.0) {
    throw std::invalid_argument("Straggler probability must be between zero and one");
  }
  if (config.normal_mean < std::chrono::microseconds::zero() ||
      config.normal_stddev < std::chrono::microseconds::zero() ||
      config.straggler_mean < std::chrono::microseconds::zero() ||
      config.straggler_stddev < std::chrono::microseconds::zero()) {
    throw std::invalid_argument("Workload delay parameters must not be negative");
  }
}

void ValidateSystematicShardCount(std::size_t systematic_shard_count) {
  if (systematic_shard_count == 0 ||
      systematic_shard_count > std::numeric_limits<ShardId>::max()) {
    throw std::invalid_argument("Systematic shard count is out of bounds");
  }
}

std::vector<std::chrono::microseconds>
GenerateRequestArrivalTimes(const WorkloadConfig& config) {
  std::mt19937 generator(config.seed ^ kArrivalSeedSalt);
  std::exponential_distribution<double> interarrival_seconds(
      config.request_arrival_rate_hz);

  std::vector<std::chrono::microseconds> arrivals;
  arrivals.reserve(config.num_requests);
  long double cumulative_microseconds = 0.0L;
  for (std::size_t request = 0; request < config.num_requests; ++request) {
    cumulative_microseconds +=
        static_cast<long double>(interarrival_seconds(generator)) * 1'000'000.0L;
    arrivals.emplace_back(static_cast<std::chrono::microseconds::rep>(
        std::llround(cumulative_microseconds)));
  }
  return arrivals;
}

class DelaySampler {
public:
  DelaySampler(const WorkloadConfig& config, std::uint32_t seed)
      : generator_(seed), is_straggler_(config.straggler_probability),
        normal_(static_cast<double>(config.normal_mean.count()),
                static_cast<double>(config.normal_stddev.count())),
        straggler_(static_cast<double>(config.straggler_mean.count()),
                   static_cast<double>(config.straggler_stddev.count())) {}

  std::chrono::microseconds Sample() {
    const double sampled =
        is_straggler_(generator_) ? straggler_(generator_) : normal_(generator_);
    return std::chrono::microseconds(static_cast<std::chrono::microseconds::rep>(
        std::llround(std::max(0.0, sampled))));
  }

private:
  std::mt19937 generator_;
  std::bernoulli_distribution is_straggler_;
  std::normal_distribution<double> normal_;
  std::normal_distribution<double> straggler_;
};

void ValidateReplicationEvents(
    const std::vector<GlobalArrivalEvent>& events,
    const std::vector<std::chrono::microseconds>& request_arrivals,
    std::size_t systematic_shard_count) {
  for (const GlobalArrivalEvent& event : events) {
    if (event.request_id >= request_arrivals.size()) {
      throw std::invalid_argument("Replication event request ID is out of bounds");
    }
    if (event.request_start_time != request_arrivals.at(event.request_id)) {
      throw std::invalid_argument(
          "Replication events do not match the workload request schedule");
    }
  }
}

bool IsSelected(ShardId shard_id, const std::vector<ShardId>& selected) {
  return std::find(selected.begin(), selected.end(), shard_id) != selected.end();
}

} // namespace

WorkloadGenerator::WorkloadGenerator(WorkloadConfig config,
                                     std::size_t systematic_shard_count,
                                     std::size_t parity_shard_count)
    : config_(std::move(config)), systematic_shard_count_(systematic_shard_count),
      parity_shard_count_(parity_shard_count) {
  ValidateConfig(config_);
  ValidateSystematicShardCount(systematic_shard_count);
  const std::uint64_t total_shards =
      static_cast<std::uint64_t>(systematic_shard_count) +
      static_cast<std::uint64_t>(parity_shard_count);
  if (total_shards >
      static_cast<std::uint64_t>(std::numeric_limits<ShardId>::max()) + 1U) {
    throw std::invalid_argument("Workload shard IDs overflow ShardId");
  }
}

std::vector<GlobalArrivalEvent> WorkloadGenerator::Generate() const {
  const std::vector<std::chrono::microseconds> request_arrivals =
      GenerateRequestArrivalTimes(config_);
  DelaySampler delays(config_, config_.seed ^ kDelaySeedSalt);
  const std::size_t shard_count = systematic_shard_count_ + parity_shard_count_;

  std::vector<GlobalArrivalEvent> events;
  events.reserve(config_.num_requests * shard_count);
  for (std::size_t request = 0; request < config_.num_requests; ++request) {
    for (std::size_t shard = 0; shard < shard_count; ++shard) {
      events.push_back(GlobalArrivalEvent{
          request_arrivals.at(request) + delays.Sample(), request,
          static_cast<ShardId>(shard), request_arrivals.at(request)});
    }
  }
  return events;
}

std::vector<ShardId> SelectCapacityNormalizedReplicas(
    std::size_t request_id, std::size_t systematic_shard_count,
    std::size_t replica_count, std::uint32_t replication_seed) {
  ValidateSystematicShardCount(systematic_shard_count);
  if (replica_count > systematic_shard_count) {
    throw std::invalid_argument("Replica count exceeds systematic shard count");
  }

  std::vector<ShardId> permutation(systematic_shard_count);
  for (std::size_t shard = 0; shard < systematic_shard_count; ++shard) {
    permutation.at(shard) = static_cast<ShardId>(shard);
  }
  std::mt19937 generator(replication_seed);
  std::shuffle(permutation.begin(), permutation.end(), generator);

  std::vector<ShardId> selected;
  selected.reserve(replica_count);
  const std::size_t start =
      (request_id % systematic_shard_count) * replica_count % systematic_shard_count;
  for (std::size_t replica = 0; replica < replica_count; ++replica) {
    selected.push_back(permutation.at((start + replica) % systematic_shard_count));
  }
  return selected;
}

std::vector<GlobalArrivalEvent>
ApplyReplication(const std::vector<GlobalArrivalEvent>& events,
                 const WorkloadConfig& config, std::size_t systematic_shard_count,
                 std::uint32_t replication_seed) {
  ValidateConfig(config);
  ValidateSystematicShardCount(systematic_shard_count);
  const std::vector<std::chrono::microseconds> request_arrivals =
      GenerateRequestArrivalTimes(config);
  ValidateReplicationEvents(events, request_arrivals, systematic_shard_count);
  DelaySampler replica_delays(config, replication_seed);

  std::vector<GlobalArrivalEvent> replicated = events;
  for (GlobalArrivalEvent& event : replicated) {
    if (event.shard_id >= systematic_shard_count) {
      continue;
    }
    const std::chrono::microseconds replica_arrival =
        request_arrivals.at(event.request_id) + replica_delays.Sample();
    event.arrival_time = std::min(event.arrival_time, replica_arrival);
  }
  return replicated;
}

std::vector<GlobalArrivalEvent> ApplyCapacityNormalizedReplication(
    const std::vector<GlobalArrivalEvent>& events, const WorkloadConfig& config,
    std::size_t systematic_shard_count, std::size_t replica_count,
    std::uint32_t replication_seed) {
  ValidateConfig(config);
  ValidateSystematicShardCount(systematic_shard_count);
  if (replica_count > systematic_shard_count) {
    throw std::invalid_argument("Replica count exceeds systematic shard count");
  }
  const std::vector<std::chrono::microseconds> request_arrivals =
      GenerateRequestArrivalTimes(config);
  ValidateReplicationEvents(events, request_arrivals, systematic_shard_count);
  DelaySampler replica_delays(config, replication_seed);

  std::vector<GlobalArrivalEvent> replicated = events;
  for (GlobalArrivalEvent& event : replicated) {
    if (event.shard_id >= systematic_shard_count ||
        !IsSelected(event.shard_id, SelectCapacityNormalizedReplicas(
                                        event.request_id, systematic_shard_count,
                                        replica_count, replication_seed))) {
      continue;
    }
    const std::chrono::microseconds replica_arrival =
        request_arrivals.at(event.request_id) + replica_delays.Sample();
    event.arrival_time = std::min(event.arrival_time, replica_arrival);
  }
  return replicated;
}

} // namespace codedllm::simulation
