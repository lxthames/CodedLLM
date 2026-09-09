#include "codedllm/coding/graph_generator.hpp"
#include "codedllm/coding/planner.hpp"
#include "codedllm/simulation/cluster_simulator.hpp"
#include "codedllm/simulation/recovery_queue.hpp"
#include "codedllm/simulation/workload_generator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace codedllm::simulation {
namespace {

using namespace std::chrono_literals;

struct ExperimentSummary {
  std::int64_t p50_us = 0;
  std::int64_t p95_us = 0;
  std::int64_t p99_us = 0;
  double admission_rate_percent = 0.0;
  double rejection_rate_percent = 0.0;
  double win_rate_percent = 0.0;
};

struct SweepOptions {
  std::uint32_t base_seed = 20260901U;
  std::size_t seed_count = 20;
  std::size_t num_requests = 1000;
  std::vector<double> arrival_rates_hz{2000.0};
};

std::uint64_t ParseUnsigned(const std::string& value, const std::string& option) {
  std::size_t parsed_characters = 0;
  const std::uint64_t parsed = std::stoull(value, &parsed_characters);
  if (parsed_characters != value.size()) {
    throw std::invalid_argument("Invalid value for " + option);
  }
  return parsed;
}

std::vector<double> ParseArrivalRates(const std::string& value) {
  std::vector<double> rates;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t delimiter = value.find(',', start);
    const std::string rate_text = value.substr(start, delimiter - start);
    std::size_t parsed_characters = 0;
    const double rate = std::stod(rate_text, &parsed_characters);
    if (parsed_characters != rate_text.size() || !std::isfinite(rate) || rate <= 0.0) {
      throw std::invalid_argument("Invalid value for --arrival-rates-hz");
    }
    rates.push_back(rate);
    if (delimiter == std::string::npos) {
      break;
    }
    start = delimiter + 1;
  }
  return rates;
}

SweepOptions ParseOptions(int argc, char** argv) {
  SweepOptions options;
  for (int argument = 1; argument < argc; argument += 2) {
    if (argument + 1 >= argc) {
      throw std::invalid_argument("Every sweep option requires a value");
    }
    const std::string option = argv[argument];
    const std::string value = argv[argument + 1];
    if (option == "--arrival-rates-hz") {
      options.arrival_rates_hz = ParseArrivalRates(value);
      continue;
    }

    const std::uint64_t parsed = ParseUnsigned(value, option);
    if (option == "--base-seed") {
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--base-seed exceeds uint32_t");
      }
      options.base_seed = static_cast<std::uint32_t>(parsed);
    } else if (option == "--seed-count") {
      options.seed_count = static_cast<std::size_t>(parsed);
    } else if (option == "--num-requests") {
      options.num_requests = static_cast<std::size_t>(parsed);
    } else {
      throw std::invalid_argument("Unknown sweep option: " + option);
    }
  }
  if (options.seed_count == 0 || options.num_requests == 0) {
    throw std::invalid_argument("Seed and request counts must be positive");
  }
  if (options.seed_count - 1 >
      std::numeric_limits<std::uint32_t>::max() - options.base_seed) {
    throw std::invalid_argument("Requested seed range exceeds uint32_t");
  }
  return options;
}

std::chrono::microseconds GetMeasuredDecodeTime(std::size_t degree,
                                                std::size_t shard_size_mib) {
  // Phase 4 measurements for the native sm_75 build. Values represent a
  // single-shard decode, including dependent operation launch overhead.
  static const std::map<std::size_t, std::int64_t> kDecodeAt16MiBUs{
      {2, 98}, {3, 126}, {5, 190}, {8, 260}};
  const auto measured = kDecodeAt16MiBUs.find(degree);
  if (measured == kDecodeAt16MiBUs.end() || shard_size_mib == 0) {
    throw std::invalid_argument("No measured decode cost for this configuration");
  }
  const double scaled = static_cast<double>(measured->second) *
                        static_cast<double>(shard_size_mib) / 16.0;
  return std::chrono::microseconds(
      std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(scaled))));
}

std::map<std::size_t, std::chrono::microseconds>
GetNaturalCompletionTimes(const std::vector<GlobalArrivalEvent>& events,
                          std::size_t k) {
  std::map<std::size_t, std::chrono::microseconds> completion_times;
  for (const GlobalArrivalEvent& event : events) {
    if (event.shard_id < k) {
      auto [completion, inserted] =
          completion_times.emplace(event.request_id, event.arrival_time);
      if (!inserted) {
        completion->second = std::max(completion->second, event.arrival_time);
      }
    }
  }
  return completion_times;
}

std::map<std::size_t, RequestMetrics>
RunStrategy(const coding::BipartiteGraph& graph,
            const std::vector<GlobalArrivalEvent>& events, std::size_t concurrency,
            std::size_t queue_depth, std::chrono::microseconds decode_time) {
  const auto natural_completion =
      GetNaturalCompletionTimes(events, graph.systematic_shard_count);
  ClusterSimulator simulator(graph, coding::DecodePlanner{},
                             RecoveryQueue(concurrency, queue_depth, decode_time));
  return simulator.Run(
      events, [&natural_completion](std::size_t request_id,
                                    std::chrono::microseconds current_time) {
        const std::chrono::microseconds completion = natural_completion.at(request_id);
        return completion > current_time ? completion - current_time : 0us;
      });
}

std::int64_t Percentile(std::vector<std::int64_t> values, double percentile) {
  if (values.empty()) {
    throw std::invalid_argument("Cannot calculate an empty percentile");
  }
  std::sort(values.begin(), values.end());
  const std::size_t rank = static_cast<std::size_t>(
      std::ceil(percentile * static_cast<double>(values.size())));
  return values.at(std::max<std::size_t>(1, rank) - 1);
}

ExperimentSummary Summarize(const std::map<std::size_t, RequestMetrics>& metrics,
                            bool report_recovery_metrics) {
  std::vector<std::int64_t> latencies;
  latencies.reserve(metrics.size());
  std::size_t admitted = 0;
  std::size_t rejected = 0;
  std::size_t won = 0;
  for (const auto& [request_id, request] : metrics) {
    static_cast<void>(request_id);
    latencies.push_back(request.latency.count());
    admitted += request.recovery_admitted ? 1U : 0U;
    rejected += request.recovery_rejected ? 1U : 0U;
    won += request.recovery_won ? 1U : 0U;
  }

  const double denominator = static_cast<double>(metrics.size());
  ExperimentSummary summary;
  summary.p50_us = Percentile(latencies, 0.50);
  summary.p95_us = Percentile(latencies, 0.95);
  summary.p99_us = Percentile(latencies, 0.99);
  if (report_recovery_metrics) {
    summary.admission_rate_percent =
        100.0 * static_cast<double>(admitted) / denominator;
    summary.rejection_rate_percent =
        100.0 * static_cast<double>(rejected) / denominator;
    summary.win_rate_percent = 100.0 * static_cast<double>(won) / denominator;
  }
  return summary;
}

void PrintRow(std::uint32_t seed, const WorkloadConfig& workload, std::size_t k,
              std::size_t m, std::size_t degree, std::size_t shard_size_mib,
              std::size_t queue_depth, std::size_t concurrency,
              const std::string& strategy, const ExperimentSummary& summary) {
  std::cout << std::fixed << std::setprecision(3) << seed << ','
            << workload.num_requests << ',' << workload.request_arrival_rate_hz << ','
            << k << ',' << m << ',' << degree << ',' << shard_size_mib << ','
            << workload.straggler_probability << ',' << queue_depth << ','
            << concurrency << ',' << strategy << ',' << summary.p50_us << ','
            << summary.p95_us << ',' << summary.p99_us << ','
            << summary.admission_rate_percent << ',' << summary.rejection_rate_percent
            << ',' << summary.win_rate_percent << '\n';
}

} // namespace
} // namespace codedllm::simulation

int main(int argc, char** argv) {
  using namespace codedllm::simulation;
  try {
    const SweepOptions options = ParseOptions(argc, argv);
    constexpr std::size_t kSystematicShards = 16;
    constexpr std::size_t kDegree = 3;
    constexpr std::size_t kShardSizeMiB = 16;
    const std::vector<double> straggler_probabilities{0.01, 0.05, 0.10};
    const std::vector<std::size_t> queue_depths{1, 4, 16};
    const std::vector<std::size_t> concurrency_limits{2, 4};
    const std::vector<std::size_t> parity_counts{4, 8};
    const std::chrono::microseconds decode_time =
        GetMeasuredDecodeTime(kDegree, kShardSizeMiB);

    std::cout << "seed,num_requests,arrival_rate_hz,k,m,degree,shard_size_mib,"
                 "straggler_probability,queue_depth,gpu_concurrency,strategy,"
                 "p50_us,p95_us,p99_us,recovery_admission_pct,"
                 "recovery_rejection_pct,recovery_win_pct\n";

    for (std::size_t seed_offset = 0; seed_offset < options.seed_count; ++seed_offset) {
      const std::uint32_t seed =
          options.base_seed + static_cast<std::uint32_t>(seed_offset);
      for (const double arrival_rate_hz : options.arrival_rates_hz) {
        for (const double straggler_probability : straggler_probabilities) {
          for (const std::size_t parity_count : parity_counts) {
            const WorkloadConfig workload{options.num_requests,     arrival_rate_hz,
                                          straggler_probability,
                                          /*normal_mean=*/2ms,
                                          /*normal_stddev=*/400us,
                                          /*straggler_mean=*/20ms,
                                          /*straggler_stddev=*/4ms, seed};
            const WorkloadGenerator generator(workload, kSystematicShards,
                                              parity_count);
            const std::vector<GlobalArrivalEvent> base_events = generator.Generate();
            const std::vector<GlobalArrivalEvent> replicated_events =
                ApplyReplication(base_events, workload, seed ^ 0xC2B2AE35U);
            const codedllm::coding::BipartiteGraph graph =
                codedllm::coding::GenerateRegularGraph(
                    kSystematicShards, parity_count, kDegree,
                    seed ^ static_cast<std::uint32_t>(parity_count));

            for (const std::size_t queue_depth : queue_depths) {
              for (const std::size_t concurrency : concurrency_limits) {
                const auto wait_metrics =
                    RunStrategy(graph, base_events, /*concurrency=*/1,
                                /*queue_depth=*/0, decode_time);
                const auto replication_metrics =
                    RunStrategy(graph, replicated_events, /*concurrency=*/1,
                                /*queue_depth=*/0, decode_time);
                const auto coded_metrics = RunStrategy(graph, base_events, concurrency,
                                                       queue_depth, decode_time);

                PrintRow(seed, workload, kSystematicShards, parity_count, kDegree,
                         kShardSizeMiB, queue_depth, concurrency, "wait",
                         Summarize(wait_metrics, false));
                PrintRow(seed, workload, kSystematicShards, parity_count, kDegree,
                         kShardSizeMiB, queue_depth, concurrency, "replication",
                         Summarize(replication_metrics, false));
                PrintRow(seed, workload, kSystematicShards, parity_count, kDegree,
                         kShardSizeMiB, queue_depth, concurrency, "codedllm",
                         Summarize(coded_metrics, true));
              }
            }
          }
        }
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "phase7_parameter_sweep: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
