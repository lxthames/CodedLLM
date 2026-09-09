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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace codedllm::simulation {
namespace {

using namespace std::chrono_literals;

constexpr std::size_t kSystematicShards = 16;
constexpr std::uint32_t kFullReplicationSeedSalt = 0xC2B2AE35U;
constexpr std::uint32_t kCapacityReplicationSeedSalt = 0xA24BAED4U;
constexpr std::uint32_t kGraphSeedSalt = 0x9E3779B9U;

struct ExperimentSummary {
  std::int64_t p50_us = 0;
  std::int64_t p95_us = 0;
  std::int64_t p99_us = 0;
  double admission_rate_percent = 0.0;
  double rejection_rate_percent = 0.0;
  double wait_rate_percent = 0.0;
  double unrecoverable_rate_percent = 0.0;
  double win_rate_percent = 0.0;
};

struct SweepOptions {
  std::uint32_t base_seed = 20260901U;
  std::size_t seed_count = 20;
  std::size_t num_requests = 1000;
  std::string scenario = "custom";
  std::vector<double> arrival_rates_hz{2000.0};
  std::vector<double> straggler_probabilities{0.05};
  std::vector<std::size_t> parity_counts{8};
  std::vector<std::size_t> degrees{3};
  std::vector<std::size_t> shard_sizes_mib{16};
  std::vector<std::size_t> queue_depths{4};
  std::vector<std::size_t> concurrency_limits{2};
  std::string calibration_csv;
};

class DecodeCostTable {
public:
  explicit DecodeCostTable(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
      throw std::invalid_argument("Cannot open decode calibration CSV: " + path);
    }

    std::string header;
    if (!std::getline(input, header) ||
        header != "degree,shard_size_mib,end_to_end_us") {
      throw std::invalid_argument(
          "Decode calibration CSV must use degree,shard_size_mib,end_to_end_us");
    }

    std::string line;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
      ++line_number;
      if (line.empty()) {
        continue;
      }
      std::stringstream row(line);
      std::string degree_text;
      std::string size_text;
      std::string cost_text;
      std::string extra;
      if (!std::getline(row, degree_text, ',') || !std::getline(row, size_text, ',') ||
          !std::getline(row, cost_text, ',') || std::getline(row, extra, ',')) {
        throw std::invalid_argument("Malformed decode calibration row " +
                                    std::to_string(line_number));
      }
      const std::size_t degree = ParseSize(degree_text, "decode calibration degree");
      const std::size_t shard_size_mib =
          ParseSize(size_text, "decode calibration shard size");
      const std::int64_t cost_us = ParseCost(cost_text, line_number);
      if (degree == 0 || shard_size_mib == 0 || cost_us <= 0) {
        throw std::invalid_argument("Decode calibration values must be positive");
      }
      if (!costs_
               .emplace(std::make_pair(degree, shard_size_mib),
                        std::chrono::microseconds(cost_us))
               .second) {
        throw std::invalid_argument("Duplicate decode calibration row");
      }
    }
    if (costs_.empty()) {
      throw std::invalid_argument("Decode calibration CSV contains no measurements");
    }
  }

  [[nodiscard]] std::chrono::microseconds Get(std::size_t degree,
                                              std::size_t shard_size_mib) const {
    const auto measured = costs_.find(std::make_pair(degree, shard_size_mib));
    if (measured == costs_.end()) {
      throw std::invalid_argument("No measured decode cost for degree " +
                                  std::to_string(degree) + " at " +
                                  std::to_string(shard_size_mib) + " MiB");
    }
    return measured->second;
  }

private:
  static std::size_t ParseSize(const std::string& value, const std::string& field) {
    std::size_t parsed_characters = 0;
    const std::uint64_t parsed = std::stoull(value, &parsed_characters);
    if (parsed_characters != value.size() ||
        parsed > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("Invalid " + field);
    }
    return static_cast<std::size_t>(parsed);
  }

  static std::int64_t ParseCost(const std::string& value, std::size_t line_number) {
    std::size_t parsed_characters = 0;
    const double parsed = std::stod(value, &parsed_characters);
    if (parsed_characters != value.size() || !std::isfinite(parsed) || parsed <= 0.0 ||
        parsed > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
      throw std::invalid_argument("Invalid decode calibration cost on row " +
                                  std::to_string(line_number));
    }
    return static_cast<std::int64_t>(std::ceil(parsed));
  }

  std::map<std::pair<std::size_t, std::size_t>, std::chrono::microseconds> costs_;
};

std::uint64_t ParseUnsigned(const std::string& value, const std::string& option) {
  std::size_t parsed_characters = 0;
  const std::uint64_t parsed = std::stoull(value, &parsed_characters);
  if (parsed_characters != value.size()) {
    throw std::invalid_argument("Invalid value for " + option);
  }
  return parsed;
}

std::vector<std::string> SplitCsv(const std::string& value, const std::string& option) {
  std::vector<std::string> values;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t delimiter = value.find(',', start);
    const std::string item = value.substr(start, delimiter - start);
    if (item.empty()) {
      throw std::invalid_argument("Invalid value for " + option);
    }
    values.push_back(item);
    if (delimiter == std::string::npos) {
      break;
    }
    start = delimiter + 1;
  }
  return values;
}

std::vector<double> ParseDoubleList(const std::string& value, const std::string& option,
                                    bool allow_zero = false) {
  std::vector<double> values;
  for (const std::string& item : SplitCsv(value, option)) {
    std::size_t parsed_characters = 0;
    const double parsed = std::stod(item, &parsed_characters);
    if (parsed_characters != item.size() || !std::isfinite(parsed) ||
        (allow_zero ? parsed < 0.0 : parsed <= 0.0)) {
      throw std::invalid_argument("Invalid value for " + option);
    }
    values.push_back(parsed);
  }
  return values;
}

std::vector<std::size_t> ParseSizeList(const std::string& value,
                                       const std::string& option,
                                       bool allow_zero = false) {
  std::vector<std::size_t> values;
  for (const std::string& item : SplitCsv(value, option)) {
    const std::uint64_t parsed = ParseUnsigned(item, option);
    if ((!allow_zero && parsed == 0) ||
        parsed > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("Invalid value for " + option);
    }
    values.push_back(static_cast<std::size_t>(parsed));
  }
  return values;
}

void RequireNoDuplicates(const std::vector<double>& values, const std::string& option) {
  std::set<double> unique(values.begin(), values.end());
  if (unique.size() != values.size()) {
    throw std::invalid_argument("Duplicate value for " + option);
  }
}

void RequireNoDuplicates(const std::vector<std::size_t>& values,
                         const std::string& option) {
  std::set<std::size_t> unique(values.begin(), values.end());
  if (unique.size() != values.size()) {
    throw std::invalid_argument("Duplicate value for " + option);
  }
}

SweepOptions ParseOptions(int argc, char** argv) {
  SweepOptions options;
  for (int argument = 1; argument < argc; argument += 2) {
    if (argument + 1 >= argc) {
      throw std::invalid_argument("Every sweep option requires a value");
    }
    const std::string option = argv[argument];
    const std::string value = argv[argument + 1];
    if (option == "--scenario") {
      if (value.empty()) {
        throw std::invalid_argument("--scenario must not be empty");
      }
      options.scenario = value;
    } else if (option == "--calibration-csv") {
      options.calibration_csv = value;
    } else if (option == "--arrival-rates-hz") {
      options.arrival_rates_hz = ParseDoubleList(value, option);
    } else if (option == "--straggler-probabilities") {
      options.straggler_probabilities = ParseDoubleList(value, option, true);
      for (const double probability : options.straggler_probabilities) {
        if (probability > 1.0) {
          throw std::invalid_argument("Straggler probability must not exceed one");
        }
      }
    } else if (option == "--parity-counts") {
      options.parity_counts = ParseSizeList(value, option);
    } else if (option == "--degrees") {
      options.degrees = ParseSizeList(value, option);
    } else if (option == "--shard-sizes-mib") {
      options.shard_sizes_mib = ParseSizeList(value, option);
    } else if (option == "--queue-depths") {
      options.queue_depths = ParseSizeList(value, option, true);
    } else if (option == "--gpu-concurrencies") {
      options.concurrency_limits = ParseSizeList(value, option);
    } else {
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
  }
  if (options.seed_count == 0 || options.num_requests == 0) {
    throw std::invalid_argument("Seed and request counts must be positive");
  }
  if (options.seed_count - 1 >
      std::numeric_limits<std::uint32_t>::max() - options.base_seed) {
    throw std::invalid_argument("Requested seed range exceeds uint32_t");
  }
  if (options.calibration_csv.empty()) {
    throw std::invalid_argument("--calibration-csv is required");
  }
  RequireNoDuplicates(options.arrival_rates_hz, "--arrival-rates-hz");
  RequireNoDuplicates(options.straggler_probabilities, "--straggler-probabilities");
  RequireNoDuplicates(options.parity_counts, "--parity-counts");
  RequireNoDuplicates(options.degrees, "--degrees");
  RequireNoDuplicates(options.shard_sizes_mib, "--shard-sizes-mib");
  RequireNoDuplicates(options.queue_depths, "--queue-depths");
  RequireNoDuplicates(options.concurrency_limits, "--gpu-concurrencies");
  for (const std::size_t parity_count : options.parity_counts) {
    if (parity_count > kSystematicShards) {
      throw std::invalid_argument("Parity count exceeds systematic shard count");
    }
  }
  for (const std::size_t degree : options.degrees) {
    if (degree > kSystematicShards) {
      throw std::invalid_argument("Degree exceeds systematic shard count");
    }
  }
  return options;
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
RunBaseline(const std::vector<GlobalArrivalEvent>& events, std::size_t k) {
  const auto natural_completion = GetNaturalCompletionTimes(events, k);
  std::map<std::size_t, std::chrono::microseconds> request_starts;
  for (const GlobalArrivalEvent& event : events) {
    const auto [start, inserted] =
        request_starts.emplace(event.request_id, event.request_start_time);
    if (!inserted && start->second != event.request_start_time) {
      throw std::invalid_argument(
          "Events for one request must share a request start time");
    }
  }

  std::map<std::size_t, RequestMetrics> metrics;
  for (const auto& [request_id, start] : request_starts) {
    const auto completion = natural_completion.find(request_id);
    if (completion == natural_completion.end()) {
      throw std::runtime_error(
          "Baseline requires a natural arrival for every systematic shard");
    }
    metrics.emplace(request_id, RequestMetrics{completion->second - start});
  }
  return metrics;
}

std::map<std::size_t, RequestMetrics>
RunCodedStrategy(const coding::BipartiteGraph& graph,
                 const std::vector<GlobalArrivalEvent>& events, std::size_t concurrency,
                 std::size_t queue_depth, std::size_t shard_size_mib,
                 const DecodeCostTable& decode_costs) {
  const auto natural_completion =
      GetNaturalCompletionTimes(events, graph.systematic_shard_count);
  const ClusterSimulator::RecoveryServiceTimeEstimator estimate_service_time =
      [&decode_costs, shard_size_mib](const DecodePlan& plan) {
        std::chrono::microseconds total{};
        for (const XorOperation& operation : plan.operations) {
          total += decode_costs.Get(operation.sources.size(), shard_size_mib);
        }
        if (total <= std::chrono::microseconds::zero()) {
          throw std::invalid_argument(
              "Recoverable decode plan must contain operations");
        }
        return total;
      };
  ClusterSimulator simulator(graph, coding::DecodePlanner{},
                             RecoveryQueue(concurrency, queue_depth),
                             estimate_service_time);
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

ExperimentSummary Summarize(const std::map<std::size_t, RequestMetrics>& metrics) {
  std::vector<std::int64_t> latencies;
  latencies.reserve(metrics.size());
  std::size_t admitted = 0;
  std::size_t rejected = 0;
  std::size_t waited = 0;
  std::size_t unrecoverable = 0;
  std::size_t won = 0;
  for (const auto& [request_id, request] : metrics) {
    static_cast<void>(request_id);
    latencies.push_back(request.latency.count());
    admitted += request.recovery_admitted ? 1U : 0U;
    rejected += request.recovery_rejected ? 1U : 0U;
    waited += request.recovery_waited ? 1U : 0U;
    unrecoverable += request.recovery_unrecoverable ? 1U : 0U;
    won += request.recovery_won ? 1U : 0U;
  }

  const double denominator = static_cast<double>(metrics.size());
  ExperimentSummary summary;
  summary.p50_us = Percentile(latencies, 0.50);
  summary.p95_us = Percentile(latencies, 0.95);
  summary.p99_us = Percentile(latencies, 0.99);
  summary.admission_rate_percent = 100.0 * static_cast<double>(admitted) / denominator;
  summary.rejection_rate_percent = 100.0 * static_cast<double>(rejected) / denominator;
  summary.wait_rate_percent = 100.0 * static_cast<double>(waited) / denominator;
  summary.unrecoverable_rate_percent =
      100.0 * static_cast<double>(unrecoverable) / denominator;
  summary.win_rate_percent = 100.0 * static_cast<double>(won) / denominator;
  return summary;
}

void PrintRow(const std::string& scenario, std::uint32_t seed,
              const WorkloadConfig& workload, std::size_t k, std::size_t m,
              std::size_t degree, std::size_t shard_size_mib, std::size_t queue_depth,
              std::size_t concurrency, const std::string& strategy,
              std::size_t added_shard_count, const ExperimentSummary& summary) {
  const double storage_overhead_percent =
      100.0 * static_cast<double>(added_shard_count) / static_cast<double>(k);
  std::cout << std::fixed << std::setprecision(3) << scenario << ',' << seed << ','
            << workload.num_requests << ',' << workload.request_arrival_rate_hz << ','
            << k << ',' << m << ',' << degree << ',' << shard_size_mib << ','
            << workload.straggler_probability << ',' << queue_depth << ','
            << concurrency << ',' << strategy << ',' << added_shard_count << ','
            << storage_overhead_percent << ',' << "end_to_end_cuda_benchmark_v1" << ','
            << summary.p50_us << ',' << summary.p95_us << ',' << summary.p99_us << ','
            << summary.admission_rate_percent << ',' << summary.rejection_rate_percent
            << ',' << summary.wait_rate_percent << ','
            << summary.unrecoverable_rate_percent << ',' << summary.win_rate_percent
            << '\n';
}

} // namespace
} // namespace codedllm::simulation

int main(int argc, char** argv) {
  using namespace codedllm::simulation;
  try {
    const SweepOptions options = ParseOptions(argc, argv);
    const DecodeCostTable decode_costs(options.calibration_csv);

    std::cout << "scenario,seed,num_requests,arrival_rate_hz,k,m,degree,shard_size_mib,"
                 "straggler_probability,queue_depth,gpu_concurrency,strategy,"
                 "added_shard_count,storage_overhead_pct,decode_cost_model,p50_us,"
                 "p95_us,p99_us,recovery_admission_pct,recovery_rejection_pct,"
                 "recovery_wait_pct,recovery_unrecoverable_pct,recovery_win_pct\n";

    for (std::size_t seed_offset = 0; seed_offset < options.seed_count; ++seed_offset) {
      const std::uint32_t seed =
          options.base_seed + static_cast<std::uint32_t>(seed_offset);
      for (const double arrival_rate_hz : options.arrival_rates_hz) {
        for (const double straggler_probability : options.straggler_probabilities) {
          for (const std::size_t parity_count : options.parity_counts) {
            const WorkloadConfig workload{options.num_requests,     arrival_rate_hz,
                                          straggler_probability,
                                          /*normal_mean=*/2ms,
                                          /*normal_stddev=*/400us,
                                          /*straggler_mean=*/20ms,
                                          /*straggler_stddev=*/4ms, seed};
            const WorkloadGenerator generator(workload, kSystematicShards,
                                              parity_count);
            const std::vector<GlobalArrivalEvent> base_events = generator.Generate();
            const std::vector<GlobalArrivalEvent> capacity_replicated_events =
                ApplyCapacityNormalizedReplication(base_events, workload,
                                                   kSystematicShards, parity_count,
                                                   seed ^ kCapacityReplicationSeedSalt);
            const std::vector<GlobalArrivalEvent> fully_replicated_events =
                ApplyReplication(base_events, workload, kSystematicShards,
                                 seed ^ kFullReplicationSeedSalt);
            const auto wait_metrics = RunBaseline(base_events, kSystematicShards);
            const auto capacity_replication_metrics =
                RunBaseline(capacity_replicated_events, kSystematicShards);
            const auto full_replication_metrics =
                RunBaseline(fully_replicated_events, kSystematicShards);

            for (const std::size_t degree : options.degrees) {
              for (const std::size_t shard_size_mib : options.shard_sizes_mib) {
                static_cast<void>(decode_costs.Get(degree, shard_size_mib));
                const codedllm::coding::BipartiteGraph graph =
                    codedllm::coding::GenerateRegularGraph(
                        kSystematicShards, parity_count, degree,
                        seed ^ kGraphSeedSalt ^
                            static_cast<std::uint32_t>(parity_count) ^
                            static_cast<std::uint32_t>(degree));
                for (const std::size_t queue_depth : options.queue_depths) {
                  for (const std::size_t concurrency : options.concurrency_limits) {
                    const auto coded_metrics =
                        RunCodedStrategy(graph, base_events, concurrency, queue_depth,
                                         shard_size_mib, decode_costs);
                    PrintRow(options.scenario, seed, workload, kSystematicShards,
                             parity_count, degree, shard_size_mib, queue_depth,
                             concurrency, "wait", /*added_shard_count=*/0,
                             Summarize(wait_metrics));
                    PrintRow(options.scenario, seed, workload, kSystematicShards,
                             parity_count, degree, shard_size_mib, queue_depth,
                             concurrency, "replication_capacity_normalized",
                             /*added_shard_count=*/parity_count,
                             Summarize(capacity_replication_metrics));
                    PrintRow(options.scenario, seed, workload, kSystematicShards,
                             parity_count, degree, shard_size_mib, queue_depth,
                             concurrency, "replication_full",
                             /*added_shard_count=*/kSystematicShards,
                             Summarize(full_replication_metrics));
                    PrintRow(options.scenario, seed, workload, kSystematicShards,
                             parity_count, degree, shard_size_mib, queue_depth,
                             concurrency, "codedllm",
                             /*added_shard_count=*/parity_count,
                             Summarize(coded_metrics));
                  }
                }
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
