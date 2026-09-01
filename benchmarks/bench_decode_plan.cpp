#include "kernels/sparse_decoder.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace codedllm {
namespace {

constexpr std::size_t kWarmupPlans = 10;

struct DependentPlanWorkload {
  DecodePlan plan;
  coding::ShardSlots initial_shards;
  coding::ShardSlots expected_shards;
};

DependentPlanWorkload MakeDependentPlanWorkload(std::size_t operation_count,
                                                std::size_t shard_size_bytes) {
  const std::size_t word_count = shard_size_bytes / sizeof(std::uint32_t);
  DependentPlanWorkload workload;
  workload.plan.is_recoverable = true;
  workload.plan.operations.reserve(operation_count);
  workload.plan.dependencies.reserve(operation_count);

  workload.initial_shards.resize(operation_count + 2);
  for (std::size_t source = 0; source < 2; ++source) {
    coding::WordShard shard(word_count);
    for (std::size_t word = 0; word < word_count; ++word) {
      shard.at(word) =
          static_cast<std::uint32_t>(source * 0x9E3779B9U + word);
    }
    workload.initial_shards.at(source) = std::move(shard);
  }

  for (std::size_t operation = 0; operation < operation_count; ++operation) {
    const ShardId output = static_cast<ShardId>(operation + 2);
    if (operation == 0) {
      workload.plan.operations.push_back({output, {0, 1}});
      workload.plan.dependencies.push_back({});
    } else {
      const ShardId previous_output = static_cast<ShardId>(operation + 1);
      const ShardId base_source = static_cast<ShardId>(operation % 2);
      workload.plan.operations.push_back(
          {output, {previous_output, base_source}});
      workload.plan.dependencies.push_back({operation - 1});
    }
  }

  workload.expected_shards = workload.initial_shards;
  coding::execute_decode_plan_cpu(workload.plan, workload.expected_shards);
  return workload;
}

double MedianMilliseconds(std::vector<float> samples) {
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  if (samples.size() % 2 == 0) {
    return (samples.at(middle - 1) + samples.at(middle)) / 2.0;
  }
  return samples.at(middle);
}

void BM_DependentDecodePlanKernelOnly(benchmark::State& state) {
  const std::size_t operation_count =
      static_cast<std::size_t>(state.range(0));
  const std::size_t shard_size_bytes =
      static_cast<std::size_t>(state.range(1));

  state.PauseTiming();
  DependentPlanWorkload workload =
      MakeDependentPlanWorkload(operation_count, shard_size_bytes);
  std::unique_ptr<DeviceDecodeContext> context =
      prepare_decode_plan_cuda(workload.plan, workload.initial_shards);
  for (std::size_t warmup = 0; warmup < kWarmupPlans; ++warmup) {
    launch_decode_plan_cuda(*context);
  }
  synchronize_decode_plan_cuda(*context);
  std::vector<float> samples_ms;
  samples_ms.reserve(100);
  state.ResumeTiming();

  for (auto _ : state) {
    const float elapsed_ms = launch_decode_plan_cuda_timed(*context);
    state.SetIterationTime(elapsed_ms / 1.0e3);
    samples_ms.push_back(elapsed_ms);
  }

  state.PauseTiming();
  coding::ShardSlots result_shards = workload.initial_shards;
  download_decode_results_cuda(*context, result_shards);
  for (const XorOperation& operation : workload.plan.operations) {
    if (result_shards.at(operation.output) !=
        workload.expected_shards.at(operation.output)) {
      state.SkipWithError("Dependent CUDA decode plan produced incorrect data");
      break;
    }
  }

  const double median_plan_ms = MedianMilliseconds(samples_ms);
  const double bytes_per_plan =
      static_cast<double>(operation_count) * 3.0 * shard_size_bytes;
  state.counters["Operations"] = static_cast<double>(operation_count);
  state.counters["Median Plan (ms)"] = median_plan_ms;
  state.counters["Median Per Operation (us)"] =
      median_plan_ms * 1.0e3 / static_cast<double>(operation_count);
  state.counters["Bandwidth (GB/s)"] =
      bytes_per_plan / (median_plan_ms / 1.0e3) / 1.0e9;
  state.SetBytesProcessed(static_cast<std::int64_t>(samples_ms.size()) *
                          static_cast<std::int64_t>(bytes_per_plan));
  state.ResumeTiming();
}

constexpr std::int64_t kOneMiB = 1024 * 1024;
constexpr std::int64_t kSixteenMiB = 16 * kOneMiB;

BENCHMARK(BM_DependentDecodePlanKernelOnly)
    ->Args({1, kOneMiB})
    ->Args({1, kSixteenMiB})
    ->Args({2, kOneMiB})
    ->Args({2, kSixteenMiB})
    ->Args({4, kOneMiB})
    ->Args({4, kSixteenMiB})
    ->Args({8, kOneMiB})
    ->Args({8, kSixteenMiB})
    ->UseManualTime()
    ->Iterations(100);

}  // namespace
}  // namespace codedllm

BENCHMARK_MAIN();
