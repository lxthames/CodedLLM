#include "kernels/sparse_decoder.cuh"

#include "cuda_memory.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace codedllm {
namespace {

template <typename T>
cuda_unique_ptr<T> AllocateDeviceBuffer(std::size_t element_count) {
  T* device_buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&device_buffer, element_count * sizeof(T)));
  return cuda_unique_ptr<T>(device_buffer);
}

class SparseDecoderBenchmark : public benchmark::Fixture {
 protected:
  void SetUp(const benchmark::State& state) override {
    degree_ = static_cast<std::size_t>(state.range(0));
    shard_size_bytes_ = static_cast<std::size_t>(state.range(1));
    const std::size_t word_count = shard_size_bytes_ / sizeof(std::uint32_t);

    plan_.operations = {
        {static_cast<ShardId>(degree_), std::vector<ShardId>(degree_)}};
    plan_.dependencies = {{}};
    plan_.is_recoverable = true;

    initial_shards_.clear();
    initial_shards_.reserve(degree_ + 1);
    for (std::size_t source = 0; source < degree_; ++source) {
      coding::WordShard shard(word_count);
      for (std::size_t word = 0; word < word_count; ++word) {
        shard[word] = static_cast<std::uint32_t>(source * 0x9E3779B9U + word);
      }
      initial_shards_.emplace_back(std::move(shard));
      plan_.operations.front().sources[source] = static_cast<ShardId>(source);
    }
    initial_shards_.emplace_back(std::nullopt);

    expected_shards_ = initial_shards_;
    coding::execute_decode_plan_cpu(plan_, expected_shards_);
  }

  coding::ShardSlots MakeIterationShards() const { return initial_shards_; }

  void PublishCounters(benchmark::State& state, float h2d_ms, float kernel_ms,
                       float d2h_ms, std::size_t iterations) const {
    const double divisor = static_cast<double>(iterations);
    const double average_h2d_ms = h2d_ms / divisor;
    const double average_kernel_ms = kernel_ms / divisor;
    const double average_d2h_ms = d2h_ms / divisor;
    const double bytes_per_kernel =
        static_cast<double>(degree_ + 1) * shard_size_bytes_;

    // CudaDecodeTimings is measured with cudaEvent_t records immediately around
    // the decoder's H2D, kernel, and D2H phases.
    state.counters["H2D (ms)"] = average_h2d_ms;
    state.counters["Kernel (ms)"] = average_kernel_ms;
    state.counters["D2H (ms)"] = average_d2h_ms;
    state.counters["Bandwidth (GB/s)"] = average_kernel_ms > 0.0
                                                ? bytes_per_kernel /
                                                      (average_kernel_ms / 1.0e3) /
                                                      1.0e9
                                                : 0.0;
  }

  std::size_t degree_ = 0;
  std::size_t shard_size_bytes_ = 0;
  DecodePlan plan_;
  coding::ShardSlots initial_shards_;
  coding::ShardSlots expected_shards_;
};

BENCHMARK_DEFINE_F(SparseDecoderBenchmark, EndToEndLatency)
(benchmark::State& state) {
  float total_h2d_ms = 0.0F;
  float total_kernel_ms = 0.0F;
  float total_d2h_ms = 0.0F;
  std::size_t iterations = 0;

  for (auto _ : state) {
    state.PauseTiming();
    coding::ShardSlots shards = MakeIterationShards();
    state.ResumeTiming();

    CudaDecodeTimings timings;
    run_decode_plan_cuda_profiled(plan_, shards, timings);
    total_h2d_ms += timings.h2d_ms;
    total_kernel_ms += timings.kernel_ms;
    total_d2h_ms += timings.d2h_ms;
    ++iterations;
  }

  PublishCounters(state, total_h2d_ms, total_kernel_ms, total_d2h_ms,
                  iterations);
}

BENCHMARK_DEFINE_F(SparseDecoderBenchmark, KernelOnlyLatency)
(benchmark::State& state) {
  state.PauseTiming();
  std::unique_ptr<DeviceDecodeContext> context =
      prepare_decode_plan_cuda(plan_, initial_shards_);
  constexpr std::size_t kWarmupLaunches = 10;
  for (std::size_t warmup = 0; warmup < kWarmupLaunches; ++warmup) {
    launch_decode_plan_cuda(*context);
  }
  synchronize_decode_plan_cuda(*context);
  std::vector<float> kernel_samples_ms;
  kernel_samples_ms.reserve(100);
  state.ResumeTiming();

  for (auto _ : state) {
    const float kernel_ms = launch_decode_plan_cuda_timed(*context);
    state.SetIterationTime(kernel_ms / 1.0e3);
    kernel_samples_ms.push_back(kernel_ms);
  }

  state.PauseTiming();
  coding::ShardSlots result_shards = initial_shards_;
  download_decode_results_cuda(*context, result_shards);
  for (const XorOperation& operation : plan_.operations) {
    if (result_shards.at(operation.output) !=
        expected_shards_.at(operation.output)) {
      state.SkipWithError("Device-resident decode result is incorrect");
      break;
    }
  }

  std::sort(kernel_samples_ms.begin(), kernel_samples_ms.end());
  const std::size_t middle = kernel_samples_ms.size() / 2;
  const double median_kernel_ms = kernel_samples_ms.size() % 2 == 0
                                      ? (kernel_samples_ms.at(middle - 1) +
                                         kernel_samples_ms.at(middle)) /
                                            2.0
                                      : kernel_samples_ms.at(middle);
  const double bytes_per_kernel =
      static_cast<double>(degree_ + 1) * shard_size_bytes_;
  state.counters["Median Kernel (ms)"] = median_kernel_ms;
  state.counters["Bandwidth (GB/s)"] =
      bytes_per_kernel / (median_kernel_ms / 1.0e3) / 1.0e9;
  state.SetBytesProcessed(
      static_cast<std::int64_t>(kernel_samples_ms.size()) *
      static_cast<std::int64_t>(bytes_per_kernel));
  state.ResumeTiming();
}

constexpr std::int64_t kOneMiB = 1024 * 1024;
constexpr std::int64_t kSixteenMiB = 16 * kOneMiB;

#define REGISTER_DECODER_BENCHMARKS(name)                         \
  BENCHMARK_REGISTER_F(SparseDecoderBenchmark, name)              \
      ->Args({2, kOneMiB})                                        \
      ->Args({2, kSixteenMiB})                                    \
      ->Args({2, kSixteenMiB + 4})                                \
      ->Args({3, kOneMiB})                                        \
      ->Args({3, kSixteenMiB})                                    \
      ->Args({3, kSixteenMiB + 4})                                \
      ->Args({5, kOneMiB})                                        \
      ->Args({5, kSixteenMiB})                                    \
      ->Args({5, kSixteenMiB + 4})                                \
      ->Args({8, kOneMiB})                                        \
      ->Args({8, kSixteenMiB})                                    \
      ->Args({8, kSixteenMiB + 4})

REGISTER_DECODER_BENCHMARKS(EndToEndLatency);
REGISTER_DECODER_BENCHMARKS(KernelOnlyLatency)
    ->UseManualTime()
    ->Iterations(100);

#undef REGISTER_DECODER_BENCHMARKS

void BM_DeviceToDeviceCopy_Roofline(benchmark::State& state) {
  constexpr std::size_t kBufferSizeBytes = 16 * 1024 * 1024;
  cuda_unique_ptr<std::uint8_t> source =
      AllocateDeviceBuffer<std::uint8_t>(kBufferSizeBytes);
  cuda_unique_ptr<std::uint8_t> destination =
      AllocateDeviceBuffer<std::uint8_t>(kBufferSizeBytes);

  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warm up the copy engine and CUDA context before collecting samples.
  CUDA_CHECK(cudaMemsetAsync(source.get(), 0, kBufferSizeBytes, stream));
  CUDA_CHECK(cudaMemcpyAsync(destination.get(), source.get(), kBufferSizeBytes,
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double total_copy_ms = 0.0;
  std::size_t iterations = 0;
  for (auto _ : state) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemcpyAsync(destination.get(), source.get(),
                               kBufferSizeBytes, cudaMemcpyDeviceToDevice,
                               stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    state.SetIterationTime(elapsed_ms / 1.0e3);
    total_copy_ms += elapsed_ms;
    ++iterations;
  }

  // A device copy transfers one buffer read plus one buffer write.
  const double average_seconds =
      (total_copy_ms / static_cast<double>(iterations)) / 1.0e3;
  const double bytes_per_copy =
      2.0 * static_cast<double>(kBufferSizeBytes);
  state.counters["Bandwidth (GB/s)"] =
      bytes_per_copy / average_seconds / 1.0e9;
  state.SetBytesProcessed(static_cast<std::int64_t>(iterations) *
                          static_cast<std::int64_t>(2 * kBufferSizeBytes));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

BENCHMARK(BM_DeviceToDeviceCopy_Roofline)->UseManualTime();

}  // namespace
}  // namespace codedllm

BENCHMARK_MAIN();
