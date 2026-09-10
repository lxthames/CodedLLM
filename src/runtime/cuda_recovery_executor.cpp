#include "codedllm/runtime/cuda_recovery_executor.hpp"

#include "kernels/sparse_decoder.cuh"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace codedllm::runtime {
namespace {

std::chrono::microseconds MillisecondsToMicroseconds(float milliseconds) {
  return std::chrono::microseconds(static_cast<std::int64_t>(
      std::llround(static_cast<double>(milliseconds) * 1000.0)));
}

} // namespace

struct CudaRecoveryExecutor::Impl {
  struct Task {
    RequestId request_id;
    DecodePlan plan;
    coding::ShardSlots shards;
    std::unique_ptr<DeviceShardStagingContext> context;
    CompletionHandler completion;
    std::chrono::steady_clock::time_point enqueued_at;
  };

  struct StagedContext {
    std::size_t shard_count;
    std::unique_ptr<DeviceShardStagingContext> context;
  };

  Impl(std::size_t concurrency_limit, std::size_t max_queue_depth,
       std::chrono::microseconds estimated_service_cost_value)
      : concurrency(concurrency_limit), capacity(concurrency_limit + max_queue_depth),
        estimated_service_cost(estimated_service_cost_value) {
    if (concurrency_limit == 0) {
      throw std::invalid_argument("CUDA recovery concurrency must be non-zero");
    }
    if (capacity < concurrency_limit) {
      throw std::invalid_argument("CUDA recovery queue capacity overflows size_t");
    }
    if (estimated_service_cost <= std::chrono::microseconds::zero()) {
      throw std::invalid_argument("CUDA recovery service estimate must be positive");
    }
    workers.reserve(concurrency);
    for (std::size_t worker = 0; worker < concurrency; ++worker) {
      workers.emplace_back([this] { WorkerLoop(); });
    }
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    work_available.notify_all();
    for (std::thread& worker : workers) {
      worker.join();
    }
  }

  std::optional<std::chrono::microseconds> EstimateRecoveryCost() const {
    std::lock_guard<std::mutex> lock(mutex);
    const std::size_t outstanding = queue.size() + running_requests.size();
    if (stopping || outstanding >= capacity) {
      return std::nullopt;
    }
    const std::size_t completion_wave = outstanding / concurrency + 1;
    return estimated_service_cost *
           static_cast<std::chrono::microseconds::rep>(completion_wave);
  }

  void StageShard(RequestId request_id, std::size_t shard_count, ShardId shard_id,
                  const coding::WordShard& payload) {
    std::lock_guard<std::mutex> lock(mutex);
    if (stopping || running_requests.count(request_id) != 0 ||
        HasQueuedRequest(request_id)) {
      return;
    }

    auto context = staged_contexts.find(request_id);
    if (context == staged_contexts.end()) {
      context = staged_contexts
                    .emplace(request_id,
                             StagedContext{shard_count,
                                           create_device_shard_staging_context_cuda(
                                               shard_count)})
                    .first;
    } else if (context->second.shard_count != shard_count) {
      throw std::invalid_argument(
          "Staged shards must use a consistent shard-slot count");
    }
    stage_decode_shard_cuda(*context->second.context, shard_id, payload);
  }

  bool Submit(RequestId request_id, DecodePlan plan, coding::ShardSlots shards,
              CompletionHandler completion) {
    if (!completion) {
      throw std::invalid_argument("CUDA recovery requires a completion handler");
    }

    std::lock_guard<std::mutex> lock(mutex);
    const std::size_t outstanding = queue.size() + running_requests.size();
    if (stopping || outstanding >= capacity ||
        running_requests.count(request_id) != 0 || HasQueuedRequest(request_id)) {
      staged_contexts.erase(request_id);
      return false;
    }

    std::unique_ptr<DeviceShardStagingContext> context;
    const auto staged = staged_contexts.find(request_id);
    if (staged != staged_contexts.end()) {
      if (staged->second.shard_count != shards.size()) {
        staged_contexts.erase(staged);
        throw std::invalid_argument(
            "Submitted shards must match the staged shard-slot count");
      }
      context = std::move(staged->second.context);
      staged_contexts.erase(staged);
    } else {
      context = create_device_shard_staging_context_cuda(shards.size());
      for (std::size_t shard_id = 0; shard_id < shards.size(); ++shard_id) {
        if (shards.at(shard_id).has_value()) {
          stage_decode_shard_cuda(*context, static_cast<ShardId>(shard_id),
                                  *shards.at(shard_id));
        }
      }
    }

    queue.push_back(Task{request_id, std::move(plan), std::move(shards),
                         std::move(context), std::move(completion),
                         std::chrono::steady_clock::now()});
    work_available.notify_one();
    return true;
  }

  bool Cancel(RequestId request_id) {
    std::lock_guard<std::mutex> lock(mutex);
    bool cancelled = staged_contexts.erase(request_id) != 0;
    const auto queued =
        std::find_if(queue.begin(), queue.end(), [request_id](const Task& task) {
          return task.request_id == request_id;
        });
    if (queued != queue.end()) {
      queue.erase(queued);
      return true;
    }
    if (running_requests.count(request_id) != 0) {
      cancelled_requests.insert(request_id);
      return true;
    }
    return cancelled;
  }

  bool HasQueuedRequest(RequestId request_id) const {
    return std::any_of(queue.begin(), queue.end(), [request_id](const Task& task) {
      return task.request_id == request_id;
    });
  }

  void WorkerLoop() {
    while (true) {
      Task task{0, {}, {}, {}, {}, {}};
      {
        std::unique_lock<std::mutex> lock(mutex);
        work_available.wait(lock, [this] { return stopping || !queue.empty(); });
        if (stopping && queue.empty()) {
          return;
        }
        task = std::move(queue.front());
        queue.pop_front();
        running_requests.insert(task.request_id);
      }

      RecoveryTaskResult result;
      const auto execution_start = std::chrono::steady_clock::now();
      result.metrics.queue_wait = std::chrono::duration_cast<std::chrono::microseconds>(
          execution_start - task.enqueued_at);
      try {
        CudaDecodeTimings cuda_timings;
        run_staged_decode_plan_cuda(*task.context, task.plan, task.shards,
                                    cuda_timings);
        result.success = true;
        result.shards = std::move(task.shards);
        result.metrics.h2d = MillisecondsToMicroseconds(cuda_timings.h2d_ms);
        result.metrics.kernel = MillisecondsToMicroseconds(cuda_timings.kernel_ms);
        result.metrics.d2h = MillisecondsToMicroseconds(cuda_timings.d2h_ms);
      } catch (const std::exception& error) {
        result.error = error.what();
      }
      const auto execution_end = std::chrono::steady_clock::now();
      result.metrics.total = result.metrics.queue_wait +
                             std::chrono::duration_cast<std::chrono::microseconds>(
                                 execution_end - execution_start);

      bool cancelled = false;
      {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled = cancelled_requests.count(task.request_id) != 0;
      }
      if (!cancelled) {
        task.completion(std::move(result));
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        running_requests.erase(task.request_id);
        cancelled_requests.erase(task.request_id);
      }
      work_available.notify_all();
    }
  }

  const std::size_t concurrency;
  const std::size_t capacity;
  const std::chrono::microseconds estimated_service_cost;
  mutable std::mutex mutex;
  std::condition_variable work_available;
  std::deque<Task> queue;
  std::unordered_map<RequestId, StagedContext> staged_contexts;
  std::unordered_set<RequestId> running_requests;
  std::unordered_set<RequestId> cancelled_requests;
  std::vector<std::thread> workers;
  bool stopping = false;
};

CudaRecoveryExecutor::CudaRecoveryExecutor(
    std::size_t concurrency_limit, std::size_t max_queue_depth,
    std::chrono::microseconds estimated_service_cost)
    : impl_(std::make_unique<Impl>(concurrency_limit, max_queue_depth,
                                   estimated_service_cost)) {}

CudaRecoveryExecutor::~CudaRecoveryExecutor() = default;

std::optional<std::chrono::microseconds>
CudaRecoveryExecutor::EstimateRecoveryCost() const {
  return impl_->EstimateRecoveryCost();
}

void CudaRecoveryExecutor::StageShard(RequestId request_id,
                                      std::size_t shard_count, ShardId shard_id,
                                      const coding::WordShard& payload) {
  impl_->StageShard(request_id, shard_count, shard_id, payload);
}

bool CudaRecoveryExecutor::Submit(RequestId request_id, DecodePlan plan,
                                  coding::ShardSlots shards,
                                  CompletionHandler completion) {
  return impl_->Submit(request_id, std::move(plan), std::move(shards),
                       std::move(completion));
}

bool CudaRecoveryExecutor::Cancel(RequestId request_id) {
  return impl_->Cancel(request_id);
}

} // namespace codedllm::runtime
