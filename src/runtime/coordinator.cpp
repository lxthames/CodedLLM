#include "codedllm/runtime/serving.hpp"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace codedllm::runtime {
namespace {

bool IsTerminal(RequestStatus status) { return status != RequestStatus::Pending; }

std::chrono::microseconds
ElapsedMicroseconds(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

} // namespace

struct RecoveryCoordinator::Impl
    : public std::enable_shared_from_this<RecoveryCoordinator::Impl> {
  struct RequestState {
    RequestState(std::size_t k, std::size_t m, WaitCostOracle oracle)
        : tracker(k, m), shards(k + m), wait_cost_oracle(std::move(oracle)) {}

    ShardArrivalTracker tracker;
    coding::ShardSlots shards;
    WaitCostOracle wait_cost_oracle;
    RequestSnapshot snapshot;
    std::size_t word_count = 0;
    bool recovery_closed = false;
  };

  Impl(coding::BipartiteGraph graph_value, coding::DecodePlanner planner_value,
       std::shared_ptr<RecoveryExecutor> executor_value)
      : graph(std::move(graph_value)), planner(std::move(planner_value)),
        policy(graph, planner), executor(std::move(executor_value)) {
    if (!executor) {
      throw std::invalid_argument("RecoveryCoordinator requires an executor");
    }
  }

  ~Impl() {
    for (const auto& [request_id, state] : requests) {
      static_cast<void>(state);
      executor->Cancel(request_id);
    }
  }

  void AttachTransport(ShardTransport& new_transport) {
    std::lock_guard<std::mutex> lock(mutex);
    if (transport != nullptr) {
      transport->SetArrivalHandler({});
    }
    transport = &new_transport;
    std::weak_ptr<Impl> weak = shared_from_this();
    transport->SetArrivalHandler(
        [weak](RequestId request_id, ShardId shard_id, coding::WordShard payload,
               std::chrono::steady_clock::time_point arrival_time) {
          if (const std::shared_ptr<Impl> impl = weak.lock()) {
            impl->OnShardArrived(request_id, shard_id, std::move(payload),
                                 arrival_time);
          }
        });
  }

  void StartRequest(RequestId request_id, WaitCostOracle wait_cost_oracle) {
    if (!wait_cost_oracle) {
      throw std::invalid_argument("Request requires a wait-cost oracle");
    }
    ShardTransport* active_transport = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (transport == nullptr) {
        throw std::logic_error("No shard transport is attached");
      }
      const auto [request, inserted] = requests.emplace(
          request_id, std::make_shared<RequestState>(graph.systematic_shard_count,
                                                     graph.parity_shard_count,
                                                     std::move(wait_cost_oracle)));
      static_cast<void>(request);
      if (!inserted) {
        throw std::invalid_argument("Request ID is already active");
      }
      active_transport = transport;
    }
    active_transport->StartRequest(request_id);
  }

  void OnShardArrived(RequestId request_id, ShardId shard_id, coding::WordShard payload,
                      std::chrono::steady_clock::time_point arrival_time) {
    DecodePlan plan;
    coding::ShardSlots task_shards;
    std::shared_ptr<RequestState> state;
    std::chrono::microseconds estimated_cost{};
    bool should_submit = false;
    bool should_cancel = false;

    {
      std::lock_guard<std::mutex> lock(mutex);
      const auto request = requests.find(request_id);
      if (request == requests.end()) {
        throw std::invalid_argument("Arrival references an unknown request");
      }
      state = request->second;
      if (IsTerminal(state->snapshot.status)) {
        ++state->snapshot.late_arrivals;
        return;
      }
      if (shard_id >= state->shards.size()) {
        throw std::invalid_argument("Arrival shard ID is out of bounds");
      }
      if (payload.empty()) {
        throw std::invalid_argument("Arrived shards must not be empty");
      }
      if (state->word_count == 0) {
        state->word_count = payload.size();
      } else if (state->word_count != payload.size()) {
        throw std::invalid_argument("Arrived shards must have equal sizes");
      }
      if (state->shards.at(shard_id).has_value()) {
        return;
      }

      executor->StageShard(request_id, state->shards.size(), shard_id, payload);
      state->shards.at(shard_id) = std::move(payload);
      state->tracker.MarkArrived(shard_id, arrival_time);
      if (state->tracker.IsSystematicComplete()) {
        state->snapshot.status = RequestStatus::NaturalComplete;
        should_cancel = true;
        condition.notify_all();
      } else if (!state->recovery_closed && !state->snapshot.recovery_submitted) {
        std::unordered_set<ShardId> available_parity;
        for (const auto& [parity, systematic] : graph.parity_to_systematic) {
          static_cast<void>(systematic);
          if (state->tracker.HasArrived(parity)) {
            available_parity.insert(parity);
          }
        }

        const auto planning_start = std::chrono::steady_clock::now();
        plan = planner.CreatePlan(graph, state->tracker.GetMissingSystematic(),
                                  available_parity);
        const auto planning_end = std::chrono::steady_clock::now();
        state->snapshot.metrics.planning +=
            ElapsedMicroseconds(planning_start, planning_end);
        if (plan.is_recoverable) {
          const std::optional<std::chrono::microseconds> estimate =
              executor->EstimateRecoveryCost();
          if (!estimate.has_value()) {
            state->snapshot.recovery_rejected = true;
            state->recovery_closed = true;
          } else {
            estimated_cost = *estimate;
            const CostEstimates costs{state->wait_cost_oracle(request_id, arrival_time),
                                      estimated_cost};
            if (policy.Evaluate(state->tracker, costs) == PolicyDecision::Recover) {
              state->snapshot.recovery_submitted = true;
              task_shards = state->shards;
              should_submit = true;
            }
          }
        }
      }
    }

    if (should_cancel) {
      executor->Cancel(request_id);
      return;
    }
    if (!should_submit) {
      return;
    }

    std::weak_ptr<Impl> weak = shared_from_this();
    const bool accepted =
        executor->Submit(request_id, std::move(plan), std::move(task_shards),
                         [weak, request_id](RecoveryTaskResult result) {
                           if (const std::shared_ptr<Impl> impl = weak.lock()) {
                             impl->OnRecoveryComplete(request_id, std::move(result));
                           }
                         });
    if (accepted) {
      bool completed_naturally = false;
      {
        std::lock_guard<std::mutex> lock(mutex);
        completed_naturally =
            state->snapshot.status == RequestStatus::NaturalComplete;
      }
      if (completed_naturally) {
        executor->Cancel(request_id);
      }
      return;
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (!IsTerminal(state->snapshot.status)) {
      state->snapshot.recovery_submitted = false;
      state->snapshot.recovery_rejected = true;
      state->recovery_closed = true;
    }
  }

  void OnRecoveryComplete(RequestId request_id, RecoveryTaskResult result) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto request = requests.find(request_id);
    if (request == requests.end() || IsTerminal(request->second->snapshot.status)) {
      return;
    }
    RequestState& state = *request->second;
    state.snapshot.metrics.queue_wait = result.metrics.queue_wait;
    state.snapshot.metrics.h2d = result.metrics.h2d;
    state.snapshot.metrics.kernel = result.metrics.kernel;
    state.snapshot.metrics.d2h = result.metrics.d2h;
    state.snapshot.metrics.handoff = result.metrics.handoff;
    state.snapshot.metrics.total =
        state.snapshot.metrics.planning + result.metrics.total;
    if (!result.success) {
      state.snapshot.status = RequestStatus::Failed;
      state.snapshot.error = std::move(result.error);
    } else {
      state.shards = std::move(result.shards);
      state.snapshot.status = RequestStatus::Recovered;
    }
    condition.notify_all();
  }

  RequestSnapshot GetSnapshot(RequestId request_id) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto request = requests.find(request_id);
    if (request == requests.end()) {
      throw std::invalid_argument("Unknown request ID");
    }
    return request->second->snapshot;
  }

  std::optional<coding::WordShard> GetShard(RequestId request_id,
                                            ShardId shard_id) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto request = requests.find(request_id);
    if (request == requests.end() || shard_id >= request->second->shards.size()) {
      throw std::invalid_argument("Unknown request or shard ID");
    }
    return request->second->shards.at(shard_id);
  }

  bool WaitForCompletion(RequestId request_id,
                         std::chrono::milliseconds timeout) const {
    std::unique_lock<std::mutex> lock(mutex);
    const auto request = requests.find(request_id);
    if (request == requests.end()) {
      throw std::invalid_argument("Unknown request ID");
    }
    return condition.wait_for(lock, timeout, [&request] {
      return IsTerminal(request->second->snapshot.status);
    });
  }

  const coding::BipartiteGraph graph;
  const coding::DecodePlanner planner;
  const RecoveryPolicy policy;
  const std::shared_ptr<RecoveryExecutor> executor;
  mutable std::mutex mutex;
  mutable std::condition_variable condition;
  std::unordered_map<RequestId, std::shared_ptr<RequestState>> requests;
  ShardTransport* transport = nullptr;
};

RecoveryCoordinator::RecoveryCoordinator(coding::BipartiteGraph graph,
                                         coding::DecodePlanner planner,
                                         std::shared_ptr<RecoveryExecutor> executor)
    : impl_(std::make_shared<Impl>(std::move(graph), std::move(planner),
                                   std::move(executor))) {}

RecoveryCoordinator::~RecoveryCoordinator() { impl_.reset(); }

void RecoveryCoordinator::AttachTransport(ShardTransport& transport) {
  impl_->AttachTransport(transport);
}

void RecoveryCoordinator::StartRequest(RequestId request_id,
                                       WaitCostOracle wait_cost_oracle) {
  impl_->StartRequest(request_id, std::move(wait_cost_oracle));
}

RequestSnapshot RecoveryCoordinator::GetSnapshot(RequestId request_id) const {
  return impl_->GetSnapshot(request_id);
}

std::optional<coding::WordShard> RecoveryCoordinator::GetShard(RequestId request_id,
                                                               ShardId shard_id) const {
  return impl_->GetShard(request_id, shard_id);
}

bool RecoveryCoordinator::WaitForCompletion(RequestId request_id,
                                            std::chrono::milliseconds timeout) const {
  return impl_->WaitForCompletion(request_id, timeout);
}

} // namespace codedllm::runtime
