#include "codedllm/simulation/recovery_queue.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace codedllm::simulation {
namespace {

void ValidateServiceTime(std::chrono::microseconds service_time) {
  if (service_time <= std::chrono::microseconds::zero()) {
    throw std::invalid_argument("RecoveryQueue service time must be positive");
  }
}

} // namespace

RecoveryQueue::RecoveryQueue(std::size_t concurrency_limit, std::size_t max_queue_depth)
    : slot_available_times_(concurrency_limit, std::chrono::microseconds::zero()),
      max_queue_depth_(max_queue_depth) {
  if (concurrency_limit == 0) {
    throw std::invalid_argument("RecoveryQueue concurrency limit must be non-zero");
  }
}

void RecoveryQueue::AdvanceTime(std::chrono::microseconds now) {
  if (now < current_time_) {
    throw std::invalid_argument("RecoveryQueue time must be monotonic");
  }

  const auto first_pending = job_completion_times_.upper_bound(now);
  const std::size_t completed = static_cast<std::size_t>(
      std::distance(job_completion_times_.begin(), first_pending));
  job_completion_times_.erase(job_completion_times_.begin(), first_pending);
  current_queue_depth_ -= completed;
  current_time_ = now;
}

std::optional<std::chrono::microseconds>
RecoveryQueue::GetExpectedRecoveryCost(std::chrono::microseconds now,
                                       std::chrono::microseconds service_time) const {
  if (now < current_time_) {
    throw std::invalid_argument("RecoveryQueue time must be monotonic");
  }
  ValidateServiceTime(service_time);
  if (current_queue_depth_ >= max_queue_depth_) {
    return std::nullopt;
  }

  const auto earliest_slot =
      std::min_element(slot_available_times_.begin(), slot_available_times_.end());
  const std::chrono::microseconds start_time = std::max(now, *earliest_slot);
  return start_time - now + service_time;
}

bool RecoveryQueue::SubmitJob(std::chrono::microseconds now,
                              std::chrono::microseconds service_time) {
  if (now < current_time_) {
    throw std::invalid_argument("RecoveryQueue time must be monotonic");
  }
  ValidateServiceTime(service_time);
  if (current_queue_depth_ >= max_queue_depth_) {
    return false;
  }

  const auto earliest_slot =
      std::min_element(slot_available_times_.begin(), slot_available_times_.end());
  const std::chrono::microseconds completion_time =
      std::max(now, *earliest_slot) + service_time;
  *earliest_slot = completion_time;
  job_completion_times_.insert(completion_time);
  ++current_queue_depth_;
  return true;
}

} // namespace codedllm::simulation
