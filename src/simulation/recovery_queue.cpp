#include "codedllm/simulation/recovery_queue.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace codedllm::simulation {

RecoveryQueue::RecoveryQueue(std::size_t concurrency_limit, std::size_t max_queue_depth,
                             std::chrono::microseconds decode_time_per_job)
    : slot_available_times_(concurrency_limit, std::chrono::microseconds::zero()),
      max_queue_depth_(max_queue_depth), decode_time_per_job_(decode_time_per_job) {
  if (concurrency_limit == 0) {
    throw std::invalid_argument("RecoveryQueue concurrency limit must be non-zero");
  }
  if (decode_time_per_job <= std::chrono::microseconds::zero()) {
    throw std::invalid_argument("RecoveryQueue decode time must be positive");
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
RecoveryQueue::GetExpectedRecoveryCost(std::chrono::microseconds now) const {
  if (now < current_time_) {
    throw std::invalid_argument("RecoveryQueue time must be monotonic");
  }
  if (current_queue_depth_ >= max_queue_depth_) {
    return std::nullopt;
  }

  const auto earliest_slot =
      std::min_element(slot_available_times_.begin(), slot_available_times_.end());
  const std::chrono::microseconds start_time = std::max(now, *earliest_slot);
  return start_time - now + decode_time_per_job_;
}

bool RecoveryQueue::SubmitJob(std::chrono::microseconds now) {
  if (now < current_time_) {
    throw std::invalid_argument("RecoveryQueue time must be monotonic");
  }
  if (current_queue_depth_ >= max_queue_depth_) {
    return false;
  }

  const auto earliest_slot =
      std::min_element(slot_available_times_.begin(), slot_available_times_.end());
  const std::chrono::microseconds completion_time =
      std::max(now, *earliest_slot) + decode_time_per_job_;
  *earliest_slot = completion_time;
  job_completion_times_.insert(completion_time);
  ++current_queue_depth_;
  return true;
}

} // namespace codedllm::simulation
