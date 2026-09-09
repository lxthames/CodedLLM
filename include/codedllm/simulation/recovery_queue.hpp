#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <set>
#include <vector>

namespace codedllm::simulation {

class RecoveryQueue {
public:
  RecoveryQueue(std::size_t concurrency_limit, std::size_t max_queue_depth);

  void AdvanceTime(std::chrono::microseconds now);

  [[nodiscard]] std::optional<std::chrono::microseconds>
  GetExpectedRecoveryCost(std::chrono::microseconds now,
                          std::chrono::microseconds service_time) const;

  bool SubmitJob(std::chrono::microseconds now, std::chrono::microseconds service_time);

private:
  std::vector<std::chrono::microseconds> slot_available_times_;
  std::size_t max_queue_depth_;
  std::size_t current_queue_depth_ = 0;
  std::multiset<std::chrono::microseconds> job_completion_times_;
  std::chrono::microseconds current_time_{};
};

} // namespace codedllm::simulation
