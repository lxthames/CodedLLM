#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace codedllm {

using ShardId = std::uint32_t;
using WordWidth = std::size_t;

struct ShardLayout {
  ShardLayout(WordWidth word_width_value, std::size_t element_count_value)
      : word_width(word_width_value),
        element_count(element_count_value),
        byte_length(CalculateByteLength(word_width_value, element_count_value)) {}

  const WordWidth word_width;
  const std::size_t element_count;
  const std::size_t byte_length;

 private:
  static std::size_t CalculateByteLength(WordWidth word_width,
                                         std::size_t element_count) {
    if (word_width == 0 || element_count == 0) {
      throw std::invalid_argument("ShardLayout dimensions must be non-zero");
    }
    if (element_count > std::numeric_limits<std::size_t>::max() / word_width) {
      throw std::invalid_argument("ShardLayout byte length overflows size_t");
    }
    return word_width * element_count;
  }
};

}  // namespace codedllm
