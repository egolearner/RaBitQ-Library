#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::simd {

void flip_sign(const uint8_t* flip, float* data, size_t dim);
void kacs_walk(float* data, size_t len);

}  // namespace rabitqlib::simd
