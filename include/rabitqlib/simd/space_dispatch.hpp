#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::simd {

void scalar_quantize_uint8(uint8_t* result, const float* vec0, size_t dim, float lo, float delta);
void scalar_quantize_uint16(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
);

}  // namespace rabitqlib::simd
