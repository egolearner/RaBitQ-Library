#pragma once

#include <arm_neon.h>

#include <cstddef>

namespace rabitqlib {
namespace detail {

inline void fht_neon(float* buffer, size_t size) {
    for (size_t half_span = 1; half_span < size; half_span *= 2) {
        const size_t span = half_span * 2;
        for (size_t base = 0; base < size; base += span) {
            size_t offset = 0;
            for (; offset + 4 <= half_span; offset += 4) {
                const float32x4_t left = vld1q_f32(buffer + base + offset);
                const float32x4_t right =
                    vld1q_f32(buffer + base + half_span + offset);
                vst1q_f32(buffer + base + offset, vaddq_f32(left, right));
                vst1q_f32(
                    buffer + base + half_span + offset, vsubq_f32(left, right)
                );
            }
            for (; offset < half_span; ++offset) {
                const float left = buffer[base + offset];
                const float right = buffer[base + half_span + offset];
                buffer[base + offset] = left + right;
                buffer[base + half_span + offset] = left - right;
            }
        }
    }
}

}  // namespace detail

inline void helper_float_6(float* buffer) {
    detail::fht_neon(buffer, size_t{1} << 6U);
}

inline void helper_float_7(float* buffer) {
    detail::fht_neon(buffer, size_t{1} << 7U);
}

inline void helper_float_8(float* buffer) {
    detail::fht_neon(buffer, size_t{1} << 8U);
}

inline void helper_float_9(float* buffer) {
    detail::fht_neon(buffer, size_t{1} << 9U);
}

inline void helper_float_10(float* buffer) {
    detail::fht_neon(buffer, size_t{1} << 10U);
}

inline void helper_float_11(float* buffer) {
    detail::fht_neon(buffer, size_t{1} << 11U);
}

}  // namespace rabitqlib
