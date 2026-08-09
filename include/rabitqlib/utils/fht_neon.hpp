#pragma once

#include <arm_neon.h>

#include <cstddef>

namespace rabitqlib {
namespace detail {

inline float32x4_t fht4(float32x4_t values) {
    const float32x4_t swapped_pairs = vrev64q_f32(values);
    const float32x4_t pair_sums = vaddq_f32(values, swapped_pairs);
    const float32x4_t pair_differences = vsubq_f32(values, swapped_pairs);
    const float32x4_t stage1 = vtrn1q_f32(pair_sums, pair_differences);
    const float32x4_t swapped_halves = vextq_f32(stage1, stage1, 2);
    return vcombine_f32(
        vget_low_f32(vaddq_f32(stage1, swapped_halves)),
        vget_low_f32(vsubq_f32(stage1, swapped_halves))
    );
}

inline void fht16(float* buffer) {
    const float32x4_t value0 = fht4(vld1q_f32(buffer));
    const float32x4_t value1 = fht4(vld1q_f32(buffer + 4));
    const float32x4_t value2 = fht4(vld1q_f32(buffer + 8));
    const float32x4_t value3 = fht4(vld1q_f32(buffer + 12));
    const float32x4_t sum01 = vaddq_f32(value0, value1);
    const float32x4_t difference01 = vsubq_f32(value0, value1);
    const float32x4_t sum23 = vaddq_f32(value2, value3);
    const float32x4_t difference23 = vsubq_f32(value2, value3);
    vst1q_f32(buffer, vaddq_f32(sum01, sum23));
    vst1q_f32(buffer + 4, vaddq_f32(difference01, difference23));
    vst1q_f32(buffer + 8, vsubq_f32(sum01, sum23));
    vst1q_f32(buffer + 12, vsubq_f32(difference01, difference23));
}

template <size_t HalfSpan, size_t Size>
inline void fht_remaining_stages(float* buffer) {
    constexpr size_t kSpan = HalfSpan * 2;
    for (size_t base = 0; base < Size; base += kSpan) {
        for (size_t offset = 0; offset < HalfSpan; offset += 8) {
            const float32x4_t left0 =
                vld1q_f32(buffer + base + offset);
            const float32x4_t left1 =
                vld1q_f32(buffer + base + offset + 4);
            const float32x4_t right0 =
                vld1q_f32(buffer + base + HalfSpan + offset);
            const float32x4_t right1 =
                vld1q_f32(buffer + base + HalfSpan + offset + 4);
            vst1q_f32(
                buffer + base + offset, vaddq_f32(left0, right0)
            );
            vst1q_f32(
                buffer + base + offset + 4, vaddq_f32(left1, right1)
            );
            vst1q_f32(
                buffer + base + HalfSpan + offset,
                vsubq_f32(left0, right0)
            );
            vst1q_f32(
                buffer + base + HalfSpan + offset + 4,
                vsubq_f32(left1, right1)
            );
        }
    }
    if constexpr (kSpan < Size) {
        fht_remaining_stages<kSpan, Size>(buffer);
    }
}

template <size_t Size>
inline void fht_neon(float* buffer) {
    static_assert(Size >= 16 && (Size & (Size - 1)) == 0);
    for (size_t base = 0; base < Size; base += 16) {
        fht16(buffer + base);
    }
    if constexpr (Size > 16) {
        fht_remaining_stages<16, Size>(buffer);
    }
}

}  // namespace detail

inline void helper_float_6(float* buffer) {
    detail::fht_neon<size_t{1} << 6U>(buffer);
}

inline void helper_float_7(float* buffer) {
    detail::fht_neon<size_t{1} << 7U>(buffer);
}

inline void helper_float_8(float* buffer) {
    detail::fht_neon<size_t{1} << 8U>(buffer);
}

inline void helper_float_9(float* buffer) {
    detail::fht_neon<size_t{1} << 9U>(buffer);
}

inline void helper_float_10(float* buffer) {
    detail::fht_neon<size_t{1} << 10U>(buffer);
}

inline void helper_float_11(float* buffer) {
    detail::fht_neon<size_t{1} << 11U>(buffer);
}

}  // namespace rabitqlib
