#include "simd/backend.hpp"

#include <arm_neon.h>

namespace rabitqlib::simd {

void flip_sign_neon(const uint8_t* flip, float* data, size_t dim) {
    static const uint32_t kLowBitsData[4] = {1, 2, 4, 8};
    static const uint32_t kHighBitsData[4] = {16, 32, 64, 128};
    const uint32x4_t low_bits = vld1q_u32(kLowBitsData);
    const uint32x4_t high_bits = vld1q_u32(kHighBitsData);
    const uint32x4_t sign_bit = vdupq_n_u32(0x80000000U);

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const uint32x4_t byte = vdupq_n_u32(flip[i / 8]);
        const uint32x4_t low_mask = vandq_u32(vtstq_u32(byte, low_bits), sign_bit);
        const uint32x4_t high_mask = vandq_u32(vtstq_u32(byte, high_bits), sign_bit);

        uint32x4_t value0 = vreinterpretq_u32_f32(vld1q_f32(data + i));
        uint32x4_t value1 = vreinterpretq_u32_f32(vld1q_f32(data + i + 4));
        vst1q_f32(data + i, vreinterpretq_f32_u32(veorq_u32(value0, low_mask)));
        vst1q_f32(data + i + 4, vreinterpretq_f32_u32(veorq_u32(value1, high_mask)));
    }
    for (; i < dim; ++i) {
        if (((flip[i / 8] >> (i % 8)) & 1U) != 0) {
            data[i] = -data[i];
        }
    }
}

void kacs_walk_neon(float* data, size_t len) {
    const size_t half = len / 2;
    size_t i = 0;
    for (; i + 4 <= half; i += 4) {
        const float32x4_t x = vld1q_f32(data + i);
        const float32x4_t y = vld1q_f32(data + half + i);
        vst1q_f32(data + i, vaddq_f32(x, y));
        vst1q_f32(data + half + i, vsubq_f32(x, y));
    }
    for (; i < half; ++i) {
        const float x = data[i];
        const float y = data[half + i];
        data[i] = x + y;
        data[half + i] = x - y;
    }
}

}  // namespace rabitqlib::simd
