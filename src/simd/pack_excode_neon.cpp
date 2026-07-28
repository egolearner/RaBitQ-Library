#include "simd/backend.hpp"

#include <arm_neon.h>

#include "simd/reference_kernels.hpp"

namespace rabitqlib::simd {

void packing_2bit_excode_neon(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    for (size_t base = 0; base < dim; base += 64) {
        const uint8x16_t code0 = vld1q_u8(raw);
        const uint8x16_t code1 = vshlq_n_u8(vld1q_u8(raw + 16), 2);
        const uint8x16_t code2 = vshlq_n_u8(vld1q_u8(raw + 32), 4);
        const uint8x16_t code3 = vshlq_n_u8(vld1q_u8(raw + 48), 6);
        vst1q_u8(compact, vorrq_u8(vorrq_u8(code0, code1), vorrq_u8(code2, code3)));
        raw += 64;
        compact += 16;
    }
}

void packing_3bit_excode_neon(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    const uint8x16_t mask = vdupq_n_u8(3);
    for (size_t base = 0; base < dim; base += 64) {
        const uint8x16_t code0 = vandq_u8(vld1q_u8(raw), mask);
        const uint8x16_t code1 =
            vshlq_n_u8(vandq_u8(vld1q_u8(raw + 16), mask), 2);
        const uint8x16_t code2 =
            vshlq_n_u8(vandq_u8(vld1q_u8(raw + 32), mask), 4);
        const uint8x16_t code3 =
            vshlq_n_u8(vandq_u8(vld1q_u8(raw + 48), mask), 6);
        vst1q_u8(compact, vorrq_u8(vorrq_u8(code0, code1), vorrq_u8(code2, code3)));
        reference::pack_top_bit(raw, 2, compact + 16);
        raw += 64;
        compact += 24;
    }
}

void packing_4bit_excode_neon(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    for (size_t base = 0; base < dim; base += 16) {
        const uint8x16_t codes = vld1q_u8(raw);
        const uint8x8_t packed =
            vorr_u8(vget_low_u8(codes), vshl_n_u8(vget_high_u8(codes), 4));
        vst1_u8(compact, packed);
        raw += 16;
        compact += 8;
    }
}

void packing_5bit_excode_neon(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    const uint8x16_t mask = vdupq_n_u8(15);
    for (size_t base = 0; base < dim; base += 64) {
        const uint8x16_t code0 = vandq_u8(vld1q_u8(raw), mask);
        const uint8x16_t code1 =
            vshlq_n_u8(vandq_u8(vld1q_u8(raw + 16), mask), 4);
        const uint8x16_t code2 = vandq_u8(vld1q_u8(raw + 32), mask);
        const uint8x16_t code3 =
            vshlq_n_u8(vandq_u8(vld1q_u8(raw + 48), mask), 4);
        vst1q_u8(compact, vorrq_u8(code0, code1));
        vst1q_u8(compact + 16, vorrq_u8(code2, code3));
        reference::pack_top_bit(raw, 4, compact + 32);
        raw += 64;
        compact += 40;
    }
}

void packing_6bit_excode_neon(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    const uint8x16_t low_mask = vdupq_n_u8(63);
    const uint8x16_t two_bit_mask = vdupq_n_u8(3);
    for (size_t base = 0; base < dim; base += 64) {
        const uint8x16_t tail = vld1q_u8(raw + 48);
        const uint8x16_t tail0 =
            vshlq_n_u8(vandq_u8(tail, two_bit_mask), 6);
        const uint8x16_t tail1 = vshlq_n_u8(
            vandq_u8(vshrq_n_u8(tail, 2), two_bit_mask), 6
        );
        const uint8x16_t tail2 = vshlq_n_u8(
            vandq_u8(vshrq_n_u8(tail, 4), two_bit_mask), 6
        );
        vst1q_u8(
            compact, vorrq_u8(vandq_u8(vld1q_u8(raw), low_mask), tail0)
        );
        vst1q_u8(
            compact + 16,
            vorrq_u8(vandq_u8(vld1q_u8(raw + 16), low_mask), tail1)
        );
        vst1q_u8(
            compact + 32,
            vorrq_u8(vandq_u8(vld1q_u8(raw + 32), low_mask), tail2)
        );
        raw += 64;
        compact += 48;
    }
}

void packing_7bit_excode_neon(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    const uint8x16_t low_mask = vdupq_n_u8(63);
    const uint8x16_t two_bit_mask = vdupq_n_u8(3);
    for (size_t base = 0; base < dim; base += 64) {
        const uint8x16_t tail = vld1q_u8(raw + 48);
        const uint8x16_t tail0 =
            vshlq_n_u8(vandq_u8(tail, two_bit_mask), 6);
        const uint8x16_t tail1 = vshlq_n_u8(
            vandq_u8(vshrq_n_u8(tail, 2), two_bit_mask), 6
        );
        const uint8x16_t tail2 = vshlq_n_u8(
            vandq_u8(vshrq_n_u8(tail, 4), two_bit_mask), 6
        );
        vst1q_u8(
            compact, vorrq_u8(vandq_u8(vld1q_u8(raw), low_mask), tail0)
        );
        vst1q_u8(
            compact + 16,
            vorrq_u8(vandq_u8(vld1q_u8(raw + 16), low_mask), tail1)
        );
        vst1q_u8(
            compact + 32,
            vorrq_u8(vandq_u8(vld1q_u8(raw + 32), low_mask), tail2)
        );
        reference::pack_top_bit(raw, 6, compact + 48);
        raw += 64;
        compact += 56;
    }
}

}  // namespace rabitqlib::simd
