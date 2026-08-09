#include "simd/backend.hpp"

#include <arm_neon.h>

namespace rabitqlib::simd {
namespace {

template <unsigned Shift>
uint64x2_t pack_top_bit_pairs(uint8x16_t raw) {
    static const uint8_t kWeightsData[16] = {
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128
    };
    const uint8x16_t bits =
        vandq_u8(vshrq_n_u8(raw, Shift), vdupq_n_u8(1));
    const uint8x16_t weighted =
        vmulq_u8(bits, vld1q_u8(kWeightsData));
    return vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(weighted)));
}

template <unsigned Shift>
void pack_top_bits_neon(const uint8_t* raw, uint8_t* compact) {
    const uint64x2_t sum0 = pack_top_bit_pairs<Shift>(vld1q_u8(raw));
    const uint64x2_t sum1 =
        pack_top_bit_pairs<Shift>(vld1q_u8(raw + 16));
    const uint64x2_t sum2 =
        pack_top_bit_pairs<Shift>(vld1q_u8(raw + 32));
    const uint64x2_t sum3 =
        pack_top_bit_pairs<Shift>(vld1q_u8(raw + 48));
    const uint32x4_t sum01 =
        vcombine_u32(vmovn_u64(sum0), vmovn_u64(sum1));
    const uint32x4_t sum23 =
        vcombine_u32(vmovn_u64(sum2), vmovn_u64(sum3));
    const uint16x8_t sums =
        vcombine_u16(vmovn_u32(sum01), vmovn_u32(sum23));
    vst1_u8(compact, vmovn_u16(sums));
}

}  // namespace

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
        pack_top_bits_neon<2>(raw, compact + 16);
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
        pack_top_bits_neon<4>(raw, compact + 32);
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
        pack_top_bits_neon<6>(raw, compact + 48);
        raw += 64;
        compact += 56;
    }
}

}  // namespace rabitqlib::simd
