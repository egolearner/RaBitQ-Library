#include "simd/backend.hpp"

#include <arm_neon.h>

#include <algorithm>

namespace rabitqlib::fastscan::simd {
namespace {

void store_permuted_u16(
    uint16x8_t low, uint16x8_t high, uint16_t* result
) {
    vst1q_u16(result, vuzp1q_u16(low, high));
    vst1q_u16(result + 8, vuzp2q_u16(low, high));
}

void accumulate_u16x8_to_u32(
    uint16x8_t values, uint32x4_t& low, uint32x4_t& high
) {
    low = vaddq_u32(low, vmovl_u16(vget_low_u16(values)));
    high = vaddq_u32(high, vmovl_high_u16(values));
}

void store_permuted_u32(
    const uint32x4_t* accumulators, int32_t* result
) {
    vst1q_s32(
        result,
        vreinterpretq_s32_u32(vuzp1q_u32(accumulators[0], accumulators[1]))
    );
    vst1q_s32(
        result + 4,
        vreinterpretq_s32_u32(vuzp1q_u32(accumulators[2], accumulators[3]))
    );
    vst1q_s32(
        result + 8,
        vreinterpretq_s32_u32(vuzp2q_u32(accumulators[0], accumulators[1]))
    );
    vst1q_s32(
        result + 12,
        vreinterpretq_s32_u32(vuzp2q_u32(accumulators[2], accumulators[3]))
    );
}

}  // namespace

void accumulate_neon(
    const uint8_t* codes,
    const uint8_t* lp_table,
    uint16_t* result,
    size_t dim
) {
    const uint8x16_t low_mask = vdupq_n_u8(15);
    uint16x8_t result0_low = vdupq_n_u16(0);
    uint16x8_t result0_high = vdupq_n_u16(0);
    uint16x8_t result1_low = vdupq_n_u16(0);
    uint16x8_t result1_high = vdupq_n_u16(0);

    for (size_t codebook = 0; codebook < dim / 4; ++codebook) {
        const uint8x16_t code = vld1q_u8(codes);
        const uint8x16_t table = vld1q_u8(lp_table);
        const uint8x16_t value0 =
            vqtbl1q_u8(table, vandq_u8(code, low_mask));
        const uint8x16_t value1 =
            vqtbl1q_u8(table, vshrq_n_u8(code, 4));
        result0_low = vaddw_u8(result0_low, vget_low_u8(value0));
        result0_high = vaddw_high_u8(result0_high, value0);
        result1_low = vaddw_u8(result1_low, vget_low_u8(value1));
        result1_high = vaddw_high_u8(result1_high, value1);
        codes += 16;
        lp_table += 16;
    }

    store_permuted_u16(result0_low, result0_high, result);
    store_permuted_u16(result1_low, result1_high, result + 16);
}

void transfer_lut_hacc_neon(
    const uint16_t* lut, size_t dim, uint8_t* high_accuracy_lut
) {
    for (size_t codebook = 0; codebook < dim / 4; ++codebook) {
        const uint16x8_t value0 = vld1q_u16(lut);
        const uint16x8_t value1 = vld1q_u16(lut + 8);
        vst1q_u8(
            high_accuracy_lut,
            vcombine_u8(vmovn_u16(value0), vmovn_u16(value1))
        );
        vst1q_u8(
            high_accuracy_lut + 16,
            vcombine_u8(
                vmovn_u16(vshrq_n_u16(value0, 8)),
                vmovn_u16(vshrq_n_u16(value1, 8))
            )
        );
        lut += 16;
        high_accuracy_lut += 32;
    }
}

void accumulate_hacc_neon(
    const uint8_t* codes,
    const uint8_t* high_accuracy_lut,
    int32_t* result,
    size_t dim
) {
    const uint8x16_t low_mask = vdupq_n_u8(15);
    uint32x4_t result0[4] = {
        vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)
    };
    uint32x4_t result1[4] = {
        vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)
    };

    for (size_t codebook = 0; codebook < dim / 4; ++codebook) {
        const uint8x16_t code = vld1q_u8(codes);
        const uint8x16_t index0 = vandq_u8(code, low_mask);
        const uint8x16_t index1 = vshrq_n_u8(code, 4);
        const uint8x16_t low_table = vld1q_u8(high_accuracy_lut);
        const uint8x16_t high_table = vld1q_u8(high_accuracy_lut + 16);

        const uint8x16_t low0 = vqtbl1q_u8(low_table, index0);
        const uint8x16_t high0 = vqtbl1q_u8(high_table, index0);
        const uint16x8_t value0_low = vorrq_u16(
            vmovl_u8(vget_low_u8(low0)),
            vshlq_n_u16(vmovl_u8(vget_low_u8(high0)), 8)
        );
        const uint16x8_t value0_high = vorrq_u16(
            vmovl_high_u8(low0),
            vshlq_n_u16(vmovl_high_u8(high0), 8)
        );
        accumulate_u16x8_to_u32(value0_low, result0[0], result0[1]);
        accumulate_u16x8_to_u32(value0_high, result0[2], result0[3]);

        const uint8x16_t low1 = vqtbl1q_u8(low_table, index1);
        const uint8x16_t high1 = vqtbl1q_u8(high_table, index1);
        const uint16x8_t value1_low = vorrq_u16(
            vmovl_u8(vget_low_u8(low1)),
            vshlq_n_u16(vmovl_u8(vget_low_u8(high1)), 8)
        );
        const uint16x8_t value1_high = vorrq_u16(
            vmovl_high_u8(low1),
            vshlq_n_u16(vmovl_high_u8(high1), 8)
        );
        accumulate_u16x8_to_u32(value1_low, result1[0], result1[1]);
        accumulate_u16x8_to_u32(value1_high, result1[2], result1[3]);

        codes += 16;
        high_accuracy_lut += 32;
    }

    store_permuted_u32(result0, result);
    store_permuted_u32(result1, result + 16);
}

}  // namespace rabitqlib::fastscan::simd
