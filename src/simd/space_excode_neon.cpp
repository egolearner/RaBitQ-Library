#include "simd/backend.hpp"

#include <arm_neon.h>

namespace rabitqlib::simd::excode_ipimpl {
namespace {

struct FloatAccumulator {
    float32x4_t values[4] = {
        vdupq_n_f32(0.0F),
        vdupq_n_f32(0.0F),
        vdupq_n_f32(0.0F),
        vdupq_n_f32(0.0F),
    };
};

void contribute_ip(
    uint8x16_t code, const float* query, FloatAccumulator& accumulator
) {
    const uint16x8_t code_low = vmovl_u8(vget_low_u8(code));
    const uint16x8_t code_high = vmovl_high_u8(code);
    accumulator.values[0] = vfmaq_f32(
        accumulator.values[0],
        vld1q_f32(query),
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(code_low)))
    );
    accumulator.values[1] = vfmaq_f32(
        accumulator.values[1],
        vld1q_f32(query + 4),
        vcvtq_f32_u32(vmovl_high_u16(code_low))
    );
    accumulator.values[2] = vfmaq_f32(
        accumulator.values[2],
        vld1q_f32(query + 8),
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(code_high)))
    );
    accumulator.values[3] = vfmaq_f32(
        accumulator.values[3],
        vld1q_f32(query + 12),
        vcvtq_f32_u32(vmovl_high_u16(code_high))
    );
}

float reduce(const FloatAccumulator& accumulator) {
    return vaddvq_f32(
        vaddq_f32(
            vaddq_f32(accumulator.values[0], accumulator.values[1]),
            vaddq_f32(accumulator.values[2], accumulator.values[3])
        )
    );
}

template <unsigned Shift, size_t Group>
uint8x16_t expand_top_bits(const uint8_t* compact) {
    static const uint8_t kBitSelectData[16] = {
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128
    };
    static const uint8_t kByteIndices[4][16] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
        {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5},
        {6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7},
    };
    const uint8x16_t packed =
        vcombine_u8(vld1_u8(compact), vdup_n_u8(0));
    const uint8x16_t repeated =
        vqtbl1q_u8(packed, vld1q_u8(kByteIndices[Group]));
    const uint8x16_t selected =
        vtstq_u8(repeated, vld1q_u8(kBitSelectData));
    return vandq_u8(selected, vdupq_n_u8(uint8_t{1} << Shift));
}

template <unsigned Shift>
void contribute_top_bits(
    const uint8_t* compact,
    const float* query,
    uint8x16_t code0,
    uint8x16_t code1,
    uint8x16_t code2,
    uint8x16_t code3,
    FloatAccumulator& accumulator
) {
    contribute_ip(
        vorrq_u8(code0, expand_top_bits<Shift, 0>(compact)),
        query,
        accumulator
    );
    contribute_ip(
        vorrq_u8(code1, expand_top_bits<Shift, 1>(compact)),
        query + 16,
        accumulator
    );
    contribute_ip(
        vorrq_u8(code2, expand_top_bits<Shift, 2>(compact)),
        query + 32,
        accumulator
    );
    contribute_ip(
        vorrq_u8(code3, expand_top_bits<Shift, 3>(compact)),
        query + 48,
        accumulator
    );
}

float dot_u8_neon(const float* query, const uint8_t* code, size_t dim) {
    const uint8x16_t index0 = {
        0, 255, 255, 255, 1, 255, 255, 255, 2, 255, 255, 255, 3, 255, 255, 255
    };
    const uint8x16_t index1 = {
        4, 255, 255, 255, 5, 255, 255, 255, 6, 255, 255, 255, 7, 255, 255, 255
    };
    const uint8x16_t index2 = {
        8, 255, 255, 255, 9, 255, 255, 255, 10, 255, 255, 255, 11, 255, 255, 255
    };
    const uint8x16_t index3 = {
        12, 255, 255, 255, 13, 255, 255, 255, 14, 255, 255, 255, 15, 255, 255, 255
    };
    float32x4_t sum0 = vdupq_n_f32(0.0F);
    float32x4_t sum1 = vdupq_n_f32(0.0F);
    float32x4_t sum2 = vdupq_n_f32(0.0F);
    float32x4_t sum3 = vdupq_n_f32(0.0F);
    float32x4_t sum4 = vdupq_n_f32(0.0F);
    float32x4_t sum5 = vdupq_n_f32(0.0F);
    float32x4_t sum6 = vdupq_n_f32(0.0F);
    float32x4_t sum7 = vdupq_n_f32(0.0F);
    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const uint8x16_t code_low = vld1q_u8(code + i);
        const uint8x16_t code_high = vld1q_u8(code + i + 16);
        sum0 = vfmaq_f32(
            sum0,
            vld1q_f32(query + i),
            vcvtq_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(code_low, index0)))
        );
        sum1 = vfmaq_f32(
            sum1,
            vld1q_f32(query + i + 4),
            vcvtq_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(code_low, index1)))
        );
        sum2 = vfmaq_f32(
            sum2,
            vld1q_f32(query + i + 8),
            vcvtq_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(code_low, index2)))
        );
        sum3 = vfmaq_f32(
            sum3,
            vld1q_f32(query + i + 12),
            vcvtq_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(code_low, index3)))
        );
        sum4 = vfmaq_f32(
            sum4,
            vld1q_f32(query + i + 16),
            vcvtq_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(code_high, index0)))
        );
        sum5 = vfmaq_f32(
            sum5,
            vld1q_f32(query + i + 20),
            vcvtq_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(code_high, index1)))
        );
        sum6 = vfmaq_f32(
            sum6,
            vld1q_f32(query + i + 24),
            vcvtq_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(code_high, index2)))
        );
        sum7 = vfmaq_f32(
            sum7,
            vld1q_f32(query + i + 28),
            vcvtq_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(code_high, index3)))
        );
    }
    const float32x4_t sum01 = vaddq_f32(sum0, sum1);
    const float32x4_t sum23 = vaddq_f32(sum2, sum3);
    const float32x4_t sum45 = vaddq_f32(sum4, sum5);
    const float32x4_t sum67 = vaddq_f32(sum6, sum7);
    float result = vaddvq_f32(
        vaddq_f32(vaddq_f32(sum01, sum23), vaddq_f32(sum45, sum67))
    );
    for (; i < dim; ++i) {
        result += query[i] * static_cast<float>(code[i]);
    }
    return result;
}

}  // namespace

float ip16_fxu1_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    static const uint8_t kBitSelectData[16] = {
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128
    };
    const uint8x16_t bit_select = vld1q_u8(kBitSelectData);
    FloatAccumulator accumulator;
    for (size_t base = 0; base < dim; base += 16) {
        const uint8x16_t packed = vcombine_u8(
            vdup_n_u8(compact[0]), vdup_n_u8(compact[1])
        );
        const uint8x16_t code =
            vshrq_n_u8(vtstq_u8(packed, bit_select), 7);
        contribute_ip(code, query + base, accumulator);
        compact += 2;
    }
    return reduce(accumulator);
}

float ip64_fxu2_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    const uint8x16_t mask = vdupq_n_u8(3);
    FloatAccumulator accumulator;
    for (size_t base = 0; base < dim; base += 64) {
        const uint8x16_t packed = vld1q_u8(compact);
        contribute_ip(vandq_u8(packed, mask), query + base, accumulator);
        contribute_ip(
            vandq_u8(vshrq_n_u8(packed, 2), mask),
            query + base + 16,
            accumulator
        );
        contribute_ip(
            vandq_u8(vshrq_n_u8(packed, 4), mask),
            query + base + 32,
            accumulator
        );
        contribute_ip(
            vshrq_n_u8(packed, 6), query + base + 48, accumulator
        );
        compact += 16;
    }
    return reduce(accumulator);
}

float ip64_fxu3_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    const uint8x16_t mask = vdupq_n_u8(3);
    FloatAccumulator accumulator;
    for (size_t base = 0; base < dim; base += 64) {
        const uint8x16_t packed = vld1q_u8(compact);
        contribute_top_bits<2>(
            compact + 16,
            query + base,
            vandq_u8(packed, mask),
            vandq_u8(vshrq_n_u8(packed, 2), mask),
            vandq_u8(vshrq_n_u8(packed, 4), mask),
            vshrq_n_u8(packed, 6),
            accumulator
        );
        compact += 24;
    }
    return reduce(accumulator);
}

float ip16_fxu4_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    const uint8x8_t mask = vdup_n_u8(15);
    FloatAccumulator accumulator;
    for (size_t base = 0; base < dim; base += 16) {
        const uint8x8_t packed = vld1_u8(compact);
        contribute_ip(
            vcombine_u8(vand_u8(packed, mask), vshr_n_u8(packed, 4)),
            query + base,
            accumulator
        );
        compact += 8;
    }
    return reduce(accumulator);
}

float ip64_fxu5_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    const uint8x16_t mask = vdupq_n_u8(15);
    FloatAccumulator accumulator;
    for (size_t base = 0; base < dim; base += 64) {
        const uint8x16_t packed0 = vld1q_u8(compact);
        const uint8x16_t packed1 = vld1q_u8(compact + 16);
        contribute_top_bits<4>(
            compact + 32,
            query + base,
            vandq_u8(packed0, mask),
            vshrq_n_u8(packed0, 4),
            vandq_u8(packed1, mask),
            vshrq_n_u8(packed1, 4),
            accumulator
        );
        compact += 40;
    }
    return reduce(accumulator);
}

float ip64_fxu6_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    const uint8x16_t mask = vdupq_n_u8(63);
    FloatAccumulator accumulator;
    for (size_t base = 0; base < dim; base += 64) {
        const uint8x16_t packed0 = vld1q_u8(compact);
        const uint8x16_t packed1 = vld1q_u8(compact + 16);
        const uint8x16_t packed2 = vld1q_u8(compact + 32);
        contribute_ip(
            vandq_u8(packed0, mask), query + base, accumulator
        );
        contribute_ip(
            vandq_u8(packed1, mask), query + base + 16, accumulator
        );
        contribute_ip(
            vandq_u8(packed2, mask), query + base + 32, accumulator
        );
        contribute_ip(
            vorrq_u8(
                vorrq_u8(
                    vshrq_n_u8(packed0, 6),
                    vshlq_n_u8(vshrq_n_u8(packed1, 6), 2)
                ),
                vshlq_n_u8(vshrq_n_u8(packed2, 6), 4)
            ),
            query + base + 48,
            accumulator
        );
        compact += 48;
    }
    return reduce(accumulator);
}

float ip64_fxu7_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    const uint8x16_t mask = vdupq_n_u8(63);
    FloatAccumulator accumulator;
    for (size_t base = 0; base < dim; base += 64) {
        const uint8x16_t packed0 = vld1q_u8(compact);
        const uint8x16_t packed1 = vld1q_u8(compact + 16);
        const uint8x16_t packed2 = vld1q_u8(compact + 32);
        const uint8x16_t code3 = vorrq_u8(
            vorrq_u8(
                vshrq_n_u8(packed0, 6),
                vshlq_n_u8(vshrq_n_u8(packed1, 6), 2)
            ),
            vshlq_n_u8(vshrq_n_u8(packed2, 6), 4)
        );
        contribute_top_bits<6>(
            compact + 48,
            query + base,
            vandq_u8(packed0, mask),
            vandq_u8(packed1, mask),
            vandq_u8(packed2, mask),
            code3,
            accumulator
        );
        compact += 56;
    }
    return reduce(accumulator);
}

float ip16_fxu8_neon(
    const float* query, const uint8_t* code, size_t dim
) {
    return dot_u8_neon(query, code, dim);
}

}  // namespace rabitqlib::simd::excode_ipimpl
