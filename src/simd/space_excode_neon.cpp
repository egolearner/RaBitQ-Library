#include "simd/backend.hpp"

#include <arm_neon.h>

#include "simd/reference_kernels.hpp"

namespace rabitqlib::simd::excode_ipimpl {
namespace {

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

float packed_ip_neon(
    const float* query, const uint8_t* compact, size_t dim, size_t bits
) {
    uint8_t raw[64];
    float result = 0.0F;
    const size_t compact_stride = bits * 8;
    for (size_t base = 0; base < dim; base += 64) {
        reference::unpack_64(compact, bits, raw);
        result += dot_u8_neon(query + base, raw, 64);
        compact += compact_stride;
    }
    return result;
}

}  // namespace

float ip16_fxu1_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    uint8_t raw[16];
    float result = 0.0F;
    for (size_t base = 0; base < dim; base += 16) {
        for (size_t lane = 0; lane < 16; ++lane) {
            raw[lane] = static_cast<uint8_t>(
                (compact[lane / 8] >> (lane % 8)) & 1U
            );
        }
        result += dot_u8_neon(query + base, raw, 16);
        compact += 2;
    }
    return result;
}

float ip64_fxu2_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    return packed_ip_neon(query, compact, dim, 2);
}

float ip64_fxu3_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    return packed_ip_neon(query, compact, dim, 3);
}

float ip16_fxu4_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    uint8_t raw[16];
    float result = 0.0F;
    for (size_t base = 0; base < dim; base += 16) {
        for (size_t lane = 0; lane < 8; ++lane) {
            raw[lane] = compact[lane] & 15U;
            raw[lane + 8] = compact[lane] >> 4U;
        }
        result += dot_u8_neon(query + base, raw, 16);
        compact += 8;
    }
    return result;
}

float ip64_fxu5_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    return packed_ip_neon(query, compact, dim, 5);
}

float ip64_fxu6_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    return packed_ip_neon(query, compact, dim, 6);
}

float ip64_fxu7_neon(
    const float* query, const uint8_t* compact, size_t dim
) {
    return packed_ip_neon(query, compact, dim, 7);
}

float ip16_fxu8_neon(
    const float* query, const uint8_t* code, size_t dim
) {
    return dot_u8_neon(query, code, dim);
}

}  // namespace rabitqlib::simd::excode_ipimpl
