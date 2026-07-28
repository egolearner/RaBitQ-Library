#include "simd/backend.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>

#include "simd/quantize_utils.hpp"

namespace rabitqlib::simd {
namespace {

uint8_t pack_bits_u16(const uint16_t* values, uint16_t bit) {
    static const uint16_t kWeightsData[8] = {128, 64, 32, 16, 8, 4, 2, 1};
    const uint16x8_t weights = vld1q_u16(kWeightsData);
    const uint16x8_t tested =
        vshrq_n_u16(vtstq_u16(vld1q_u16(values), vdupq_n_u16(bit)), 15);
    return static_cast<uint8_t>(vaddvq_u16(vmulq_u16(tested, weights)));
}

uint8_t pack_bits_u8(const uint8_t* values, uint8_t bit) {
    static const uint8_t kWeightsData[8] = {128, 64, 32, 16, 8, 4, 2, 1};
    const uint8x8_t weights = vld1_u8(kWeightsData);
    const uint8x8_t tested =
        vshr_n_u8(vtst_u8(vld1_u8(values), vdup_n_u8(bit)), 7);
    return static_cast<uint8_t>(vaddlv_u8(vmul_u8(tested, weights)));
}

}  // namespace

void scalar_quantize_uint8_neon(
    uint8_t* result, const float* input, size_t dim, float lo, float delta
) {
    const float32x4_t lo_vec = vdupq_n_f32(lo);
    const float32x4_t inverse_delta = vdupq_n_f32(1.0F / delta);
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const uint32x4_t value0 = vcvtnq_u32_f32(
            vmulq_f32(vsubq_f32(vld1q_f32(input + i), lo_vec), inverse_delta)
        );
        const uint32x4_t value1 = vcvtnq_u32_f32(
            vmulq_f32(vsubq_f32(vld1q_f32(input + i + 4), lo_vec), inverse_delta)
        );
        const uint32x4_t value2 = vcvtnq_u32_f32(
            vmulq_f32(vsubq_f32(vld1q_f32(input + i + 8), lo_vec), inverse_delta)
        );
        const uint32x4_t value3 = vcvtnq_u32_f32(
            vmulq_f32(vsubq_f32(vld1q_f32(input + i + 12), lo_vec), inverse_delta)
        );
        const uint16x8_t value01 =
            vcombine_u16(vqmovn_u32(value0), vqmovn_u32(value1));
        const uint16x8_t value23 =
            vcombine_u16(vqmovn_u32(value2), vqmovn_u32(value3));
        vst1q_u8(result + i, vcombine_u8(vqmovn_u16(value01), vqmovn_u16(value23)));
    }
    const float inverse = 1.0F / delta;
    for (; i < dim; ++i) {
        result[i] = detail::quantize_nearest_even<uint8_t>((input[i] - lo) * inverse);
    }
}

void scalar_quantize_uint16_neon(
    uint16_t* result, const float* input, size_t dim, float lo, float delta
) {
    const float32x4_t lo_vec = vdupq_n_f32(lo);
    const float32x4_t inverse_delta = vdupq_n_f32(1.0F / delta);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const uint32x4_t value0 = vcvtnq_u32_f32(
            vmulq_f32(vsubq_f32(vld1q_f32(input + i), lo_vec), inverse_delta)
        );
        const uint32x4_t value1 = vcvtnq_u32_f32(
            vmulq_f32(vsubq_f32(vld1q_f32(input + i + 4), lo_vec), inverse_delta)
        );
        vst1q_u16(result + i, vcombine_u16(vqmovn_u32(value0), vqmovn_u32(value1)));
    }
    const float inverse = 1.0F / delta;
    for (; i < dim; ++i) {
        result[i] = detail::quantize_nearest_even<uint16_t>((input[i] - lo) * inverse);
    }
}

void new_transpose_bin_neon(
    const uint16_t* query, uint64_t* transposed, size_t padded_dim, size_t b_query
) {
    for (size_t base = 0; base < padded_dim; base += 64) {
        for (size_t bit = 0; bit < b_query; ++bit) {
            uint64_t value = 0;
            for (size_t lane = 0; lane < 64; lane += 8) {
                value = (value << 8U) |
                        pack_bits_u16(query + base + lane, static_cast<uint16_t>(1U << bit));
            }
            transposed[bit] = value;
        }
        transposed += b_query;
    }
}

void new_transpose_bin_512_neon(
    const uint8_t* query, uint64_t* transposed, size_t padded_dim, size_t b_query
) {
    for (size_t base = 0; base < padded_dim;) {
        const size_t block_size = std::min<size_t>(512, padded_dim - base);
        const size_t chunks = block_size / 64;
        for (size_t bit = 0; bit < b_query; ++bit) {
            for (size_t chunk = 0; chunk < chunks; ++chunk) {
                uint64_t value = 0;
                const uint8_t* values = query + base + chunk * 64;
                for (size_t lane = 0; lane < 64; lane += 8) {
                    value = (value << 8U) |
                            pack_bits_u8(values + lane, static_cast<uint8_t>(1U << bit));
                }
                transposed[bit * chunks + chunk] = value;
            }
        }
        base += block_size;
        transposed += chunks * b_query;
    }
}

float mask_ip_x0_q_neon(
    const float* query, const uint64_t* data, size_t padded_dim
) {
    static const uint32_t kBitSelectData[4] = {8, 4, 2, 1};
    const uint32x4_t bit_select = vld1q_u32(kBitSelectData);
    float32x4_t sum = vdupq_n_f32(0.0F);
    for (size_t block = 0; block < padded_dim / 64; ++block) {
        const uint64_t bits = data[block];
        for (size_t lane = 0; lane < 64; lane += 4) {
            const unsigned shift = static_cast<unsigned>(60U - lane);
            const uint32_t nibble = static_cast<uint32_t>((bits >> shift) & 15U);
            const uint32x4_t selected =
                vandq_u32(vdupq_n_u32(nibble), bit_select);
            const uint32x4_t mask = vcgtq_u32(selected, vdupq_n_u32(0));
            sum = vaddq_f32(
                sum, vbslq_f32(mask, vld1q_f32(query + block * 64 + lane), vdupq_n_f32(0.0F))
            );
        }
    }
    return vaddvq_f32(sum);
}

}  // namespace rabitqlib::simd
