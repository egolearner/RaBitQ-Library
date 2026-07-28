#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/simd/dispatch.hpp"

namespace rabitqlib::simd {

namespace excode_ipimpl {

#if defined(RABITQ_TARGET_X86_64)
float ip16_fxu1_avx2(const float*, const uint8_t*, size_t);
float ip64_fxu2_avx2(const float*, const uint8_t*, size_t);
float ip64_fxu3_avx2(const float*, const uint8_t*, size_t);
float ip16_fxu4_avx2(const float*, const uint8_t*, size_t);
float ip64_fxu5_avx2(const float*, const uint8_t*, size_t);
float ip64_fxu6_avx2(const float*, const uint8_t*, size_t);
float ip64_fxu7_avx2(const float*, const uint8_t*, size_t);
float ip16_fxu8_avx2(const float*, const uint8_t*, size_t);

float ip16_fxu1_avx512(const float*, const uint8_t*, size_t);
float ip64_fxu2_avx512(const float*, const uint8_t*, size_t);
float ip64_fxu3_avx512(const float*, const uint8_t*, size_t);
float ip16_fxu4_avx512(const float*, const uint8_t*, size_t);
float ip64_fxu5_avx512(const float*, const uint8_t*, size_t);
float ip64_fxu6_avx512(const float*, const uint8_t*, size_t);
float ip64_fxu7_avx512(const float*, const uint8_t*, size_t);
float ip16_fxu8_avx512(const float*, const uint8_t*, size_t);
#endif

#if defined(RABITQ_TARGET_AARCH64)
float ip16_fxu1_neon(const float*, const uint8_t*, size_t);
float ip64_fxu2_neon(const float*, const uint8_t*, size_t);
float ip64_fxu3_neon(const float*, const uint8_t*, size_t);
float ip16_fxu4_neon(const float*, const uint8_t*, size_t);
float ip64_fxu5_neon(const float*, const uint8_t*, size_t);
float ip64_fxu6_neon(const float*, const uint8_t*, size_t);
float ip64_fxu7_neon(const float*, const uint8_t*, size_t);
float ip16_fxu8_neon(const float*, const uint8_t*, size_t);
#endif

#if defined(RABITQ_ENABLE_TEST_BACKENDS)
float ip16_fxu1_scalar(const float*, const uint8_t*, size_t);
float ip64_fxu2_scalar(const float*, const uint8_t*, size_t);
float ip64_fxu3_scalar(const float*, const uint8_t*, size_t);
float ip16_fxu4_scalar(const float*, const uint8_t*, size_t);
float ip64_fxu5_scalar(const float*, const uint8_t*, size_t);
float ip64_fxu6_scalar(const float*, const uint8_t*, size_t);
float ip64_fxu7_scalar(const float*, const uint8_t*, size_t);
float ip16_fxu8_scalar(const float*, const uint8_t*, size_t);
#endif

}  // namespace excode_ipimpl

#define RABITQ_DECLARE_SPACE_BACKEND(suffix)                                                \
    void new_transpose_bin_##suffix(const uint16_t*, uint64_t*, size_t, size_t);            \
    void new_transpose_bin_512_##suffix(const uint8_t*, uint64_t*, size_t, size_t);          \
    float mask_ip_x0_q_##suffix(const float*, const uint64_t*, size_t);                      \
    void scalar_quantize_uint8_##suffix(uint8_t*, const float*, size_t, float, float);       \
    void scalar_quantize_uint16_##suffix(uint16_t*, const float*, size_t, float, float);     \
    void flip_sign_##suffix(const uint8_t*, float*, size_t);                                 \
    void kacs_walk_##suffix(float*, size_t);                                                  \
    void packing_2bit_excode_##suffix(const uint8_t*, uint8_t*, size_t);                     \
    void packing_3bit_excode_##suffix(const uint8_t*, uint8_t*, size_t);                     \
    void packing_4bit_excode_##suffix(const uint8_t*, uint8_t*, size_t);                     \
    void packing_5bit_excode_##suffix(const uint8_t*, uint8_t*, size_t);                     \
    void packing_6bit_excode_##suffix(const uint8_t*, uint8_t*, size_t);                     \
    void packing_7bit_excode_##suffix(const uint8_t*, uint8_t*, size_t);                     \
    float warmup_ip_x0_q_512_##suffix(                                                       \
        const uint64_t*, const uint64_t*, float, float, size_t, size_t)

#if defined(RABITQ_TARGET_X86_64)
RABITQ_DECLARE_SPACE_BACKEND(avx2);
RABITQ_DECLARE_SPACE_BACKEND(avx512);
#endif

#if defined(RABITQ_TARGET_AARCH64)
RABITQ_DECLARE_SPACE_BACKEND(neon);
#endif

#if defined(RABITQ_ENABLE_TEST_BACKENDS)
RABITQ_DECLARE_SPACE_BACKEND(scalar);
#endif

#undef RABITQ_DECLARE_SPACE_BACKEND

}  // namespace rabitqlib::simd

namespace rabitqlib::fastscan::simd {

#define RABITQ_DECLARE_FASTSCAN_BACKEND(suffix)                                              \
    void accumulate_##suffix(const uint8_t*, const uint8_t*, uint16_t*, size_t);             \
    void transfer_lut_hacc_##suffix(const uint16_t*, size_t, uint8_t*);                      \
    void accumulate_hacc_##suffix(const uint8_t*, const uint8_t*, int32_t*, size_t)

#if defined(RABITQ_TARGET_X86_64)
RABITQ_DECLARE_FASTSCAN_BACKEND(avx2);
RABITQ_DECLARE_FASTSCAN_BACKEND(avx512);
#endif

#if defined(RABITQ_TARGET_AARCH64)
RABITQ_DECLARE_FASTSCAN_BACKEND(neon);
#endif

#if defined(RABITQ_ENABLE_TEST_BACKENDS)
RABITQ_DECLARE_FASTSCAN_BACKEND(scalar);
#endif

#undef RABITQ_DECLARE_FASTSCAN_BACKEND

}  // namespace rabitqlib::fastscan::simd
