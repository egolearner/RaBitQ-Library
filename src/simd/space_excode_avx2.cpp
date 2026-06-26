#include <immintrin.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::simd::excode_ipimpl {

#if defined(__AVX2__)
// helper function for AVX2 inner product
inline void contribute_ip(__m128i vec, const float* __restrict__ query, __m256& sum) {
    /* // Equivalent AVX512 code:
        __m512 q = _mm512_loadu_ps(&query[i]);
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);
    */
    __m256 q = _mm256_loadu_ps(query);
    __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vec));
    sum = _mm256_fmadd_ps(q, cf, sum);

    q = _mm256_loadu_ps(query + 8);
    cf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(vec, 8)));
    sum = _mm256_fmadd_ps(q, cf, sum);
};

inline void contribute_ip_signed(
    __m128i vec, const float* __restrict__ query, __m256& sum
) {
    /* // Equivalent AVX512 code:
        __m512 q = _mm512_loadu_ps(&query[i]);
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c8));
        sum = _mm512_fmadd_ps(cf, q, sum);
    */
    __m256 q = _mm256_loadu_ps(query);
    __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vec));
    sum = _mm256_fmadd_ps(cf, q, sum);

    q = _mm256_loadu_ps(query + 8);
    cf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vec, 8)));
    sum = _mm256_fmadd_ps(cf, q, sum);
};

inline float mm256_reduce_add_ps(__m256 v) {
    std::array<float, 8> accumulator{};
    _mm256_storeu_ps(accumulator.data(), v);
    float result = 0.0F;
    for (const auto& i : accumulator) {
        result += i;
    }
    return result;
}
#endif

// ip16: this function is used to compute inner product of
// vectors padded to multiple of 16
// fxu1: the inner product is computed between float and 1-bit unsigned int (lay out can be
// found rabitq_impl.hpp)
// avx512: only applicable for avx512
float ip16_fxu1_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    float result = 0;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    for (size_t i = 0; i < dim; i += 16) {
        __mmask16 mask = *reinterpret_cast<const __mmask16*>(compact_code);
        __m512 q = _mm512_loadu_ps(query);

        sum = _mm512_add_ps(_mm512_maskz_mov_ps(mask, q), sum);

        compact_code += 2;
        query += 16;
    }
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();

    const __m256i bitmask = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

    for (size_t i = 0; i < dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query);

        __m256i byte_v = _mm256_set1_epi32(*compact_code);
        __m256i isolated = _mm256_and_si256(byte_v, bitmask);
        __m256i mask = _mm256_cmpeq_epi32(isolated, bitmask);
        __m256 masked = _mm256_and_ps(q, _mm256_castsi256_ps(mask));

        sum = _mm256_add_ps(sum, masked);
        query += 8;
        ++compact_code;
    }
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

float ip64_fxu2_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif
    float result = 0;
    const __m128i mask = _mm_set1_epi8(0b00000011);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));

        __m128i vec_00_to_15 = _mm_and_si128(compact, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact, 6), mask);
#if defined(__AVX512F__)
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
#elif defined(__AVX2__)
        contribute_ip(vec_00_to_15, &query[i], sum);
        contribute_ip(vec_16_to_31, &query[i + 16], sum);
        contribute_ip(vec_32_to_47, &query[i + 32], sum);
        contribute_ip(vec_48_to_63, &query[i + 48], sum);
#endif
        compact_code += 16;
    }

#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

float ip64_fxu3_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif
    float result = 0;
    const __m128i mask = _mm_set1_epi8(0b11);
    const __m128i top_mask = _mm_set1_epi8(0b100);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        compact_code += 16;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact2, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact2, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact2, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact2, 6), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 5, top_bit >> 4), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);
#if defined(__AVX512F__)
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
#elif defined(__AVX2__)
        contribute_ip(vec_00_to_15, &query[i], sum);
        contribute_ip(vec_16_to_31, &query[i + 16], sum);
        contribute_ip(vec_32_to_47, &query[i + 32], sum);
        contribute_ip(vec_48_to_63, &query[i + 48], sum);
#endif
    }

#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

float ip16_fxu4_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif
    float result = 0.0F;
    constexpr int64_t kMask = 0x0f0f0f0f0f0f0f0f;
    for (size_t i = 0; i < dim; i += 16) {
        int64_t compact = *reinterpret_cast<const int64_t*>(compact_code);
        int64_t code0 = compact & kMask;
        int64_t code1 = (compact >> 4) & kMask;

        __m128i c8 = _mm_set_epi64x(code1, code0);
#if defined(__AVX512F__)
        __m512 q = _mm512_loadu_ps(&query[i]);
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c8));
        sum = _mm512_fmadd_ps(cf, q, sum);
#elif defined(__AVX2__)
        contribute_ip_signed(c8, &query[i], sum);
#endif
        compact_code += 8;
    }
#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

float ip64_fxu5_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif

    float result = 0.0F;
    const __m128i mask = _mm_set1_epi8(0b1111);
    const __m128i top_mask = _mm_set1_epi8(0b10000);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact4_1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i compact4_2 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        compact_code += 32;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact4_1, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact4_1, 4), mask);
        __m128i vec_32_to_47 = _mm_and_si128(compact4_2, mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact4_2, 4), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

#if defined(__AVX512F__)
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
#elif defined(__AVX2__)
        contribute_ip(vec_00_to_15, &query[i], sum);
        contribute_ip(vec_16_to_31, &query[i + 16], sum);
        contribute_ip(vec_32_to_47, &query[i + 32], sum);
        contribute_ip(vec_48_to_63, &query[i + 48], sum);
#endif
    }
#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

float ip64_fxu6_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif
    float result = 0.0F;
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));

    for (size_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 32));

        compact_code += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

#if defined(__AVX512F__)
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
#elif defined(__AVX2__)
        contribute_ip(vec_00_to_15, &query[i], sum);
        contribute_ip(vec_16_to_31, &query[i + 16], sum);
        contribute_ip(vec_32_to_47, &query[i + 32], sum);
        contribute_ip(vec_48_to_63, &query[i + 48], sum);
#endif
    }
#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

float ip64_fxu7_avx2(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif

    float result = 0.0F;
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));
    const __m128i top_mask = _mm_set1_epi8(0b1000000);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 32));
        compact_code += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 5, top_bit << 6), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit << 0), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

#if defined(__AVX512F__)
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
#elif defined(__AVX2__)
        contribute_ip(vec_00_to_15, &query[i], sum);
        contribute_ip(vec_16_to_31, &query[i + 16], sum);
        contribute_ip(vec_32_to_47, &query[i + 32], sum);
        contribute_ip(vec_48_to_63, &query[i + 48], sum);
#endif
    }

#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

}  // namespace rabitqlib::simd::excode_ipimpl
