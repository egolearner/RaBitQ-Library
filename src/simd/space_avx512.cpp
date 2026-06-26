#include <immintrin.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::simd {

void new_transpose_bin_avx512(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    // Easy
#if defined(__AVX512BW__)
    // 512 / 16 = 32
    for (size_t i = 0; i < padded_dim; i += 64) {
        __m512i vec_00_to_31 = _mm512_loadu_si512(q);
        __m512i vec_32_to_63 = _mm512_loadu_si512(q + 32);

        // the first (16 - b_query) bits are empty
        vec_00_to_31 = _mm512_slli_epi32(vec_00_to_31, (16 - b_query));
        vec_32_to_63 = _mm512_slli_epi32(vec_32_to_63, (16 - b_query));

        for (size_t j = 0; j < b_query; ++j) {
            uint32_t v0 = _mm512_movepi16_mask(vec_00_to_31);  // get most significant bit
            uint32_t v1 = _mm512_movepi16_mask(vec_32_to_63);  // get most significant bit
            // [TODO: remove all reverse_bits]
            v0 = reverse_bits(v0);
            v1 = reverse_bits(v1);
            uint64_t v = (static_cast<uint64_t>(v0) << 32) + v1;

            tq[b_query - j - 1] = v;

            vec_00_to_31 = _mm512_slli_epi16(vec_00_to_31, 1);
            vec_32_to_63 = _mm512_slli_epi16(vec_32_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i < padded_dim; i += 64) {
        __m256i vec_00_to_15 = _mm256_loadu_si256((__m256i const*)(q));
        __m256i vec_16_to_31 = _mm256_loadu_si256((__m256i const*)(q + 16));
        __m256i vec_32_to_47 = _mm256_loadu_si256((__m256i const*)(q + 32));
        __m256i vec_48_to_63 = _mm256_loadu_si256((__m256i const*)(q + 48));

        // the first (16 - b_query) bits are empty
        vec_00_to_15 = _mm256_slli_epi32(vec_00_to_15, (16 - b_query));
        vec_16_to_31 = _mm256_slli_epi32(vec_16_to_31, (16 - b_query));
        vec_32_to_47 = _mm256_slli_epi32(vec_32_to_47, (16 - b_query));
        vec_48_to_63 = _mm256_slli_epi32(vec_48_to_63, (16 - b_query));

        for (size_t j = 0; j < b_query; ++j) {
            // pack two 16-bit vectors to 8-bit interleaved vectors
            __m256i p0 = _mm256_packs_epi16(vec_00_to_15, vec_16_to_31);
            __m256i p1 = _mm256_packs_epi16(vec_32_to_47, vec_48_to_63);

            uint32_t m0 = _mm256_movemask_epi8(p0);
            uint32_t m1 = _mm256_movemask_epi8(p1);

            // Fix AVX2 Lane Ordering of the interleaved mask
            auto fix_avx2_mask = [](uint32_t m) {
                return (m & 0xFF0000FF) | ((m & 0x00FF0000) >> 8) | ((m & 0x0000FF00) << 8);
            };

            m0 = fix_avx2_mask(m0);
            m1 = fix_avx2_mask(m1);

            m0 = reverse_bits(m0);
            m1 = reverse_bits(m1);

            uint64_t v = (static_cast<uint64_t>(m0) << 32) | m1;

            tq[b_query - j - 1] = v;

            vec_00_to_15 = _mm256_slli_epi16(vec_00_to_15, 1);
            vec_16_to_31 = _mm256_slli_epi16(vec_16_to_31, 1);
            vec_32_to_47 = _mm256_slli_epi16(vec_32_to_47, 1);
            vec_48_to_63 = _mm256_slli_epi16(vec_48_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
#else
    std::cerr << "AVX512 or AVX2 is required for new transpose bin\n";
    exit(1);
#endif
}

void new_transpose_bin_512_avx512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
#if defined(__AVX512BW__)
    // Keep full 512-dim blocks as 8 chunks, but store the tail as compact
    // [b_query x num_chunks] so runtime can use maskz loads without query padding.
    for (size_t i = 0; i < padded_dim;) {
        size_t block_size = 512;
        if (i + 512 > padded_dim) {
            block_size = padded_dim - i;
        }
        size_t num_chunks = block_size / 64;

        for (size_t k = 0; k < num_chunks; ++k) {
            const uint8_t* current_q = q + i + k * 64;
            __m512i vec = _mm512_loadu_si512(current_q);

            for (size_t j = 0; j < b_query; ++j) {
                int bit_idx = b_query - 1 - j;
                __mmask64 m = _mm512_test_epi8_mask(vec, _mm512_set1_epi8(1 << bit_idx));
                tq[(b_query - j - 1) * num_chunks + k] = reverse_bits_u64(static_cast<uint64_t>(m));
            }
        }

        i += block_size;
        tq += num_chunks * b_query;
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i < padded_dim;) {
        size_t block_size = 512;
        if (i + 512 > padded_dim) {
            block_size = padded_dim - i;
        }
        // Each chunk represents 64 bytes (512 bits) of dimensions
        size_t num_chunks = block_size / 64;

        for (size_t k = 0; k < num_chunks; ++k) {
            // Load 64 bytes using two sequential 32-byte AVX2 registers
            const uint8_t* current_q_lo = q + i + k * 64;
            const uint8_t* current_q_hi = q + i + k * 64 + 32;

            __m256i vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_q_lo));
            __m256i vec_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_q_hi));

            for (size_t j = 0; j < b_query; ++j) {
                int bit_idx = b_query - 1 - j;
                __m256i mask_vec = _mm256_set1_epi8(static_cast<char>(1 << bit_idx));

                // Process lower 32 bytes
                __m256i res_lo = _mm256_and_si256(vec_lo, mask_vec);
                __m256i eq_lo = _mm256_cmpeq_epi8(res_lo, _mm256_setzero_si256());
                uint32_t m_lo = ~static_cast<uint32_t>(_mm256_movemask_epi8(eq_lo));

                // Process upper 32 bytes
                __m256i res_hi = _mm256_and_si256(vec_hi, mask_vec);
                __m256i eq_hi = _mm256_cmpeq_epi8(res_hi, _mm256_setzero_si256());
                uint32_t m_hi = ~static_cast<uint32_t>(_mm256_movemask_epi8(eq_hi));

                // Combine both 32-bit masks into a single 64-bit mask
                uint64_t m = (static_cast<uint64_t>(m_hi) << 32) | m_lo;

                // Write into the 64-bit structured macro-layout
                tq[(b_query - j - 1) * num_chunks + k] = reverse_bits_u64(m);
            }
        }

        i += block_size;
        tq += num_chunks * b_query;
    }
#else
    std::cerr << "AVX512BW or AVX2 is required for new_transpose_bin_512\n";
    exit(1);
#endif
}

float mask_ip_x0_q_avx512(const float* query, const uint64_t* data, size_t padded_dim) {
    const size_t num_blk = padded_dim / 64;
    const uint64_t* it_data = data;
    const float* it_query = query;
// Easier
#if defined(__AVX512F__)

    //    __m512 sum0 = _mm512_setzero_ps();
    //    __m512 sum1 = _mm512_setzero_ps();
    //    __m512 sum2 = _mm512_setzero_ps();
    //    __m512 sum3 = _mm512_setzero_ps();

    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);

        auto mask0 = static_cast<__mmask16>(bits);
        auto mask1 = static_cast<__mmask16>(bits >> 16);
        auto mask2 = static_cast<__mmask16>(bits >> 32);
        auto mask3 = static_cast<__mmask16>(bits >> 48);

        __m512 masked0 = _mm512_maskz_loadu_ps(mask0, it_query);
        __m512 masked1 = _mm512_maskz_loadu_ps(mask1, it_query + 16);
        __m512 masked2 = _mm512_maskz_loadu_ps(mask2, it_query + 32);
        __m512 masked3 = _mm512_maskz_loadu_ps(mask3, it_query + 48);

        sum = _mm512_add_ps(sum, masked0);
        sum = _mm512_add_ps(sum, masked1);
        sum = _mm512_add_ps(sum, masked2);
        sum = _mm512_add_ps(sum, masked3);

        //         _mm_prefetch(reinterpret_cast<const char*>(it_query + 128), _MM_HINT_T1);

        ++it_data;
        it_query += 64;
    }

    //    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    return _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)

    __m256 sum = _mm256_setzero_ps();

    __m256i bit_checker = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);

    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);

        // 64 bits / 8 floats = 8 iterations
        for (int j = 0; j < 8; ++j) {
            uint8_t current_byte = static_cast<uint8_t>(bits >> (j * 8));
            __m256i v_byte = _mm256_set1_epi32(current_byte);
            __m256i masked_bits = _mm256_and_si256(v_byte, bit_checker);
            __m256i mask = _mm256_cmpgt_epi32(masked_bits, _mm256_setzero_si256());

            __m256 q_vals = _mm256_loadu_ps(it_query);
            __m256 masked = _mm256_and_ps(q_vals, _mm256_castsi256_ps(mask));

            sum = _mm256_add_ps(sum, masked);

            it_query += 8;
        }
        ++it_data;
    }

    float result = 0.0f;
    for (int i = 0; i < 8; ++i) {
        result += reinterpret_cast<float*>(&sum)[i];
    }
    return result;
#else
    std::cerr << "AVX512 or AVX2 is required for mask ip x0 q\n";
    exit(1);
#endif
    return 0.0F;
}

}  // namespace rabitqlib::simd
