#include "simd/backend.hpp"

#include <arm_neon.h>

#include <algorithm>

namespace rabitqlib::simd {
namespace {

size_t popcount_words_neon(const uint64_t* words, size_t count) {
    size_t result = 0;
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        const uint8x16_t bytes =
            vld1q_u8(reinterpret_cast<const uint8_t*>(words + i));
        result += static_cast<size_t>(vaddlvq_u8(vcntq_u8(bytes)));
    }
    for (; i < count; ++i) {
        result += static_cast<size_t>(__builtin_popcountll(words[i]));
    }
    return result;
}

size_t popcount_and_words_neon(
    const uint64_t* left, const uint64_t* right, size_t count
) {
    size_t result = 0;
    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        const uint8x16_t left_bytes =
            vld1q_u8(reinterpret_cast<const uint8_t*>(left + i));
        const uint8x16_t right_bytes =
            vld1q_u8(reinterpret_cast<const uint8_t*>(right + i));
        result += static_cast<size_t>(
            vaddlvq_u8(vcntq_u8(vandq_u8(left_bytes, right_bytes)))
        );
    }
    for (; i < count; ++i) {
        result += static_cast<size_t>(__builtin_popcountll(left[i] & right[i]));
    }
    return result;
}

}  // namespace

float warmup_ip_x0_q_512_neon(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    size_t ip = 0;
    size_t ppc = 0;
    for (size_t base = 0; base < padded_dim;) {
        const size_t block_size = std::min<size_t>(512, padded_dim - base);
        const size_t chunks = block_size / 64;
        const uint64_t* data_block = data;
        ppc += popcount_words_neon(data_block, chunks);

        for (size_t bit = 0; bit < b_query; ++bit) {
            ip += popcount_and_words_neon(
                      data_block, query + bit * chunks, chunks
                  )
                  << bit;
        }

        data += chunks;
        query += chunks * b_query;
        base += block_size;
    }
    return delta * static_cast<float>(ip) + vl * static_cast<float>(ppc);
}

}  // namespace rabitqlib::simd
