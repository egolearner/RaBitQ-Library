#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "simd/quantize_utils.hpp"

namespace rabitqlib::simd::reference {

inline void pack_top_bit(
    const uint8_t* raw, unsigned shift, uint8_t* compact
) {
    for (size_t group = 0; group < 8; ++group) {
        uint8_t packed = 0;
        for (size_t lane = 0; lane < 8; ++lane) {
            packed |= static_cast<uint8_t>(
                ((raw[group * 8 + lane] >> shift) & 1U) << lane
            );
        }
        compact[group] = packed;
    }
}

inline void packing_2bit_excode(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    for (size_t base = 0; base < dim; base += 64) {
        for (size_t i = 0; i < 16; ++i) {
            compact[i] = static_cast<uint8_t>(
                raw[i] | (raw[i + 16] << 2U) | (raw[i + 32] << 4U) |
                (raw[i + 48] << 6U)
            );
        }
        raw += 64;
        compact += 16;
    }
}

inline void packing_3bit_excode(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    for (size_t base = 0; base < dim; base += 64) {
        for (size_t i = 0; i < 16; ++i) {
            compact[i] = static_cast<uint8_t>(
                (raw[i] & 3U) | ((raw[i + 16] & 3U) << 2U) |
                ((raw[i + 32] & 3U) << 4U) | ((raw[i + 48] & 3U) << 6U)
            );
        }
        pack_top_bit(raw, 2, compact + 16);
        raw += 64;
        compact += 24;
    }
}

inline void packing_4bit_excode(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    for (size_t base = 0; base < dim; base += 16) {
        for (size_t i = 0; i < 8; ++i) {
            compact[i] = static_cast<uint8_t>(raw[i] | (raw[i + 8] << 4U));
        }
        raw += 16;
        compact += 8;
    }
}

inline void packing_5bit_excode(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    for (size_t base = 0; base < dim; base += 64) {
        for (size_t i = 0; i < 16; ++i) {
            compact[i] =
                static_cast<uint8_t>((raw[i] & 15U) | ((raw[i + 16] & 15U) << 4U));
            compact[i + 16] = static_cast<uint8_t>(
                (raw[i + 32] & 15U) | ((raw[i + 48] & 15U) << 4U)
            );
        }
        pack_top_bit(raw, 4, compact + 32);
        raw += 64;
        compact += 40;
    }
}

inline void packing_6bit_excode(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    for (size_t base = 0; base < dim; base += 64) {
        for (size_t i = 0; i < 16; ++i) {
            compact[i] =
                static_cast<uint8_t>((raw[i] & 63U) | ((raw[i + 48] & 3U) << 6U));
            compact[i + 16] = static_cast<uint8_t>(
                (raw[i + 16] & 63U) | (((raw[i + 48] >> 2U) & 3U) << 6U)
            );
            compact[i + 32] = static_cast<uint8_t>(
                (raw[i + 32] & 63U) | (((raw[i + 48] >> 4U) & 3U) << 6U)
            );
        }
        raw += 64;
        compact += 48;
    }
}

inline void packing_7bit_excode(
    const uint8_t* raw, uint8_t* compact, size_t dim
) {
    for (size_t base = 0; base < dim; base += 64) {
        for (size_t i = 0; i < 16; ++i) {
            compact[i] =
                static_cast<uint8_t>((raw[i] & 63U) | ((raw[i + 48] & 3U) << 6U));
            compact[i + 16] = static_cast<uint8_t>(
                (raw[i + 16] & 63U) | (((raw[i + 48] >> 2U) & 3U) << 6U)
            );
            compact[i + 32] = static_cast<uint8_t>(
                (raw[i + 32] & 63U) | (((raw[i + 48] >> 4U) & 3U) << 6U)
            );
        }
        pack_top_bit(raw, 6, compact + 48);
        raw += 64;
        compact += 56;
    }
}

inline void unpack_64(
    const uint8_t* compact, size_t bits, uint8_t* raw
) {
    if (bits == 2 || bits == 3) {
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t value = compact[i];
            raw[i] = value & 3U;
            raw[i + 16] = (value >> 2U) & 3U;
            raw[i + 32] = (value >> 4U) & 3U;
            raw[i + 48] = (value >> 6U) & 3U;
        }
        if (bits == 3) {
            for (size_t i = 0; i < 64; ++i) {
                raw[i] |= static_cast<uint8_t>(
                    ((compact[16 + i / 8] >> (i % 8)) & 1U) << 2U
                );
            }
        }
        return;
    }

    if (bits == 5) {
        for (size_t i = 0; i < 16; ++i) {
            raw[i] = compact[i] & 15U;
            raw[i + 16] = (compact[i] >> 4U) & 15U;
            raw[i + 32] = compact[i + 16] & 15U;
            raw[i + 48] = (compact[i + 16] >> 4U) & 15U;
        }
        for (size_t i = 0; i < 64; ++i) {
            raw[i] |= static_cast<uint8_t>(
                ((compact[32 + i / 8] >> (i % 8)) & 1U) << 4U
            );
        }
        return;
    }

    for (size_t i = 0; i < 16; ++i) {
        raw[i] = compact[i] & 63U;
        raw[i + 16] = compact[i + 16] & 63U;
        raw[i + 32] = compact[i + 32] & 63U;
        raw[i + 48] = static_cast<uint8_t>(
            (compact[i] >> 6U) | ((compact[i + 16] >> 6U) << 2U) |
            ((compact[i + 32] >> 6U) << 4U)
        );
    }
    if (bits == 7) {
        for (size_t i = 0; i < 64; ++i) {
            raw[i] |= static_cast<uint8_t>(
                ((compact[48 + i / 8] >> (i % 8)) & 1U) << 6U
            );
        }
    }
}

inline float excode_ip(
    const float* query, const uint8_t* compact, size_t dim, size_t bits
) {
    float result = 0.0F;
    if (bits == 1) {
        for (size_t i = 0; i < dim; ++i) {
            result += query[i] * static_cast<float>(
                (compact[i / 8] >> (i % 8)) & 1U
            );
        }
        return result;
    }
    if (bits == 4) {
        for (size_t i = 0; i < dim; i += 16) {
            for (size_t j = 0; j < 8; ++j) {
                result += query[i + j] * static_cast<float>(compact[j] & 15U);
                result += query[i + j + 8] * static_cast<float>(compact[j] >> 4U);
            }
            compact += 8;
        }
        return result;
    }
    if (bits == 8) {
        for (size_t i = 0; i < dim; ++i) {
            result += query[i] * static_cast<float>(compact[i]);
        }
        return result;
    }

    uint8_t raw[64];
    const size_t compact_stride = bits * 8;
    for (size_t i = 0; i < dim; i += 64) {
        unpack_64(compact, bits, raw);
        for (size_t j = 0; j < 64; ++j) {
            result += query[i + j] * static_cast<float>(raw[j]);
        }
        compact += compact_stride;
    }
    return result;
}

template <typename T>
inline void scalar_quantize(
    T* result, const float* input, size_t dim, float lo, float delta
) {
    const float inverse_delta = 1.0F / delta;
    for (size_t i = 0; i < dim; ++i) {
        result[i] = detail::quantize_nearest_even<T>((input[i] - lo) * inverse_delta);
    }
}

inline void new_transpose_bin(
    const uint16_t* query, uint64_t* transposed, size_t padded_dim, size_t bits
) {
    for (size_t base = 0; base < padded_dim; base += 64) {
        for (size_t bit = 0; bit < bits; ++bit) {
            uint64_t value = 0;
            for (size_t lane = 0; lane < 64; ++lane) {
                value |= static_cast<uint64_t>((query[base + lane] >> bit) & 1U)
                         << (63U - lane);
            }
            transposed[bit] = value;
        }
        transposed += bits;
    }
}

inline void new_transpose_bin_512(
    const uint8_t* query, uint64_t* transposed, size_t padded_dim, size_t bits
) {
    for (size_t base = 0; base < padded_dim;) {
        const size_t block_size = std::min<size_t>(512, padded_dim - base);
        const size_t chunks = block_size / 64;
        for (size_t bit = 0; bit < bits; ++bit) {
            for (size_t chunk = 0; chunk < chunks; ++chunk) {
                uint64_t value = 0;
                for (size_t lane = 0; lane < 64; ++lane) {
                    value |= static_cast<uint64_t>(
                                 (query[base + chunk * 64 + lane] >> bit) & 1U
                             )
                             << (63U - lane);
                }
                transposed[bit * chunks + chunk] = value;
            }
        }
        base += block_size;
        transposed += chunks * bits;
    }
}

inline float mask_ip_x0_q(
    const float* query, const uint64_t* data, size_t padded_dim
) {
    float result = 0.0F;
    for (size_t block = 0; block < padded_dim / 64; ++block) {
        const uint64_t bits = data[block];
        for (size_t lane = 0; lane < 64; ++lane) {
            if (((bits >> (63U - lane)) & 1U) != 0) {
                result += query[block * 64 + lane];
            }
        }
    }
    return result;
}

inline void flip_sign(const uint8_t* flip, float* data, size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
        if (((flip[i / 8] >> (i % 8)) & 1U) != 0) {
            data[i] = -data[i];
        }
    }
}

inline void kacs_walk(float* data, size_t len) {
    const size_t half = len / 2;
    for (size_t i = 0; i < half; ++i) {
        const float x = data[i];
        const float y = data[i + half];
        data[i] = x + y;
        data[i + half] = x - y;
    }
}

inline float warmup_ip_x0_q_512(
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
        for (size_t chunk = 0; chunk < chunks; ++chunk) {
            ppc += static_cast<size_t>(__builtin_popcountll(data_block[chunk]));
        }
        data += chunks;
        for (size_t bit = 0; bit < b_query; ++bit) {
            for (size_t chunk = 0; chunk < chunks; ++chunk) {
                ip += static_cast<size_t>(
                          __builtin_popcountll(
                              data_block[chunk] & query[bit * chunks + chunk]
                          )
                      )
                      << bit;
            }
        }
        query += chunks * b_query;
        base += block_size;
    }
    return delta * static_cast<float>(ip) + vl * static_cast<float>(ppc);
}

inline constexpr size_t kFastScanPermutation[16] = {
    0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
};

inline void fastscan_accumulate(
    const uint8_t* codes, const uint8_t* lut, uint16_t* result, size_t dim
) {
    std::fill(result, result + 32, 0);
    for (size_t codebook = 0; codebook < dim / 4; ++codebook) {
        for (size_t lane = 0; lane < 16; ++lane) {
            const uint8_t code = codes[lane];
            result[kFastScanPermutation[lane]] = static_cast<uint16_t>(
                result[kFastScanPermutation[lane]] + lut[code & 15U]
            );
            result[16 + kFastScanPermutation[lane]] = static_cast<uint16_t>(
                result[16 + kFastScanPermutation[lane]] + lut[code >> 4U]
            );
        }
        codes += 16;
        lut += 16;
    }
}

inline void transfer_lut_hacc(
    const uint16_t* lut, size_t dim, uint8_t* high_accuracy_lut
) {
    for (size_t codebook = 0; codebook < dim / 4; ++codebook) {
        for (size_t entry = 0; entry < 16; ++entry) {
            high_accuracy_lut[entry] = static_cast<uint8_t>(lut[entry]);
            high_accuracy_lut[16 + entry] = static_cast<uint8_t>(lut[entry] >> 8U);
        }
        lut += 16;
        high_accuracy_lut += 32;
    }
}

inline void fastscan_accumulate_hacc(
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
) {
    std::fill(result, result + 32, 0);
    for (size_t codebook = 0; codebook < dim / 4; ++codebook) {
        for (size_t lane = 0; lane < 16; ++lane) {
            const uint8_t code = codes[lane];
            const size_t low_index = code & 15U;
            const size_t high_index = code >> 4U;
            result[kFastScanPermutation[lane]] +=
                static_cast<int32_t>(lut[low_index]) |
                (static_cast<int32_t>(lut[16 + low_index]) << 8);
            result[16 + kFastScanPermutation[lane]] +=
                static_cast<int32_t>(lut[high_index]) |
                (static_cast<int32_t>(lut[16 + high_index]) << 8);
        }
        codes += 16;
        lut += 32;
    }
}

}  // namespace rabitqlib::simd::reference
