#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace rabitqlib::quant::rabitq_impl::ex_bits {
inline void packing_1bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 16 == 0
    for (size_t j = 0; j < dim; j += 16) {
        uint16_t code = 0;
        for (size_t i = 0; i < 16; ++i) {
            code |= static_cast<uint16_t>(o_raw[i]) << i;
        }
        std::memcpy(o_compact, &code, sizeof(uint16_t));

        o_raw += 16;
        o_compact += 2;
    }
}

inline void packing_2bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 64 == 0
    for (size_t j = 0; j < dim; j += 64) {
        // Pack 64 2-bit codes into 16 bytes. Byte k stores dimensions
        // k, k + 16, k + 32, and k + 48 in bits [1:0], [3:2], [5:4], [7:6].
        for (size_t k = 0; k < 16; ++k) {
            o_compact[k] = static_cast<uint8_t>(
                (o_raw[k] & 0x03U) | ((o_raw[k + 16] & 0x03U) << 2) |
                ((o_raw[k + 32] & 0x03U) << 4) | ((o_raw[k + 48] & 0x03U) << 6)
            );
        }

        o_raw += 64;
        o_compact += 16;
    }
}

inline void packing_3bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 64 == 0
    for (size_t d = 0; d < dim; d += 64) {
        // split 3-bit codes into 2 bits and 1 bit
        // for 2-bit part, compact it like 2-bit code
        // for 1-bit part, compact 64 1-bit code into a int64
        for (size_t k = 0; k < 16; ++k) {
            o_compact[k] = static_cast<uint8_t>(
                (o_raw[k] & 0x03U) | ((o_raw[k + 16] & 0x03U) << 2) |
                ((o_raw[k + 32] & 0x03U) << 4) | ((o_raw[k + 48] & 0x03U) << 6)
            );
        }
        o_compact += 16;

        uint64_t top_bit = 0;
        for (size_t lane = 0; lane < 8; ++lane) {
            uint8_t packed = 0;
            for (size_t group = 0; group < 8; ++group) {
                packed |= static_cast<uint8_t>(((o_raw[group * 8 + lane] >> 2) & 0x01U) << group);
            }
            top_bit |= static_cast<uint64_t>(packed) << (lane * 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));

        o_raw += 64;
        o_compact += 8;
    }
}

inline void packing_4bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 16 == 0
    for (size_t j = 0; j < dim; j += 16) {
        // pack 16 4-bit codes into uint64
        // the lower 4 bits represent vec00 to vec07
        // the upper 4 bits represent vec08 to vec15
        for (size_t k = 0; k < 8; ++k) {
            o_compact[k] = static_cast<uint8_t>((o_raw[k] & 0x0FU) | ((o_raw[k + 8] & 0x0FU) << 4));
        }

        o_raw += 16;
        o_compact += 8;
    }
}

inline void packing_5bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 64 == 0
    for (size_t j = 0; j < dim; j += 64) {
        for (size_t k = 0; k < 16; ++k) {
            o_compact[k] = static_cast<uint8_t>((o_raw[k] & 0x0FU) | ((o_raw[k + 16] & 0x0FU) << 4));
            o_compact[k + 16] = static_cast<uint8_t>((o_raw[k + 32] & 0x0FU) | ((o_raw[k + 48] & 0x0FU) << 4));
        }

        o_compact += 32;

        uint64_t top_bit = 0;
        for (size_t lane = 0; lane < 8; ++lane) {
            uint8_t packed = 0;
            for (size_t group = 0; group < 8; ++group) {
                packed |= static_cast<uint8_t>(((o_raw[group * 8 + lane] >> 4) & 0x01U) << group);
            }
            top_bit |= static_cast<uint64_t>(packed) << (lane * 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));

        o_raw += 64;
        o_compact += 8;
    }
}

inline void packing_6bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // for vec00 to vec47, split code into 6
    // for vec48 to vec63, split code into 2 + 2 + 2
    for (size_t d = 0; d < dim; d += 64) {
        for (size_t k = 0; k < 16; ++k) {
            const uint8_t tail = o_raw[k + 48];
            o_compact[k] = static_cast<uint8_t>((o_raw[k] & 0x3FU) | ((tail & 0x03U) << 6));
            o_compact[k + 16] = static_cast<uint8_t>((o_raw[k + 16] & 0x3FU) | (((tail >> 2) & 0x03U) << 6));
            o_compact[k + 32] = static_cast<uint8_t>((o_raw[k + 32] & 0x3FU) | (((tail >> 4) & 0x03U) << 6));
        }

        o_compact += 48;
        o_raw += 64;
    }
}

inline void packing_7bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // for vec00 to vec47, split code into 6 + 1
    // for vec48 to vec63, split code into 2 + 2 + 2 + 1
    for (size_t d = 0; d < dim; d += 64) {
        for (size_t k = 0; k < 16; ++k) {
            const uint8_t tail = o_raw[k + 48];
            o_compact[k] = static_cast<uint8_t>((o_raw[k] & 0x3FU) | ((tail & 0x03U) << 6));
            o_compact[k + 16] = static_cast<uint8_t>((o_raw[k + 16] & 0x3FU) | (((tail >> 2) & 0x03U) << 6));
            o_compact[k + 32] = static_cast<uint8_t>((o_raw[k + 32] & 0x3FU) | (((tail >> 4) & 0x03U) << 6));
        }
        o_compact += 48;

        uint64_t top_bit = 0;
        for (size_t lane = 0; lane < 8; ++lane) {
            uint8_t packed = 0;
            for (size_t group = 0; group < 8; ++group) {
                packed |= static_cast<uint8_t>(((o_raw[group * 8 + lane] >> 6) & 0x01U) << group);
            }
            top_bit |= static_cast<uint64_t>(packed) << (lane * 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));

        o_compact += 8;
        o_raw += 64;
    }
}

inline void packing_8bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    std::memcpy(o_compact, o_raw, sizeof(uint8_t) * dim);
}

/**
 * @brief Packing ex_bits code to save space. For example, two 4-bit code will be
 * stored as 1 uint8. To compute inner product with the support of SIMD, the
 * packed codes need to be stored in different patterns. For details, please check the
 * code and comments for certain number of bits.
 *
 * @param o_raw unpacked code, code for each dim is represented by uint8
 * @param o_compact compact format of code
 * @param dim   dimension of code, NOTICE: different num of bits requried different
 *               dimension padding, dim should obey the corresponding requirement
 * @param ex_bits number of bits used for code
 */
inline void packing_rabitqplus_code(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim, size_t ex_bits
) {
    if (ex_bits == 1) {
        packing_1bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 2) {
        packing_2bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 3) {
        packing_3bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 4) {
        packing_4bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 5) {
        packing_5bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 6) {
        packing_6bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 7) {
        packing_7bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 8) {
        packing_8bit_excode(o_raw, o_compact, dim);
    } else {
        std::cerr << "Bad value for ex_bits in packing_rabitqplus_code()\b";
        exit(1);
    }
}
}  // namespace rabitqlib::quant::rabitq_impl::ex_bits
