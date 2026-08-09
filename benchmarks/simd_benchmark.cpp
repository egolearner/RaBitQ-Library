#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "simd/backend.hpp"

#if !defined(RABITQ_TARGET_AARCH64)

int main() {
    std::cerr << "The scalar-versus-NEON benchmark requires an AArch64 build\n";
    return 2;
}

#else

#include "rabitqlib/utils/fht_neon.hpp"

namespace {

volatile double g_sink = 0.0;

using IpFunction = float (*)(const float*, const uint8_t*, size_t);
using PackFunction = void (*)(const uint8_t*, uint8_t*, size_t);
using FhtFunction = void (*)(float*);

IpFunction scalar_ip(size_t bits) {
    static const std::array<IpFunction, 9> functions = {
        rabitqlib::simd::excode_ipimpl::ip16_fxu1_scalar,
        rabitqlib::simd::excode_ipimpl::ip16_fxu1_scalar,
        rabitqlib::simd::excode_ipimpl::ip64_fxu2_scalar,
        rabitqlib::simd::excode_ipimpl::ip64_fxu3_scalar,
        rabitqlib::simd::excode_ipimpl::ip16_fxu4_scalar,
        rabitqlib::simd::excode_ipimpl::ip64_fxu5_scalar,
        rabitqlib::simd::excode_ipimpl::ip64_fxu6_scalar,
        rabitqlib::simd::excode_ipimpl::ip64_fxu7_scalar,
        rabitqlib::simd::excode_ipimpl::ip16_fxu8_scalar,
    };
    return functions.at(bits);
}

IpFunction neon_ip(size_t bits) {
    static const std::array<IpFunction, 9> functions = {
        rabitqlib::simd::excode_ipimpl::ip16_fxu1_neon,
        rabitqlib::simd::excode_ipimpl::ip16_fxu1_neon,
        rabitqlib::simd::excode_ipimpl::ip64_fxu2_neon,
        rabitqlib::simd::excode_ipimpl::ip64_fxu3_neon,
        rabitqlib::simd::excode_ipimpl::ip16_fxu4_neon,
        rabitqlib::simd::excode_ipimpl::ip64_fxu5_neon,
        rabitqlib::simd::excode_ipimpl::ip64_fxu6_neon,
        rabitqlib::simd::excode_ipimpl::ip64_fxu7_neon,
        rabitqlib::simd::excode_ipimpl::ip16_fxu8_neon,
    };
    return functions.at(bits);
}

PackFunction scalar_pack(size_t bits) {
    static const std::array<PackFunction, 8> functions = {
        nullptr,
        nullptr,
        rabitqlib::simd::packing_2bit_excode_scalar,
        rabitqlib::simd::packing_3bit_excode_scalar,
        rabitqlib::simd::packing_4bit_excode_scalar,
        rabitqlib::simd::packing_5bit_excode_scalar,
        rabitqlib::simd::packing_6bit_excode_scalar,
        rabitqlib::simd::packing_7bit_excode_scalar,
    };
    return functions.at(bits);
}

PackFunction neon_pack(size_t bits) {
    static const std::array<PackFunction, 8> functions = {
        nullptr,
        nullptr,
        rabitqlib::simd::packing_2bit_excode_neon,
        rabitqlib::simd::packing_3bit_excode_neon,
        rabitqlib::simd::packing_4bit_excode_neon,
        rabitqlib::simd::packing_5bit_excode_neon,
        rabitqlib::simd::packing_6bit_excode_neon,
        rabitqlib::simd::packing_7bit_excode_neon,
    };
    return functions.at(bits);
}

FhtFunction neon_fht(size_t size) {
    switch (size) {
        case 64:
            return rabitqlib::helper_float_6;
        case 256:
            return rabitqlib::helper_float_8;
        case 1024:
            return rabitqlib::helper_float_10;
        default:
            return nullptr;
    }
}

void pack_one_bit(
    const std::vector<uint8_t>& raw, std::vector<uint8_t>& compact
) {
    std::fill(compact.begin(), compact.end(), 0);
    for (size_t i = 0; i < raw.size(); ++i) {
        compact[i / 8] |= static_cast<uint8_t>((raw[i] & 1U) << (i % 8));
    }
}

void scalar_fht(float* values, size_t size) {
    for (size_t half_span = 1; half_span < size; half_span *= 2) {
        const size_t span = half_span * 2;
        for (size_t base = 0; base < size; base += span) {
            for (size_t offset = 0; offset < half_span; ++offset) {
                const float left = values[base + offset];
                const float right = values[base + half_span + offset];
                values[base + offset] = left + right;
                values[base + half_span + offset] = left - right;
            }
        }
    }
}

template <typename Function>
double median_nanoseconds(Function&& function, size_t iterations) {
    constexpr size_t kRounds = 7;
    std::array<double, kRounds> samples{};
    for (size_t warmup = 0; warmup < iterations / 10 + 1; ++warmup) {
        function(warmup);
    }
    for (double& sample : samples) {
        const auto start = std::chrono::steady_clock::now();
        for (size_t iteration = 0; iteration < iterations; ++iteration) {
            function(iteration);
        }
        const auto stop = std::chrono::steady_clock::now();
        sample = static_cast<double>(
                     std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
                         .count()
                 ) /
                 static_cast<double>(iterations);
    }
    std::sort(samples.begin(), samples.end());
    return samples[kRounds / 2];
}

template <typename ScalarFunction, typename NeonFunction>
bool compare(
    const std::string& name,
    ScalarFunction&& scalar_function,
    NeonFunction&& neon_function,
    size_t iterations
) {
    const double scalar_ns =
        median_nanoseconds(std::forward<ScalarFunction>(scalar_function), iterations);
    const double neon_ns =
        median_nanoseconds(std::forward<NeonFunction>(neon_function), iterations);
    const double speedup = scalar_ns / neon_ns;
    std::cout << std::left << std::setw(28) << name << std::right << std::setw(14)
              << std::fixed << std::setprecision(2) << scalar_ns << std::setw(14)
              << neon_ns << std::setw(12) << speedup << "x\n";
    return neon_ns <= scalar_ns * 1.10;
}

}  // namespace

int main() {
    constexpr size_t kDimension = 960;
    constexpr size_t kBits = 8;
    std::mt19937_64 rng(42);

    std::vector<float> query(kDimension);
    for (size_t i = 0; i < kDimension; ++i) {
        query[i] = static_cast<float>(static_cast<int64_t>(rng() % 2000) - 1000) / 101.0F;
    }
    std::array<std::vector<uint8_t>, 9> raw_codes;
    std::array<std::vector<uint8_t>, 9> compact_codes;
    for (size_t bits = 1; bits <= 8; ++bits) {
        raw_codes[bits].resize(kDimension);
        compact_codes[bits].resize(kDimension * bits / 8);
        for (uint8_t& value : raw_codes[bits]) {
            value = static_cast<uint8_t>(rng() & ((1U << bits) - 1U));
        }
        if (bits == 1) {
            pack_one_bit(raw_codes[bits], compact_codes[bits]);
        } else if (bits == 8) {
            compact_codes[bits] = raw_codes[bits];
        } else {
            scalar_pack(bits)(
                raw_codes[bits].data(), compact_codes[bits].data(), kDimension
            );
        }
    }

    std::vector<uint64_t> binary_data(kDimension / 64);
    for (uint64_t& value : binary_data) {
        value = rng();
    }
    std::vector<uint8_t> integer_query(kDimension);
    for (uint8_t& value : integer_query) {
        value = static_cast<uint8_t>(rng());
    }
    std::vector<uint64_t> transposed(kDimension / 64 * kBits);
    rabitqlib::simd::new_transpose_bin_512_scalar(
        integer_query.data(), transposed.data(), kDimension, kBits
    );
    std::vector<uint16_t> integer_query16(
        integer_query.begin(), integer_query.end()
    );
    std::vector<uint64_t> transpose_result(kDimension / 64 * kBits);

    constexpr size_t kNumVectors = 32;
    std::vector<uint8_t> binary_codes(kNumVectors * kDimension / 8);
    for (uint8_t& value : binary_codes) {
        value = static_cast<uint8_t>(rng());
    }
    std::vector<uint8_t> packed_codes(kDimension * 4);
    rabitqlib::fastscan::pack_codes(
        kDimension, binary_codes.data(), kNumVectors, packed_codes.data()
    );
    std::vector<uint8_t> lut8(kDimension * 4);
    std::vector<uint16_t> lut16(kDimension * 4);
    for (size_t i = 0; i < lut8.size(); ++i) {
        lut8[i] = static_cast<uint8_t>(rng());
        lut16[i] = static_cast<uint16_t>(rng());
    }
    std::vector<uint8_t> hacc_lut(kDimension * 8);
    rabitqlib::fastscan::simd::transfer_lut_hacc_scalar(
        lut16.data(), kDimension, hacc_lut.data()
    );

    std::array<uint16_t, 32> fastscan_result{};
    std::array<int32_t, 32> fastscan_hacc_result{};
    std::cout << "kernel                         scalar ns/op    NEON ns/op     speedup\n";
    bool passed = true;
    for (size_t bits = 1; bits <= 8; ++bits) {
        const IpFunction scalar_function = scalar_ip(bits);
        const IpFunction neon_function = neon_ip(bits);
        const size_t iterations = bits == 8 ? 20000 : 12000;
        passed &= compare(
            "excode " + std::to_string(bits) + "-bit inner product",
            [&, bits, scalar_function](size_t) {
                g_sink += scalar_function(
                    query.data(), compact_codes[bits].data(), kDimension
                );
            },
            [&, bits, neon_function](size_t) {
                g_sink += neon_function(
                    query.data(), compact_codes[bits].data(), kDimension
                );
            },
            iterations
        );
    }
    passed &= compare(
        "mask inner product",
        [&](size_t) {
            g_sink += rabitqlib::simd::mask_ip_x0_q_scalar(
                query.data(), binary_data.data(), kDimension
            );
        },
        [&](size_t) {
            g_sink += rabitqlib::simd::mask_ip_x0_q_neon(
                query.data(), binary_data.data(), kDimension
            );
        },
        20000
    );
    passed &= compare(
        "warmup/popcount",
        [&](size_t) {
            g_sink += rabitqlib::simd::warmup_ip_x0_q_512_scalar(
                binary_data.data(), transposed.data(), 0.25F, -0.5F, kDimension, kBits
            );
        },
        [&](size_t) {
            g_sink += rabitqlib::simd::warmup_ip_x0_q_512_neon(
                binary_data.data(), transposed.data(), 0.25F, -0.5F, kDimension, kBits
            );
        },
        25000
    );
    passed &= compare(
        "transpose uint16",
        [&](size_t iteration) {
            rabitqlib::simd::new_transpose_bin_scalar(
                integer_query16.data(),
                transpose_result.data(),
                kDimension,
                kBits
            );
            g_sink += transpose_result[iteration % transpose_result.size()];
        },
        [&](size_t iteration) {
            rabitqlib::simd::new_transpose_bin_neon(
                integer_query16.data(),
                transpose_result.data(),
                kDimension,
                kBits
            );
            g_sink += transpose_result[iteration % transpose_result.size()];
        },
        3000
    );
    passed &= compare(
        "transpose uint8/512",
        [&](size_t iteration) {
            rabitqlib::simd::new_transpose_bin_512_scalar(
                integer_query.data(),
                transpose_result.data(),
                kDimension,
                kBits
            );
            g_sink += transpose_result[iteration % transpose_result.size()];
        },
        [&](size_t iteration) {
            rabitqlib::simd::new_transpose_bin_512_neon(
                integer_query.data(),
                transpose_result.data(),
                kDimension,
                kBits
            );
            g_sink += transpose_result[iteration % transpose_result.size()];
        },
        3000
    );
    passed &= compare(
        "FastScan accumulate",
        [&](size_t iteration) {
            rabitqlib::fastscan::simd::accumulate_scalar(
                packed_codes.data(), lut8.data(), fastscan_result.data(), kDimension
            );
            g_sink += fastscan_result[iteration % fastscan_result.size()];
        },
        [&](size_t iteration) {
            rabitqlib::fastscan::simd::accumulate_neon(
                packed_codes.data(), lut8.data(), fastscan_result.data(), kDimension
            );
            g_sink += fastscan_result[iteration % fastscan_result.size()];
        },
        12000
    );
    passed &= compare(
        "FastScan high accuracy",
        [&](size_t iteration) {
            rabitqlib::fastscan::simd::accumulate_hacc_scalar(
                packed_codes.data(),
                hacc_lut.data(),
                fastscan_hacc_result.data(),
                kDimension
            );
            g_sink += fastscan_hacc_result[iteration % fastscan_hacc_result.size()];
        },
        [&](size_t iteration) {
            rabitqlib::fastscan::simd::accumulate_hacc_neon(
                packed_codes.data(),
                hacc_lut.data(),
                fastscan_hacc_result.data(),
                kDimension
            );
            g_sink += fastscan_hacc_result[iteration % fastscan_hacc_result.size()];
        },
        8000
    );
    for (size_t bits : {size_t{3}, size_t{5}, size_t{7}}) {
        const PackFunction scalar_function = scalar_pack(bits);
        const PackFunction neon_function = neon_pack(bits);
        passed &= compare(
            std::to_string(bits) + "-bit packing",
            [&, bits, scalar_function](size_t iteration) {
                scalar_function(
                    raw_codes[bits].data(),
                    compact_codes[bits].data(),
                    kDimension
                );
                g_sink += compact_codes[bits][
                    iteration % compact_codes[bits].size()
                ];
            },
            [&, bits, neon_function](size_t iteration) {
                neon_function(
                    raw_codes[bits].data(),
                    compact_codes[bits].data(),
                    kDimension
                );
                g_sink += compact_codes[bits][
                    iteration % compact_codes[bits].size()
                ];
            },
            12000
        );
    }
    for (size_t size : {size_t{64}, size_t{256}, size_t{1024}}) {
        std::vector<float> fht_source(size);
        for (float& value : fht_source) {
            value = static_cast<float>(
                static_cast<int64_t>(rng() % 2000) - 1000
            ) / 1000.0F;
        }
        std::vector<float> scalar_work = fht_source;
        std::vector<float> neon_work = fht_source;
        const FhtFunction neon_function = neon_fht(size);
        passed &= compare(
            "FHT " + std::to_string(size),
            [&, size](size_t iteration) {
                if ((iteration & 15U) == 0) {
                    scalar_work = fht_source;
                }
                scalar_fht(scalar_work.data(), size);
                g_sink += scalar_work[iteration % size];
            },
            [&, size, neon_function](size_t iteration) {
                if ((iteration & 15U) == 0) {
                    neon_work = fht_source;
                }
                neon_function(neon_work.data());
                g_sink += neon_work[iteration % size];
            },
            size == 1024 ? 2000 : 5000
        );
    }

    if (!passed) {
        std::cerr << "A NEON kernel was more than 10% slower than its scalar oracle\n";
        return 1;
    }
    return g_sink == 0.123456789 ? 3 : 0;
}

#endif
