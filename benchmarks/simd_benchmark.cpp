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

namespace {

volatile double g_sink = 0.0;

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
    std::vector<uint8_t> code8(kDimension);
    std::vector<uint8_t> code2_raw(kDimension);
    for (size_t i = 0; i < kDimension; ++i) {
        query[i] = static_cast<float>(static_cast<int64_t>(rng() % 2000) - 1000) / 101.0F;
        code8[i] = static_cast<uint8_t>(rng());
        code2_raw[i] = static_cast<uint8_t>(rng() & 3U);
    }
    std::vector<uint8_t> code2(kDimension * 2 / 8);
    rabitqlib::simd::packing_2bit_excode_scalar(
        code2_raw.data(), code2.data(), kDimension
    );

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
    std::vector<uint8_t> packed7(kDimension * 7 / 8);
    std::vector<uint8_t> raw7(kDimension);
    for (uint8_t& value : raw7) {
        value = static_cast<uint8_t>(rng() & 127U);
    }

    std::cout << "kernel                         scalar ns/op    NEON ns/op     speedup\n";
    bool passed = true;
    passed &= compare(
        "excode 8-bit inner product",
        [&](size_t) {
            g_sink += rabitqlib::simd::excode_ipimpl::ip16_fxu8_scalar(
                query.data(), code8.data(), kDimension
            );
        },
        [&](size_t) {
            g_sink += rabitqlib::simd::excode_ipimpl::ip16_fxu8_neon(
                query.data(), code8.data(), kDimension
            );
        },
        20000
    );
    passed &= compare(
        "excode 2-bit inner product",
        [&](size_t) {
            g_sink += rabitqlib::simd::excode_ipimpl::ip64_fxu2_scalar(
                query.data(), code2.data(), kDimension
            );
        },
        [&](size_t) {
            g_sink += rabitqlib::simd::excode_ipimpl::ip64_fxu2_neon(
                query.data(), code2.data(), kDimension
            );
        },
        15000
    );
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
    passed &= compare(
        "7-bit packing",
        [&](size_t iteration) {
            rabitqlib::simd::packing_7bit_excode_scalar(
                raw7.data(), packed7.data(), kDimension
            );
            g_sink += packed7[iteration % packed7.size()];
        },
        [&](size_t iteration) {
            rabitqlib::simd::packing_7bit_excode_neon(
                raw7.data(), packed7.data(), kDimension
            );
            g_sink += packed7[iteration % packed7.size()];
        },
        20000
    );

    if (!passed) {
        std::cerr << "A NEON kernel was more than 10% slower than its scalar oracle\n";
        return 1;
    }
    return g_sink == 0.123456789 ? 3 : 0;
}

#endif
