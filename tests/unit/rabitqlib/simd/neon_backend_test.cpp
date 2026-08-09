#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/simd/dispatch.hpp"
#include "rabitqlib/simd/rotator_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "simd/backend.hpp"

#if defined(RABITQ_TARGET_AARCH64)

#include "rabitqlib/utils/fht_neon.hpp"

namespace {

using rabitqlib::simd::excode_ipimpl::ip16_fxu1_neon;
using rabitqlib::simd::excode_ipimpl::ip16_fxu1_scalar;
using rabitqlib::simd::excode_ipimpl::ip16_fxu4_neon;
using rabitqlib::simd::excode_ipimpl::ip16_fxu4_scalar;
using rabitqlib::simd::excode_ipimpl::ip16_fxu8_neon;
using rabitqlib::simd::excode_ipimpl::ip16_fxu8_scalar;
using rabitqlib::simd::excode_ipimpl::ip64_fxu2_neon;
using rabitqlib::simd::excode_ipimpl::ip64_fxu2_scalar;
using rabitqlib::simd::excode_ipimpl::ip64_fxu3_neon;
using rabitqlib::simd::excode_ipimpl::ip64_fxu3_scalar;
using rabitqlib::simd::excode_ipimpl::ip64_fxu5_neon;
using rabitqlib::simd::excode_ipimpl::ip64_fxu5_scalar;
using rabitqlib::simd::excode_ipimpl::ip64_fxu6_neon;
using rabitqlib::simd::excode_ipimpl::ip64_fxu6_scalar;
using rabitqlib::simd::excode_ipimpl::ip64_fxu7_neon;
using rabitqlib::simd::excode_ipimpl::ip64_fxu7_scalar;

using IpFunction = float (*)(const float*, const uint8_t*, size_t);
using PackFunction = void (*)(const uint8_t*, uint8_t*, size_t);

class RoundingModeGuard {
   public:
    RoundingModeGuard() : original_(std::fegetround()) {}
    ~RoundingModeGuard() { std::fesetround(original_); }

    RoundingModeGuard(const RoundingModeGuard&) = delete;
    RoundingModeGuard& operator=(const RoundingModeGuard&) = delete;

   private:
    int original_;
};

IpFunction scalar_ip(size_t bits) {
    static const std::array<IpFunction, 9> functions = {
        ip16_fxu1_scalar,
        ip16_fxu1_scalar,
        ip64_fxu2_scalar,
        ip64_fxu3_scalar,
        ip16_fxu4_scalar,
        ip64_fxu5_scalar,
        ip64_fxu6_scalar,
        ip64_fxu7_scalar,
        ip16_fxu8_scalar,
    };
    return functions.at(bits);
}

IpFunction neon_ip(size_t bits) {
    static const std::array<IpFunction, 9> functions = {
        ip16_fxu1_neon,
        ip16_fxu1_neon,
        ip64_fxu2_neon,
        ip64_fxu3_neon,
        ip16_fxu4_neon,
        ip64_fxu5_neon,
        ip64_fxu6_neon,
        ip64_fxu7_neon,
        ip16_fxu8_neon,
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

void pack_one_bit(
    const std::vector<uint8_t>& raw, std::vector<uint8_t>& compact
) {
    std::fill(compact.begin(), compact.end(), 0);
    for (size_t i = 0; i < raw.size(); ++i) {
        compact[i / 8] |= static_cast<uint8_t>((raw[i] & 1U) << (i % 8));
    }
}

void scalar_fht(std::vector<float>& values) {
    for (size_t half_span = 1; half_span < values.size(); half_span *= 2) {
        const size_t span = half_span * 2;
        for (size_t base = 0; base < values.size(); base += span) {
            for (size_t offset = 0; offset < half_span; ++offset) {
                const float left = values[base + offset];
                const float right = values[base + half_span + offset];
                values[base + offset] = left + right;
                values[base + half_span + offset] = left - right;
            }
        }
    }
}

}  // namespace

TEST(NeonBackend, CpuAndBuildCapabilitiesAreVisible) {
    EXPECT_TRUE(rabitqlib::cpu::has_neon());
    EXPECT_TRUE(rabitqlib::simd::backend_is_compiled(rabitqlib::simd::Backend::Neon));
    EXPECT_TRUE(rabitqlib::simd::backend_is_compiled(rabitqlib::simd::Backend::Scalar));
    EXPECT_FALSE(rabitqlib::simd::backend_is_compiled(rabitqlib::simd::Backend::Avx2));
    EXPECT_NE(rabitqlib::simd::selected_backend(), rabitqlib::simd::Backend::Unavailable);
}

TEST(NeonBackend, PackingMatchesScalarFormatForEveryBitWidth) {
    std::mt19937 rng(42);
    for (size_t bits = 2; bits <= 7; ++bits) {
        const size_t dim = bits == 4 ? 80 : 192;
        std::vector<uint8_t> raw(dim);
        for (uint8_t& value : raw) {
            value = static_cast<uint8_t>(rng() & ((1U << bits) - 1U));
        }
        std::vector<uint8_t> expected(dim * bits / 8);
        std::vector<uint8_t> actual(expected.size());
        scalar_pack(bits)(raw.data(), expected.data(), dim);
        neon_pack(bits)(raw.data(), actual.data(), dim);
        EXPECT_EQ(actual, expected) << "bits=" << bits;
    }
}

TEST(NeonBackend, ExcodeInnerProductsMatchScalarOracle) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> query_distribution(-10.0F, 10.0F);

    for (size_t bits = 1; bits <= 8; ++bits) {
        const size_t dim =
            bits == 8 ? 181 : (bits == 1 || bits == 4 ? 80 : 192);
        std::vector<float> query(dim);
        std::vector<uint8_t> raw(dim);
        for (size_t i = 0; i < dim; ++i) {
            query[i] = query_distribution(rng);
            raw[i] = static_cast<uint8_t>(rng() & ((1U << bits) - 1U));
        }

        std::vector<uint8_t> compact((dim * bits + 7) / 8);
        if (bits == 1) {
            pack_one_bit(raw, compact);
        } else if (bits == 8) {
            compact = raw;
        } else {
            scalar_pack(bits)(raw.data(), compact.data(), dim);
        }

        const float expected = scalar_ip(bits)(query.data(), compact.data(), dim);
        const float actual = neon_ip(bits)(query.data(), compact.data(), dim);
        EXPECT_NEAR(actual, expected, std::max(1e-4F, std::abs(expected) * 3e-6F))
            << "bits=" << bits;
    }
}

TEST(NeonBackend, QuantizationHandlesVectorTails) {
    constexpr float lo = -8.0F;
    constexpr float delta = 0.125F;
    for (size_t dim : {size_t{1}, size_t{3}, size_t{15}, size_t{16}, size_t{17}, size_t{37}}) {
        std::vector<float> input(dim);
        for (size_t i = 0; i < dim; ++i) {
            input[i] = lo + delta * (static_cast<float>((i * 29) % 200) + 0.2F);
        }
        std::vector<uint8_t> expected8(dim);
        std::vector<uint8_t> actual8(dim);
        std::vector<uint16_t> expected16(dim);
        std::vector<uint16_t> actual16(dim);
        rabitqlib::simd::scalar_quantize_uint8_scalar(
            expected8.data(), input.data(), dim, lo, delta
        );
        rabitqlib::simd::scalar_quantize_uint8_neon(
            actual8.data(), input.data(), dim, lo, delta
        );
        rabitqlib::simd::scalar_quantize_uint16_scalar(
            expected16.data(), input.data(), dim, lo, delta
        );
        rabitqlib::simd::scalar_quantize_uint16_neon(
            actual16.data(), input.data(), dim, lo, delta
        );
        EXPECT_EQ(actual8, expected8) << "dim=" << dim;
        EXPECT_EQ(actual16, expected16) << "dim=" << dim;
    }
}

TEST(NeonBackend, QuantizationUsesRoundToNearestEvenAtMidpoints) {
    RoundingModeGuard rounding_mode_guard;
    constexpr size_t dim = 17;
    std::array<float, dim> input{};
    std::array<uint8_t, dim> expected8{};
    std::array<uint16_t, dim> expected16{};
    for (size_t i = 0; i < dim; ++i) {
        input[i] = static_cast<float>(i) + 0.5F;
        expected8[i] = static_cast<uint8_t>(i + (i & 1U));
        expected16[i] = static_cast<uint16_t>(i + (i & 1U));
    }

    for (int rounding_mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        ASSERT_EQ(std::fesetround(rounding_mode), 0);
        std::array<uint8_t, dim> scalar8{};
        std::array<uint8_t, dim> neon8{};
        std::array<uint16_t, dim> scalar16{};
        std::array<uint16_t, dim> neon16{};

        rabitqlib::simd::scalar_quantize_uint8_scalar(
            scalar8.data(), input.data(), dim, 0.0F, 1.0F
        );
        rabitqlib::simd::scalar_quantize_uint8_neon(
            neon8.data(), input.data(), dim, 0.0F, 1.0F
        );
        rabitqlib::simd::scalar_quantize_uint16_scalar(
            scalar16.data(), input.data(), dim, 0.0F, 1.0F
        );
        rabitqlib::simd::scalar_quantize_uint16_neon(
            neon16.data(), input.data(), dim, 0.0F, 1.0F
        );

        EXPECT_EQ(scalar8, expected8) << "rounding mode=" << rounding_mode;
        EXPECT_EQ(neon8, expected8) << "rounding mode=" << rounding_mode;
        EXPECT_EQ(scalar16, expected16) << "rounding mode=" << rounding_mode;
        EXPECT_EQ(neon16, expected16) << "rounding mode=" << rounding_mode;
    }
}

TEST(NeonBackend, TransposeLayoutsMatchScalarAtBlockBoundaries) {
    std::mt19937 rng(42);
    for (size_t dim : {size_t{64}, size_t{128}, size_t{448}, size_t{512}, size_t{960}}) {
        for (size_t bits : {size_t{1}, size_t{5}, size_t{8}}) {
            std::vector<uint8_t> query8(dim);
            std::vector<uint16_t> query16(dim);
            for (size_t i = 0; i < dim; ++i) {
                query8[i] = static_cast<uint8_t>(rng() & ((1U << bits) - 1U));
                query16[i] = query8[i];
            }

            std::vector<uint64_t> expected(dim / 64 * bits);
            std::vector<uint64_t> actual(expected.size());
            rabitqlib::simd::new_transpose_bin_scalar(
                query16.data(), expected.data(), dim, bits
            );
            rabitqlib::simd::new_transpose_bin_neon(
                query16.data(), actual.data(), dim, bits
            );
            EXPECT_EQ(actual, expected) << "regular dim=" << dim << " bits=" << bits;

            rabitqlib::simd::new_transpose_bin_512_scalar(
                query8.data(), expected.data(), dim, bits
            );
            rabitqlib::simd::new_transpose_bin_512_neon(
                query8.data(), actual.data(), dim, bits
            );
            EXPECT_EQ(actual, expected) << "512 dim=" << dim << " bits=" << bits;
        }
    }
}

TEST(NeonBackend, MaskAndWarmupMatchScalar) {
    std::mt19937_64 rng(42);
    for (size_t dim : {size_t{64}, size_t{448}, size_t{512}, size_t{960}}) {
        std::vector<float> float_query(dim);
        std::vector<uint64_t> data(dim / 64);
        for (float& value : float_query) {
            value = static_cast<float>(static_cast<int64_t>(rng() % 2000) - 1000) / 37.0F;
        }
        for (uint64_t& value : data) {
            value = rng();
        }
        const float scalar_mask =
            rabitqlib::simd::mask_ip_x0_q_scalar(float_query.data(), data.data(), dim);
        const float neon_mask =
            rabitqlib::simd::mask_ip_x0_q_neon(float_query.data(), data.data(), dim);
        EXPECT_NEAR(neon_mask, scalar_mask, std::max(1e-4F, std::abs(scalar_mask) * 3e-6F));

        constexpr size_t bits = 8;
        std::vector<uint8_t> integer_query(dim);
        for (uint8_t& value : integer_query) {
            value = static_cast<uint8_t>(rng());
        }
        std::vector<uint64_t> transposed(dim / 64 * bits);
        rabitqlib::simd::new_transpose_bin_512_scalar(
            integer_query.data(), transposed.data(), dim, bits
        );
        const float scalar_warmup = rabitqlib::simd::warmup_ip_x0_q_512_scalar(
            data.data(), transposed.data(), 0.25F, -0.75F, dim, bits
        );
        const float neon_warmup = rabitqlib::simd::warmup_ip_x0_q_512_neon(
            data.data(), transposed.data(), 0.25F, -0.75F, dim, bits
        );
        EXPECT_FLOAT_EQ(neon_warmup, scalar_warmup) << "dim=" << dim;
    }
}

TEST(NeonBackend, FhtMatchesScalarForEverySupportedSize) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-10.0F, 10.0F);
    for (size_t log_size = 6; log_size <= 11; ++log_size) {
        const size_t size = size_t{1} << log_size;
        std::vector<float> expected(size);
        for (float& value : expected) {
            value = distribution(rng);
        }
        std::vector<float> actual = expected;
        scalar_fht(expected);
        switch (log_size) {
            case 6:
                rabitqlib::helper_float_6(actual.data());
                break;
            case 7:
                rabitqlib::helper_float_7(actual.data());
                break;
            case 8:
                rabitqlib::helper_float_8(actual.data());
                break;
            case 9:
                rabitqlib::helper_float_9(actual.data());
                break;
            case 10:
                rabitqlib::helper_float_10(actual.data());
                break;
            case 11:
                rabitqlib::helper_float_11(actual.data());
                break;
            default:
                FAIL() << "unsupported FHT size";
        }
        for (size_t i = 0; i < size; ++i) {
            EXPECT_NEAR(
                actual[i],
                expected[i],
                std::max(1e-5F, std::abs(expected[i]) * 1e-6F)
            ) << "size=" << size << " lane=" << i;
        }
    }
}

TEST(NeonBackend, RotatorPrimitivesHandleTails) {
    std::mt19937 rng(42);
    for (size_t dim : {size_t{1}, size_t{7}, size_t{8}, size_t{17}, size_t{128}}) {
        std::vector<uint8_t> flip((dim + 7) / 8);
        std::vector<float> expected(dim);
        for (uint8_t& value : flip) {
            value = static_cast<uint8_t>(rng());
        }
        for (size_t i = 0; i < dim; ++i) {
            expected[i] = static_cast<float>(i + 1);
        }
        std::vector<float> actual = expected;
        rabitqlib::simd::flip_sign_scalar(flip.data(), expected.data(), dim);
        rabitqlib::simd::flip_sign_neon(flip.data(), actual.data(), dim);
        EXPECT_EQ(actual, expected) << "dim=" << dim;
    }

    for (size_t len : {size_t{2}, size_t{10}, size_t{64}}) {
        std::vector<float> expected(len);
        for (size_t i = 0; i < len; ++i) {
            expected[i] = static_cast<float>(i) / 3.0F;
        }
        std::vector<float> actual = expected;
        rabitqlib::simd::kacs_walk_scalar(expected.data(), len);
        rabitqlib::simd::kacs_walk_neon(actual.data(), len);
        EXPECT_EQ(actual, expected) << "len=" << len;
    }
}

TEST(NeonBackend, FastScanMatchesScalarForNormalAndHighAccuracy) {
    std::mt19937 rng(42);
    for (size_t dim : {size_t{16}, size_t{64}, size_t{96}}) {
        constexpr size_t num_vectors = 32;
        std::vector<uint8_t> binary_codes(num_vectors * dim / 8);
        for (uint8_t& value : binary_codes) {
            value = static_cast<uint8_t>(rng());
        }
        std::vector<uint8_t> packed_codes(dim * 4);
        rabitqlib::fastscan::pack_codes(
            dim, binary_codes.data(), num_vectors, packed_codes.data()
        );

        std::vector<uint8_t> lut8(dim * 4);
        std::vector<uint16_t> lut16(dim * 4);
        for (size_t i = 0; i < lut8.size(); ++i) {
            lut8[i] = static_cast<uint8_t>(rng());
            lut16[i] = static_cast<uint16_t>(rng());
        }

        std::array<uint16_t, 32> expected{};
        std::array<uint16_t, 32> actual{};
        rabitqlib::fastscan::simd::accumulate_scalar(
            packed_codes.data(), lut8.data(), expected.data(), dim
        );
        rabitqlib::fastscan::simd::accumulate_neon(
            packed_codes.data(), lut8.data(), actual.data(), dim
        );
        EXPECT_EQ(actual, expected) << "normal dim=" << dim;

        std::vector<uint8_t> scalar_hacc_lut(dim * 8);
        std::vector<uint8_t> neon_hacc_lut(dim * 8);
        rabitqlib::fastscan::simd::transfer_lut_hacc_scalar(
            lut16.data(), dim, scalar_hacc_lut.data()
        );
        rabitqlib::fastscan::simd::transfer_lut_hacc_neon(
            lut16.data(), dim, neon_hacc_lut.data()
        );
        EXPECT_EQ(neon_hacc_lut, scalar_hacc_lut);

        std::array<int32_t, 32> expected_hacc{};
        std::array<int32_t, 32> actual_hacc{};
        rabitqlib::fastscan::simd::accumulate_hacc_scalar(
            packed_codes.data(), scalar_hacc_lut.data(), expected_hacc.data(), dim
        );
        rabitqlib::fastscan::simd::accumulate_hacc_neon(
            packed_codes.data(), neon_hacc_lut.data(), actual_hacc.data(), dim
        );
        EXPECT_EQ(actual_hacc, expected_hacc) << "high accuracy dim=" << dim;
    }
}

TEST(NeonBackend, EmptyInputsAndInvalidBitWidthAreHandled) {
    std::array<uint16_t, 32> normal{};
    normal.fill(1);
    rabitqlib::fastscan::simd::accumulate_neon(nullptr, nullptr, normal.data(), 0);
    EXPECT_TRUE(std::all_of(normal.begin(), normal.end(), [](uint16_t value) {
        return value == 0;
    }));

    std::array<int32_t, 32> high_accuracy{};
    high_accuracy.fill(1);
    rabitqlib::fastscan::simd::accumulate_hacc_neon(
        nullptr, nullptr, high_accuracy.data(), 0
    );
    EXPECT_TRUE(std::all_of(
        high_accuracy.begin(), high_accuracy.end(), [](int32_t value) {
            return value == 0;
        }
    ));
    EXPECT_THROW(rabitqlib::select_excode_ipfunc(9), std::invalid_argument);
}

TEST(Dispatch, UnavailableBackendDefersFailureUntilKernelUse) {
    if (rabitqlib::simd::selected_backend() != rabitqlib::simd::Backend::Unavailable) {
        GTEST_SKIP() << "requires an unavailable or invalid forced backend";
    }
    EXPECT_THROW(rabitqlib::simd::flip_sign(nullptr, nullptr, 0), std::runtime_error);
}

#endif
