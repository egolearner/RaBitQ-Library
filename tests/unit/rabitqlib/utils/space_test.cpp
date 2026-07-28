#include <gtest/gtest.h>
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/defines.hpp"
#include "simd/backend.hpp"
#include "test_helpers.hpp"
#include "test_data.hpp"
#include <vector>
#include <cmath>

using namespace rabitqlib;
using namespace rabitq_test;

TEST(Select_IP_Func, returns_stable_function_pointer) {
    auto ip_func = select_excode_ipfunc(0);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(0));

    ip_func = select_excode_ipfunc(1);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(1));

    ip_func = select_excode_ipfunc(2);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(2));

    ip_func = select_excode_ipfunc(3);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(3));

    ip_func = select_excode_ipfunc(4);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(4));

    ip_func = select_excode_ipfunc(5);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(5));

    ip_func = select_excode_ipfunc(6);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(6));

    ip_func = select_excode_ipfunc(7);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, select_excode_ipfunc(7));

    ip_func = select_excode_ipfunc(8);
    ASSERT_NE(ip_func, nullptr);
#if defined(RABITQ_TARGET_AARCH64)
    if (simd::selected_backend() == simd::Backend::Scalar) {
        ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_scalar);
    } else {
        ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_neon);
    }
#elif defined(RABITQ_TARGET_X86_64)
    if (cpu::has_avx512_core()) {
        ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx512);
    } else {
        ASSERT_EQ(ip_func, simd::excode_ipimpl::ip16_fxu8_avx2);
    }
#endif
}

TEST(ScalarQuantize, Uint8MatchesRoundedScalar) {
    constexpr size_t dim = 37;
    constexpr float lo = -3.0F;
    constexpr float delta = 0.25F;
    std::vector<float> input(dim);
    std::vector<uint8_t> result(dim);
    std::vector<uint8_t> expected(dim);

    for (size_t i = 0; i < dim; ++i) {
        float quantized = static_cast<float>((i * 7) % 251) + (static_cast<int>(i % 3) - 1) * 0.2F;
        input[i] = lo + delta * quantized;
        expected[i] = static_cast<uint8_t>(std::nearbyint((input[i] - lo) / delta));
    }

    scalar_quantize<uint8_t>(result.data(), input.data(), dim, lo, delta);

    ASSERT_EQ(result, expected);
}

TEST(ScalarQuantize, Uint16MatchesRoundedScalar) {
    constexpr size_t dim = 41;
    constexpr float lo = 2.0F;
    constexpr float delta = 0.125F;
    std::vector<float> input(dim);
    std::vector<uint16_t> result(dim);
    std::vector<uint16_t> expected(dim);

    for (size_t i = 0; i < dim; ++i) {
        float quantized = static_cast<float>(1000 + i * 317) + (static_cast<int>(i % 5) - 2) * 0.1F;
        input[i] = lo + delta * quantized;
        expected[i] = static_cast<uint16_t>(std::nearbyint((input[i] - lo) / delta));
    }

    scalar_quantize<uint16_t>(result.data(), input.data(), dim, lo, delta);

    ASSERT_EQ(result, expected);
}

TEST(ip16_fxu1_avx, ip_works) {
    srand(42);
    constexpr size_t dim = 64;
    float query[dim];
    uint8_t codes[dim/8];
    
    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }

    for (size_t i = 0; i < dim / 8; ++i) {
        codes[i] = static_cast<uint8_t>(rand() % 256);
    }

    const float expected =
        simd::excode_ipimpl::ip16_fxu1_scalar(query, codes, dim);
    ASSERT_NEAR(
        rabitqlib::excode_ipimpl::ip16_fxu1_avx(query, codes, dim),
        expected,
        std::max(0.1F, std::abs(expected) * 2e-6F)
    );
}

TEST(ip64_fxu2_avx, ip_works) {
    srand(42);
    constexpr size_t dim = 64*4;
    float query[dim];
    uint8_t codes[dim/4];
    
    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }

    for (size_t i = 0; i < dim / 4; ++i) {
        codes[i] = static_cast<uint8_t>(rand() % 256);
    }
    const float expected =
        simd::excode_ipimpl::ip64_fxu2_scalar(query, codes, dim);
    ASSERT_NEAR(
        rabitqlib::excode_ipimpl::ip64_fxu2_avx(query, codes, dim),
        expected,
        std::max(0.1F, std::abs(expected) * 2e-6F)
    );
}

TEST(ip_fxu8_avx, ip_works) {
    constexpr size_t dim = 1024;
    std::vector<float> query(dim);
    std::vector<uint8_t> codes(dim);
    double expected = 0.0;

    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(i % 97) / 17.0F;
        codes[i] = static_cast<uint8_t>(i % 251);
        expected += static_cast<double>(query[i]) * static_cast<double>(codes[i]);
    }

    const float expected_float = static_cast<float>(expected);
    const float tolerance = std::max(0.1F, std::abs(expected_float) * 2e-6F);
    ex_ipfunc ip_func = select_excode_ipfunc(8);
    ASSERT_NEAR(ip_func(query.data(), codes.data(), dim), expected_float, tolerance);
#if defined(RABITQ_TARGET_X86_64)
    if (cpu::has_avx2()) {
        ASSERT_NEAR(
            simd::excode_ipimpl::ip16_fxu8_avx2(query.data(), codes.data(), dim),
            expected_float,
            tolerance
        );
    }
    if (cpu::has_avx512_core()) {
        ASSERT_NEAR(
            simd::excode_ipimpl::ip16_fxu8_avx512(query.data(), codes.data(), dim),
            expected_float,
            tolerance
        );
    }
#elif defined(RABITQ_TARGET_AARCH64)
    ASSERT_NEAR(
        simd::excode_ipimpl::ip16_fxu8_neon(query.data(), codes.data(), dim),
        expected_float,
        tolerance
    );
#endif
}
