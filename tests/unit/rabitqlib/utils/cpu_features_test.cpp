#include <gtest/gtest.h>

#include <cstdlib>

#include "rabitqlib/utils/cpu_features.hpp"

TEST(CpuFeatures, returns_stable_result) {
    const auto& first = rabitqlib::cpu::features();
    const auto& second = rabitqlib::cpu::features();

    ASSERT_EQ(&first, &second);
    ASSERT_EQ(rabitqlib::cpu::has_avx512_popcnt(), rabitqlib::cpu::has_avx512_core() && first.avx512vpopcntdq);
}

TEST(CpuFeatures, parses_env_flags) {
    constexpr const char* kName = "RABITQ_TEST_ENV_FLAG";

    unsetenv(kName);
    EXPECT_FALSE(rabitqlib::cpu::env_flag_enabled(kName));

    setenv(kName, "0", 1);
    EXPECT_FALSE(rabitqlib::cpu::env_flag_enabled(kName));

    setenv(kName, "false", 1);
    EXPECT_FALSE(rabitqlib::cpu::env_flag_enabled(kName));

    setenv(kName, "no", 1);
    EXPECT_FALSE(rabitqlib::cpu::env_flag_enabled(kName));

    setenv(kName, "off", 1);
    EXPECT_FALSE(rabitqlib::cpu::env_flag_enabled(kName));

    setenv(kName, "1", 1);
    EXPECT_TRUE(rabitqlib::cpu::env_flag_enabled(kName));

    setenv(kName, "true", 1);
    EXPECT_TRUE(rabitqlib::cpu::env_flag_enabled(kName));

    unsetenv(kName);
}
