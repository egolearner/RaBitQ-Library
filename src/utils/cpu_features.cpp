#include "rabitqlib/utils/cpu_features.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

namespace rabitqlib::cpu {
namespace {

Features detect_features() {
    Features detected{};
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    detected.avx2 = __builtin_cpu_supports("avx2");
    detected.fma = __builtin_cpu_supports("fma");
    detected.avx512f = __builtin_cpu_supports("avx512f");
    detected.avx512bw = __builtin_cpu_supports("avx512bw");
    detected.avx512dq = __builtin_cpu_supports("avx512dq");
    detected.avx512vpopcntdq = __builtin_cpu_supports("avx512vpopcntdq");
#endif
#endif

    if (env_flag_enabled("RABITQ_DISABLE_AVX512")) {
        detected.avx512f = false;
        detected.avx512bw = false;
        detected.avx512dq = false;
        detected.avx512vpopcntdq = false;
    }
    if (env_flag_enabled("RABITQ_DISABLE_AVX2")) {
        detected.avx2 = false;
        detected.fma = false;
    }
    return detected;
}

}  // namespace

const Features& features() {
    static const Features detected = detect_features();
    return detected;
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }

    std::string normalized;
    for (const char* it = value; *it != '\0'; ++it) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*it))));
    }

    return !normalized.empty() && normalized != "0" && normalized != "false" &&
           normalized != "no" && normalized != "off";
}

bool has_avx2() {
    const Features& detected = features();
    return detected.avx2 && detected.fma;
}

bool has_avx512_core() {
    const Features& detected = features();
    return detected.avx512f && detected.avx512bw && detected.avx512dq;
}

bool has_avx512_popcnt() {
    return has_avx512_core() && features().avx512vpopcntdq;
}

}  // namespace rabitqlib::cpu
