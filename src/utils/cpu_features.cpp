#include "rabitqlib/utils/cpu_features.hpp"

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(RABITQ_TARGET_X86_64) && !defined(_MSC_VER)
#include <cpuid.h>
#endif

#if defined(RABITQ_TARGET_AARCH64) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace rabitqlib::cpu {
namespace {

#if defined(_MSC_VER) && defined(RABITQ_TARGET_X86_64)
void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d) {
    int info[4];
    __cpuidex(info, static_cast<int>(leaf), static_cast<int>(subleaf));
    *a = static_cast<uint32_t>(info[0]);
    *b = static_cast<uint32_t>(info[1]);
    *c = static_cast<uint32_t>(info[2]);
    *d = static_cast<uint32_t>(info[3]);
}

uint64_t xgetbv(uint32_t index) {
    return _xgetbv(index);
}
#elif defined(RABITQ_TARGET_X86_64)
void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d) {
    __cpuid_count(leaf, subleaf, *a, *b, *c, *d);
}

uint64_t xgetbv(uint32_t index) {
    uint32_t eax;
    uint32_t edx;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(index));
    return (static_cast<uint64_t>(edx) << 32U) | eax;
}
#endif

bool detect_neon() noexcept {
#if defined(RABITQ_TARGET_AARCH64) && defined(__APPLE__)
    // Advanced SIMD is part of the arm64 execution environment on Apple platforms.
    return true;
#elif defined(RABITQ_TARGET_AARCH64) && defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#elif defined(RABITQ_TARGET_AARCH64)
    return true;
#else
    return false;
#endif
}

Features detect_features() noexcept {
    Features detected{};

#if defined(RABITQ_TARGET_X86_64)
    uint32_t max_leaf = 0;
    uint32_t ignored_b = 0;
    uint32_t ignored_c = 0;
    uint32_t ignored_d = 0;
    cpuid(0, 0, &max_leaf, &ignored_b, &ignored_c, &ignored_d);
    if (max_leaf >= 1) {
        // leaf 1: ECX[28]=AVX, ECX[27]=OSXSAVE, ECX[12]=FMA
        uint32_t eax = 0;
        uint32_t ebx = 0;
        uint32_t ecx = 0;
        uint32_t edx = 0;
        cpuid(1, 0, &eax, &ebx, &ecx, &edx);

        const bool avx = ((ecx >> 28U) & 1U) != 0;
        const bool osxsave = ((ecx >> 27U) & 1U) != 0;
        detected.fma = ((ecx >> 12U) & 1U) != 0;

        if (avx && osxsave) {
            const uint64_t xcr0 = xgetbv(0);
            detected.avx_os_support = (xcr0 & 0x6U) == 0x6U;
            detected.avx512_os_support = (xcr0 & 0xe6U) == 0xe6U;
        }
    }

    if (max_leaf >= 7) {
        // leaf 7 (subleaf 0): EBX[5]=AVX2, EBX[16]=AVX512F,
        //                     EBX[17]=AVX512DQ, EBX[30]=AVX512BW,
        //                     ECX[14]=AVX512_VPOPCNTDQ
        uint32_t eax = 0;
        uint32_t ebx = 0;
        uint32_t ecx = 0;
        uint32_t edx = 0;
        cpuid(7, 0, &eax, &ebx, &ecx, &edx);

        detected.avx2 = ((ebx >> 5U) & 1U) != 0;
        detected.avx512f = ((ebx >> 16U) & 1U) != 0;
        detected.avx512dq = ((ebx >> 17U) & 1U) != 0;
        detected.avx512bw = ((ebx >> 30U) & 1U) != 0;
        detected.avx512vpopcntdq = ((ecx >> 14U) & 1U) != 0;
    }
#endif

    detected.neon = detect_neon();
    return detected;
}

}  // namespace

const Features& features() {
    static const Features detected = detect_features();
    return detected;
}

bool has_avx2() {
    const Features& detected = features();
    return detected.avx2 && detected.fma && detected.avx_os_support;
}

bool has_avx512_core() {
    const Features& detected = features();
    return detected.avx512f && detected.avx512bw && detected.avx512dq &&
           detected.avx512_os_support;
}

bool has_avx512_popcnt() {
    return has_avx512_core() && features().avx512vpopcntdq;
}

bool has_neon() {
    return features().neon;
}

}  // namespace rabitqlib::cpu
