#pragma once

#if !defined(RABITQ_TARGET_X86_64) && !defined(RABITQ_TARGET_AARCH64)
#if defined(__x86_64__) || defined(_M_X64)
#define RABITQ_TARGET_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define RABITQ_TARGET_AARCH64 1
#endif
#endif

#if defined(RABITQ_TARGET_X86_64) && defined(RABITQ_TARGET_AARCH64)
#error "RaBitQ target architecture is ambiguous"
#endif
