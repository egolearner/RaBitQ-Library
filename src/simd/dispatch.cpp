#include "rabitqlib/simd/dispatch.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "rabitqlib/simd/fastscan_dispatch.hpp"
#include "rabitqlib/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/simd/rotator_dispatch.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "simd/backend.hpp"

namespace rabitqlib::simd {
namespace {

using FlipSignFn = void (*)(const uint8_t*, float*, size_t);
using KacsWalkFn = void (*)(float*, size_t);
using ScalarQuantizeUint8Fn = void (*)(uint8_t*, const float*, size_t, float, float);
using ScalarQuantizeUint16Fn = void (*)(uint16_t*, const float*, size_t, float, float);
using PackExcodeFn = void (*)(const uint8_t*, uint8_t*, size_t);
using NewTransposeBinFn = void (*)(const uint16_t*, uint64_t*, size_t, size_t);
using NewTransposeBin512Fn = void (*)(const uint8_t*, uint64_t*, size_t, size_t);
using MaskIpX0QFn = float (*)(const float*, const uint64_t*, size_t);
using AccumulateFn = void (*)(const uint8_t*, const uint8_t*, uint16_t*, size_t);
using TransferLutHaccFn = void (*)(const uint16_t*, size_t, uint8_t*);
using AccumulateHaccFn = void (*)(const uint8_t*, const uint8_t*, int32_t*, size_t);
using WarmupIpX0Q512Fn =
    float (*)(const uint64_t*, const uint64_t*, float, float, size_t, size_t);

struct DispatchTable {
    Backend backend;
    ExcodeIpTable excode_ip;
    FlipSignFn flip_sign;
    KacsWalkFn kacs_walk;
    ScalarQuantizeUint8Fn quantize_uint8;
    ScalarQuantizeUint16Fn quantize_uint16;
    std::array<PackExcodeFn, 8> pack_excode;
    NewTransposeBinFn transpose_bin;
    NewTransposeBin512Fn transpose_bin_512;
    MaskIpX0QFn mask_ip;
    AccumulateFn accumulate;
    TransferLutHaccFn transfer_lut_hacc;
    AccumulateHaccFn accumulate_hacc;
    WarmupIpX0Q512Fn warmup_ip;
};

[[noreturn]] void missing_void() {
    throw std::runtime_error("No usable SIMD backend was built for this operation");
}

float missing_excode(const float*, const uint8_t*, size_t) {
    missing_void();
}

void missing_flip(const uint8_t*, float*, size_t) {
    missing_void();
}

void missing_kacs(float*, size_t) {
    missing_void();
}

void missing_quantize_uint8(uint8_t*, const float*, size_t, float, float) {
    missing_void();
}

void missing_quantize_uint16(uint16_t*, const float*, size_t, float, float) {
    missing_void();
}

void missing_pack(const uint8_t*, uint8_t*, size_t) {
    missing_void();
}

void missing_transpose(const uint16_t*, uint64_t*, size_t, size_t) {
    missing_void();
}

void missing_transpose_512(const uint8_t*, uint64_t*, size_t, size_t) {
    missing_void();
}

float missing_mask(const float*, const uint64_t*, size_t) {
    missing_void();
}

void missing_accumulate(const uint8_t*, const uint8_t*, uint16_t*, size_t) {
    missing_void();
}

void missing_transfer(const uint16_t*, size_t, uint8_t*) {
    missing_void();
}

void missing_accumulate_hacc(const uint8_t*, const uint8_t*, int32_t*, size_t) {
    missing_void();
}

float missing_warmup(const uint64_t*, const uint64_t*, float, float, size_t, size_t) {
    missing_void();
}

DispatchTable missing_table() noexcept {
    return {
        Backend::Unavailable,
        {missing_excode,
         missing_excode,
         missing_excode,
         missing_excode,
         missing_excode,
         missing_excode,
         missing_excode,
         missing_excode,
         missing_excode},
        missing_flip,
        missing_kacs,
        missing_quantize_uint8,
        missing_quantize_uint16,
        {missing_pack,
         missing_pack,
         missing_pack,
         missing_pack,
         missing_pack,
         missing_pack,
         missing_pack,
         missing_pack},
        missing_transpose,
        missing_transpose_512,
        missing_mask,
        missing_accumulate,
        missing_transfer,
        missing_accumulate_hacc,
        missing_warmup,
    };
}

#if defined(RABITQ_ENABLE_TEST_BACKENDS)
constexpr bool kScalarCompiled = true;
#else
constexpr bool kScalarCompiled = false;
#endif
#if defined(RABITQ_TARGET_AARCH64)
constexpr bool kNeonCompiled = true;
#else
constexpr bool kNeonCompiled = false;
#endif
#if defined(RABITQ_TARGET_X86_64)
constexpr bool kX86Compiled = true;
#else
constexpr bool kX86Compiled = false;
#endif

bool unavailable_is_usable() noexcept {
    return false;
}

bool scalar_is_usable() noexcept {
    return kScalarCompiled;
}

bool neon_is_usable() noexcept {
    return kNeonCompiled && cpu::has_neon();
}

bool avx2_is_usable() noexcept {
    return kX86Compiled && cpu::has_avx2();
}

bool avx512_core_is_usable() noexcept {
    return kX86Compiled && cpu::has_avx512_core() && cpu::has_avx2();
}

bool avx512_popcnt_is_usable() noexcept {
    return kX86Compiled && cpu::has_avx512_popcnt() && cpu::has_avx2();
}

using BackendUsableFn = bool (*)() noexcept;

struct BackendDescriptor {
    Backend backend;
    const char* name;
    bool compiled;
    BackendUsableFn is_usable;
};

constexpr std::array<BackendDescriptor, 6> kBackendDescriptors = {{
    {Backend::Unavailable, "unavailable", true, unavailable_is_usable},
    {Backend::Scalar, "scalar", kScalarCompiled, scalar_is_usable},
    {Backend::Neon, "neon", kNeonCompiled, neon_is_usable},
    {Backend::Avx2, "avx2", kX86Compiled, avx2_is_usable},
    {Backend::Avx512Core, "avx512_core", kX86Compiled, avx512_core_is_usable},
    {Backend::Avx512Popcnt, "avx512_popcnt", kX86Compiled, avx512_popcnt_is_usable},
}};

const BackendDescriptor* find_backend_descriptor(Backend backend) noexcept {
    for (const BackendDescriptor& descriptor : kBackendDescriptors) {
        if (descriptor.backend == backend) {
            return &descriptor;
        }
    }
    return nullptr;
}

bool backend_is_usable(Backend backend) noexcept {
    const BackendDescriptor* descriptor = find_backend_descriptor(backend);
    return descriptor != nullptr && descriptor->is_usable();
}

#if defined(RABITQ_ENABLE_TEST_BACKENDS)
Backend parse_test_backend(const char* value) noexcept {
    if (value == nullptr) {
        return Backend::Unavailable;
    }
    if (std::strcmp(value, "avx512") == 0) {
        return Backend::Avx512Core;
    }
    for (const BackendDescriptor& descriptor : kBackendDescriptors) {
        if (std::strcmp(value, descriptor.name) == 0) {
            return descriptor.backend;
        }
    }
    return Backend::Unavailable;
}
#endif

Backend choose_backend() noexcept {
#if defined(RABITQ_ENABLE_TEST_BACKENDS)
    const char* forced = std::getenv("RABITQ_TEST_SIMD_BACKEND");
    if (forced != nullptr) {
        const Backend requested = parse_test_backend(forced);
        return backend_is_usable(requested) ? requested : Backend::Unavailable;
    }
#endif

#if defined(RABITQ_TARGET_X86_64)
    if (backend_is_usable(Backend::Avx512Popcnt)) {
        return Backend::Avx512Popcnt;
    }
    if (backend_is_usable(Backend::Avx512Core)) {
        return Backend::Avx512Core;
    }
    if (backend_is_usable(Backend::Avx2)) {
        return Backend::Avx2;
    }
#elif defined(RABITQ_TARGET_AARCH64)
    if (backend_is_usable(Backend::Neon)) {
        return Backend::Neon;
    }
#endif
    return Backend::Unavailable;
}

template <typename Ip1, typename Ip2, typename Ip3, typename Ip4, typename Ip5, typename Ip6,
          typename Ip7, typename Ip8>
void set_excode_table(
    DispatchTable& table,
    Ip1 ip1,
    Ip2 ip2,
    Ip3 ip3,
    Ip4 ip4,
    Ip5 ip5,
    Ip6 ip6,
    Ip7 ip7,
    Ip8 ip8
) noexcept {
    table.excode_ip = {ip1, ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8};
}

#define RABITQ_SET_COMMON_BACKEND(table, suffix)                                             \
    do {                                                                                     \
        (table).flip_sign = flip_sign_##suffix;                                              \
        (table).kacs_walk = kacs_walk_##suffix;                                              \
        (table).quantize_uint8 = scalar_quantize_uint8_##suffix;                             \
        (table).quantize_uint16 = scalar_quantize_uint16_##suffix;                           \
        (table).pack_excode[2] = packing_2bit_excode_##suffix;                               \
        (table).pack_excode[3] = packing_3bit_excode_##suffix;                               \
        (table).pack_excode[4] = packing_4bit_excode_##suffix;                               \
        (table).pack_excode[5] = packing_5bit_excode_##suffix;                               \
        (table).pack_excode[6] = packing_6bit_excode_##suffix;                               \
        (table).pack_excode[7] = packing_7bit_excode_##suffix;                               \
        (table).transpose_bin = new_transpose_bin_##suffix;                                  \
        (table).transpose_bin_512 = new_transpose_bin_512_##suffix;                          \
        (table).mask_ip = mask_ip_x0_q_##suffix;                                             \
        (table).accumulate = fastscan::simd::accumulate_##suffix;                            \
        (table).transfer_lut_hacc = fastscan::simd::transfer_lut_hacc_##suffix;               \
        (table).accumulate_hacc = fastscan::simd::accumulate_hacc_##suffix;                   \
        (table).warmup_ip = warmup_ip_x0_q_512_##suffix;                                     \
    } while (false)

DispatchTable make_dispatch_table() noexcept {
    DispatchTable table = missing_table();
    table.backend = choose_backend();

    switch (table.backend) {
#if defined(RABITQ_ENABLE_TEST_BACKENDS)
        case Backend::Scalar:
            set_excode_table(
                table,
                excode_ipimpl::ip16_fxu1_scalar,
                excode_ipimpl::ip64_fxu2_scalar,
                excode_ipimpl::ip64_fxu3_scalar,
                excode_ipimpl::ip16_fxu4_scalar,
                excode_ipimpl::ip64_fxu5_scalar,
                excode_ipimpl::ip64_fxu6_scalar,
                excode_ipimpl::ip64_fxu7_scalar,
                excode_ipimpl::ip16_fxu8_scalar
            );
            RABITQ_SET_COMMON_BACKEND(table, scalar);
            break;
#endif
#if defined(RABITQ_TARGET_AARCH64)
        case Backend::Neon:
            set_excode_table(
                table,
                excode_ipimpl::ip16_fxu1_neon,
                excode_ipimpl::ip64_fxu2_neon,
                excode_ipimpl::ip64_fxu3_neon,
                excode_ipimpl::ip16_fxu4_neon,
                excode_ipimpl::ip64_fxu5_neon,
                excode_ipimpl::ip64_fxu6_neon,
                excode_ipimpl::ip64_fxu7_neon,
                excode_ipimpl::ip16_fxu8_neon
            );
            RABITQ_SET_COMMON_BACKEND(table, neon);
            break;
#endif
#if defined(RABITQ_TARGET_X86_64)
        case Backend::Avx512Popcnt:
        case Backend::Avx512Core:
            set_excode_table(
                table,
                excode_ipimpl::ip16_fxu1_avx512,
                excode_ipimpl::ip64_fxu2_avx512,
                excode_ipimpl::ip64_fxu3_avx512,
                excode_ipimpl::ip16_fxu4_avx512,
                excode_ipimpl::ip64_fxu5_avx512,
                excode_ipimpl::ip64_fxu6_avx512,
                excode_ipimpl::ip64_fxu7_avx512,
                excode_ipimpl::ip16_fxu8_avx512
            );
            RABITQ_SET_COMMON_BACKEND(table, avx512);
            if (table.backend == Backend::Avx512Core) {
                table.warmup_ip = warmup_ip_x0_q_512_avx2;
            }
            break;
        case Backend::Avx2:
            set_excode_table(
                table,
                excode_ipimpl::ip16_fxu1_avx2,
                excode_ipimpl::ip64_fxu2_avx2,
                excode_ipimpl::ip64_fxu3_avx2,
                excode_ipimpl::ip16_fxu4_avx2,
                excode_ipimpl::ip64_fxu5_avx2,
                excode_ipimpl::ip64_fxu6_avx2,
                excode_ipimpl::ip64_fxu7_avx2,
                excode_ipimpl::ip16_fxu8_avx2
            );
            RABITQ_SET_COMMON_BACKEND(table, avx2);
            break;
#endif
        case Backend::Unavailable:
            break;
        default:
            table = missing_table();
            break;
    }
    return table;
}

#undef RABITQ_SET_COMMON_BACKEND

const DispatchTable kDispatchTable = make_dispatch_table();

}  // namespace

Backend selected_backend() noexcept {
    return kDispatchTable.backend;
}

const char* backend_name(Backend backend) noexcept {
    const BackendDescriptor* descriptor = find_backend_descriptor(backend);
    return descriptor != nullptr ? descriptor->name : "unavailable";
}

bool backend_is_compiled(Backend backend) noexcept {
    const BackendDescriptor* descriptor = find_backend_descriptor(backend);
    return descriptor != nullptr && descriptor->compiled;
}

ExcodeIpTable resolve_excode_ip_table() {
    return kDispatchTable.excode_ip;
}

void flip_sign(const uint8_t* flip, float* data, size_t dim) {
    kDispatchTable.flip_sign(flip, data, dim);
}

void kacs_walk(float* data, size_t len) {
    kDispatchTable.kacs_walk(data, len);
}

void scalar_quantize_uint8(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    kDispatchTable.quantize_uint8(result, vec0, dim, lo, delta);
}

void scalar_quantize_uint16(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    kDispatchTable.quantize_uint16(result, vec0, dim, lo, delta);
}

void packing_2bit_excode(const uint8_t* raw, uint8_t* compact, size_t dim) {
    kDispatchTable.pack_excode[2](raw, compact, dim);
}

void packing_3bit_excode(const uint8_t* raw, uint8_t* compact, size_t dim) {
    kDispatchTable.pack_excode[3](raw, compact, dim);
}

void packing_4bit_excode(const uint8_t* raw, uint8_t* compact, size_t dim) {
    kDispatchTable.pack_excode[4](raw, compact, dim);
}

void packing_5bit_excode(const uint8_t* raw, uint8_t* compact, size_t dim) {
    kDispatchTable.pack_excode[5](raw, compact, dim);
}

void packing_6bit_excode(const uint8_t* raw, uint8_t* compact, size_t dim) {
    kDispatchTable.pack_excode[6](raw, compact, dim);
}

void packing_7bit_excode(const uint8_t* raw, uint8_t* compact, size_t dim) {
    kDispatchTable.pack_excode[7](raw, compact, dim);
}

}  // namespace rabitqlib::simd

namespace rabitqlib {

const simd::ExcodeIpTable kExcodeIpTable = simd::resolve_excode_ip_table();

const ex_ipfunc kIp16Fxu1AvxFn = kExcodeIpTable[1];
const ex_ipfunc kIp64Fxu2AvxFn = kExcodeIpTable[2];
const ex_ipfunc kIp64Fxu3AvxFn = kExcodeIpTable[3];
const ex_ipfunc kIp16Fxu4AvxFn = kExcodeIpTable[4];
const ex_ipfunc kIp64Fxu5AvxFn = kExcodeIpTable[5];
const ex_ipfunc kIp64Fxu6AvxFn = kExcodeIpTable[6];
const ex_ipfunc kIp64Fxu7AvxFn = kExcodeIpTable[7];

ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    if (ex_bits <= 8) {
        return kExcodeIpTable[ex_bits];
    }
    throw std::invalid_argument("Bad IP function for IVF");
}

float excode_ipimpl::ip16_fxu1_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp16Fxu1AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu2_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu2AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu3_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu3AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip16_fxu4_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp16Fxu4AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu5_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu5AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu6_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu6AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu7_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu7AvxFn(query, compact_code, dim);
}

void new_transpose_bin(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    simd::kDispatchTable.transpose_bin(q, tq, padded_dim, b_query);
}

void new_transpose_bin_512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    simd::kDispatchTable.transpose_bin_512(q, tq, padded_dim, b_query);
}

float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
    return simd::kDispatchTable.mask_ip(query, data, padded_dim);
}

}  // namespace rabitqlib

namespace rabitqlib::fastscan {

void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
) {
    rabitqlib::simd::kDispatchTable.accumulate(codes, lp_table, result, dim);
}

void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    rabitqlib::simd::kDispatchTable.transfer_lut_hacc(lut, dim, hc_lut);
}

void accumulate_hacc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* result,
    size_t dim
) {
    rabitqlib::simd::kDispatchTable.accumulate_hacc(codes, hc_lut, result, dim);
}

}  // namespace rabitqlib::fastscan

namespace rabitqlib {

float warmup_ip_x0_q_512(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    return simd::kDispatchTable.warmup_ip(
        data, query, delta, vl, padded_dim, b_query
    );
}

}  // namespace rabitqlib
