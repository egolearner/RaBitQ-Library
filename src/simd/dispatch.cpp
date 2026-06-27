#include "rabitqlib/simd/dispatch.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/fastscan_dispatch.hpp"
#include "rabitqlib/simd/rotator_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::simd {

[[noreturn]] static void missing_feature(const char* feature_name) {
    throw std::runtime_error(std::string(feature_name) + " requires AVX2/FMA or AVX512 support");
}

ExcodeIpTable resolve_excode_ip_table() {
    if (cpu::has_avx512_core()) {
        return {
            excode_ipimpl::ip16_fxu1_avx512,
            excode_ipimpl::ip16_fxu1_avx512,
            excode_ipimpl::ip64_fxu2_avx512,
            excode_ipimpl::ip64_fxu3_avx512,
            excode_ipimpl::ip16_fxu4_avx512,
            excode_ipimpl::ip64_fxu5_avx512,
            excode_ipimpl::ip64_fxu6_avx512,
            excode_ipimpl::ip64_fxu7_avx512,
            rabitqlib::excode_ipimpl::ip_fxi<float, uint8_t>,
        };
    } else if (cpu::has_avx2()) {
        return {
            excode_ipimpl::ip16_fxu1_avx2,
            excode_ipimpl::ip16_fxu1_avx2,
            excode_ipimpl::ip64_fxu2_avx2,
            excode_ipimpl::ip64_fxu3_avx2,
            excode_ipimpl::ip16_fxu4_avx2,
            excode_ipimpl::ip64_fxu5_avx2,
            excode_ipimpl::ip64_fxu6_avx2,
            excode_ipimpl::ip64_fxu7_avx2,
            rabitqlib::excode_ipimpl::ip_fxi<float, uint8_t>,
        };
    } else {
        missing_feature("excode ip functions");
    }
}

void flip_sign(const uint8_t* flip, float* data, size_t dim) {
    using Fn = void (*)(const uint8_t*, float*, size_t);
    static const Fn fn = [] {
        if (cpu::has_avx512_core()) {
            return flip_sign_avx512;
        } else if (cpu::has_avx2()) {
            return flip_sign_avx2;
        } else {
            missing_feature("sign flip");
        }
    }();
    fn(flip, data, dim);
}

void kacs_walk(float* data, size_t len) {
    using Fn = void (*)(float*, size_t);
    static const Fn fn = [] {
        if (cpu::has_avx512_core()) {
            return kacs_walk_avx512;
        } else if (cpu::has_avx2()) {
            return kacs_walk_avx2;
        } else {
            missing_feature("FhtKacRotator");
        }
    }();
    fn(data, len);
}

void scalar_quantize_uint8(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    using Fn = void (*)(uint8_t*, const float*, size_t, float, float);
    static const Fn fn = [] {
        if (cpu::has_avx512_core()) {
            return scalar_quantize_uint8_avx512;
        } else if (cpu::has_avx2()) {
            return scalar_quantize_uint8_avx2;
        } else {
            missing_feature("uint8 quantize");
        }
    }();
    fn(result, vec0, dim, lo, delta);
}

void scalar_quantize_uint16(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    using Fn = void (*)(uint16_t*, const float*, size_t, float, float);
    static const Fn fn = [] {
        if (cpu::has_avx512_core()) {
            return scalar_quantize_uint16_avx512;
        } else if (cpu::has_avx2()) {
            return scalar_quantize_uint16_avx2;
        } else {
            missing_feature("uint16 quantize");
        }
    }();
    fn(result, vec0, dim, lo, delta);
}

}  // namespace rabitqlib::simd

namespace rabitqlib {

ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    static const auto table = simd::resolve_excode_ip_table();
    if (ex_bits <= 8) {
        return table[ex_bits];
    }

    throw std::invalid_argument("Bad IP function for IVF");
}

float excode_ipimpl::ip16_fxu1_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    static const ex_ipfunc fn = select_excode_ipfunc(1);
    return fn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu2_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    static const ex_ipfunc fn = select_excode_ipfunc(2);
    return fn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu3_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    static const ex_ipfunc fn = select_excode_ipfunc(3);
    return fn(query, compact_code, dim);
}

float excode_ipimpl::ip16_fxu4_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    static const ex_ipfunc fn = select_excode_ipfunc(4);
    return fn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu5_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    static const ex_ipfunc fn = select_excode_ipfunc(5);
    return fn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu6_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    static const ex_ipfunc fn = select_excode_ipfunc(6);
    return fn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu7_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    static const ex_ipfunc fn = select_excode_ipfunc(7);
    return fn(query, compact_code, dim);
}

void new_transpose_bin(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    using Fn = void (*)(const uint16_t*, uint64_t*, size_t, size_t);
    static const Fn fn = [] {
        if (cpu::has_avx512_core()) {
            return simd::new_transpose_bin_avx512;
        } else if (cpu::has_avx2()) {
            return simd::new_transpose_bin_avx2;
        } else {
            simd::missing_feature("new transpose bin");
        }
    }();
    fn(q, tq, padded_dim, b_query);
}

void new_transpose_bin_512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    using Fn = void (*)(const uint8_t*, uint64_t*, size_t, size_t);
    static const Fn fn = [] {
        if (cpu::has_avx512_core()) {
            return simd::new_transpose_bin_512_avx512;
        } else if (cpu::has_avx2()) {
            return simd::new_transpose_bin_512_avx2;
        } else {
            simd::missing_feature("new_transpose_bin_512");
        }
    }();
    fn(q, tq, padded_dim, b_query);
}

float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
    using Fn = float (*)(const float*, const uint64_t*, size_t);
    static const Fn fn = [] {
        if (cpu::has_avx512_core()) {
            return simd::mask_ip_x0_q_avx512;
        } else if (cpu::has_avx2()) {
            return simd::mask_ip_x0_q_avx2;
        } else {
            simd::missing_feature("mask ip x0 q");
        }
    }();
    return fn(query, data, padded_dim);
}

}  // namespace rabitqlib

namespace rabitqlib::fastscan {

void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
) {
    using Fn = void (*)(const uint8_t*, const uint8_t*, uint16_t*, size_t);
    static const Fn fn = [] {
        if (cpu::has_avx512_core()) {
            return simd::accumulate_avx512;
        } else if (cpu::has_avx2()) {
            return simd::accumulate_avx2;
        } else {
            rabitqlib::simd::missing_feature("fastscan accumulate");
        }
    }();
    fn(codes, lp_table, result, dim);
}

void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    using Fn = void (*)(const uint16_t*, size_t, uint8_t*);
    static const Fn fn = [] {
        if (cpu::has_avx512_core()) {
            return simd::transfer_lut_hacc_avx512;
        } else if (cpu::has_avx2()) {
            return simd::transfer_lut_hacc_avx2;
        } else {
            rabitqlib::simd::missing_feature("fastscan high-accuracy LUT transfer");
        }
    }();
    fn(lut, dim, hc_lut);
}

void accumulate_hacc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
) {
    using Fn = void (*)(const uint8_t*, const uint8_t*, int32_t*, size_t);
    static const Fn fn = [] {
        if (cpu::has_avx512_core()) {
            return simd::accumulate_hacc_avx512;
        } else if (cpu::has_avx2()) {
            return simd::accumulate_hacc_avx2;
        } else {
            rabitqlib::simd::missing_feature("fastscan high-accuracy accumulate");
        }
    }();
    fn(codes, hc_lut, accu_res, dim);
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
    using Fn = float (*)(const uint64_t*, const uint64_t*, float, float, size_t, size_t);
    static const Fn fn = [] {
        if (rabitqlib::cpu::has_avx512_popcnt()) {
            return rabitqlib::simd::warmup_ip_x0_q_512_avx512;
        } else if (rabitqlib::cpu::has_avx2()) {
            return rabitqlib::simd::warmup_ip_x0_q_512_avx2;
        } else {
            rabitqlib::simd::missing_feature("warmup_ip_x0_q_512");
        }
    }();
    return fn(data, query, delta, vl, padded_dim, b_query);
}

}  // namespace rabitqlib
