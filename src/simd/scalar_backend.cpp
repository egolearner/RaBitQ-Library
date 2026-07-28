#include "simd/backend.hpp"

#include "simd/reference_kernels.hpp"

namespace rabitqlib::simd {
namespace excode_ipimpl {

float ip16_fxu1_scalar(const float* query, const uint8_t* code, size_t dim) {
    return reference::excode_ip(query, code, dim, 1);
}

float ip64_fxu2_scalar(const float* query, const uint8_t* code, size_t dim) {
    return reference::excode_ip(query, code, dim, 2);
}

float ip64_fxu3_scalar(const float* query, const uint8_t* code, size_t dim) {
    return reference::excode_ip(query, code, dim, 3);
}

float ip16_fxu4_scalar(const float* query, const uint8_t* code, size_t dim) {
    return reference::excode_ip(query, code, dim, 4);
}

float ip64_fxu5_scalar(const float* query, const uint8_t* code, size_t dim) {
    return reference::excode_ip(query, code, dim, 5);
}

float ip64_fxu6_scalar(const float* query, const uint8_t* code, size_t dim) {
    return reference::excode_ip(query, code, dim, 6);
}

float ip64_fxu7_scalar(const float* query, const uint8_t* code, size_t dim) {
    return reference::excode_ip(query, code, dim, 7);
}

float ip16_fxu8_scalar(const float* query, const uint8_t* code, size_t dim) {
    return reference::excode_ip(query, code, dim, 8);
}

}  // namespace excode_ipimpl

void packing_2bit_excode_scalar(const uint8_t* raw, uint8_t* compact, size_t dim) {
    reference::packing_2bit_excode(raw, compact, dim);
}

void packing_3bit_excode_scalar(const uint8_t* raw, uint8_t* compact, size_t dim) {
    reference::packing_3bit_excode(raw, compact, dim);
}

void packing_4bit_excode_scalar(const uint8_t* raw, uint8_t* compact, size_t dim) {
    reference::packing_4bit_excode(raw, compact, dim);
}

void packing_5bit_excode_scalar(const uint8_t* raw, uint8_t* compact, size_t dim) {
    reference::packing_5bit_excode(raw, compact, dim);
}

void packing_6bit_excode_scalar(const uint8_t* raw, uint8_t* compact, size_t dim) {
    reference::packing_6bit_excode(raw, compact, dim);
}

void packing_7bit_excode_scalar(const uint8_t* raw, uint8_t* compact, size_t dim) {
    reference::packing_7bit_excode(raw, compact, dim);
}

void scalar_quantize_uint8_scalar(
    uint8_t* result, const float* input, size_t dim, float lo, float delta
) {
    reference::scalar_quantize(result, input, dim, lo, delta);
}

void scalar_quantize_uint16_scalar(
    uint16_t* result, const float* input, size_t dim, float lo, float delta
) {
    reference::scalar_quantize(result, input, dim, lo, delta);
}

void new_transpose_bin_scalar(
    const uint16_t* query, uint64_t* transposed, size_t dim, size_t bits
) {
    reference::new_transpose_bin(query, transposed, dim, bits);
}

void new_transpose_bin_512_scalar(
    const uint8_t* query, uint64_t* transposed, size_t dim, size_t bits
) {
    reference::new_transpose_bin_512(query, transposed, dim, bits);
}

float mask_ip_x0_q_scalar(const float* query, const uint64_t* data, size_t dim) {
    return reference::mask_ip_x0_q(query, data, dim);
}

void flip_sign_scalar(const uint8_t* flip, float* data, size_t dim) {
    reference::flip_sign(flip, data, dim);
}

void kacs_walk_scalar(float* data, size_t len) {
    reference::kacs_walk(data, len);
}

float warmup_ip_x0_q_512_scalar(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t dim,
    size_t bits
) {
    return reference::warmup_ip_x0_q_512(data, query, delta, vl, dim, bits);
}

}  // namespace rabitqlib::simd

namespace rabitqlib::fastscan::simd {

void accumulate_scalar(
    const uint8_t* codes, const uint8_t* lut, uint16_t* result, size_t dim
) {
    rabitqlib::simd::reference::fastscan_accumulate(codes, lut, result, dim);
}

void transfer_lut_hacc_scalar(
    const uint16_t* lut, size_t dim, uint8_t* high_accuracy_lut
) {
    rabitqlib::simd::reference::transfer_lut_hacc(
        lut, dim, high_accuracy_lut
    );
}

void accumulate_hacc_scalar(
    const uint8_t* codes, const uint8_t* lut, int32_t* result, size_t dim
) {
    rabitqlib::simd::reference::fastscan_accumulate_hacc(
        codes, lut, result, dim
    );
}

}  // namespace rabitqlib::fastscan::simd
