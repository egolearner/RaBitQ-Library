#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::fastscan {

void accumulate(const uint8_t* codes, const uint8_t* lp_table, uint16_t* result, size_t dim);
void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut);
void accumulate_hacc(const uint8_t* codes, const uint8_t* hc_lut, int32_t* result, size_t dim);

}  // namespace rabitqlib::fastscan
