#pragma once

#include <array>
#include <cstddef>

#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::simd {

using ExcodeIpTable = std::array<ex_ipfunc, 9>;

ExcodeIpTable resolve_excode_ip_table();
void require_avx2(const char* feature_name);

}  // namespace rabitqlib::simd
