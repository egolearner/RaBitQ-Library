#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::simd {

enum class Backend : uint8_t {
    Unavailable,
    Scalar,
    Neon,
    Avx2,
    Avx512Core,
    Avx512Popcnt,
};

using ExcodeIpTable = std::array<ex_ipfunc, 9>;

ExcodeIpTable resolve_excode_ip_table();
Backend selected_backend() noexcept;
const char* backend_name(Backend backend) noexcept;
bool backend_is_compiled(Backend backend) noexcept;

}  // namespace rabitqlib::simd
