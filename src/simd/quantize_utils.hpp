#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

namespace rabitqlib::simd::detail {

template <typename T>
inline T quantize_nearest_even(float value) noexcept {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);

    if (!(value > 0.0F)) {
        return 0;
    }

    constexpr T kMaximum = std::numeric_limits<T>::max();
    if (value >= static_cast<float>(kMaximum)) {
        return kMaximum;
    }

    const float integral = std::floor(value);
    const float fraction = value - integral;
    auto rounded = static_cast<unsigned long long>(integral);
    if (fraction > 0.5F || (fraction == 0.5F && (rounded & 1U) != 0)) {
        ++rounded;
    }
    return static_cast<T>(rounded);
}

}  // namespace rabitqlib::simd::detail
