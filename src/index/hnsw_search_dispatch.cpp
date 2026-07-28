#include "index/hnsw_search_backend.hpp"

#include <stdexcept>

#include "rabitqlib/simd/dispatch.hpp"

namespace rabitqlib::hnsw {

maxheap<std::pair<float, PID>> HierarchicalNSW::search_knn(
    const float* rotated_query, size_t topk
) {
    switch (rabitqlib::simd::selected_backend()) {
#if defined(RABITQ_TARGET_X86_64)
        case rabitqlib::simd::Backend::Avx512Popcnt:
            return detail::search_knn_avx512_popcnt(*this, rotated_query, topk);
        case rabitqlib::simd::Backend::Avx512Core:
            return detail::search_knn_avx512_core(*this, rotated_query, topk);
        case rabitqlib::simd::Backend::Avx2:
            return detail::search_knn_avx2(*this, rotated_query, topk);
#endif
#if defined(RABITQ_TARGET_AARCH64)
        case rabitqlib::simd::Backend::Neon:
            return detail::search_knn_neon(*this, rotated_query, topk);
#endif
#if defined(RABITQ_ENABLE_TEST_BACKENDS)
        case rabitqlib::simd::Backend::Scalar:
            return detail::search_knn_scalar(*this, rotated_query, topk);
#endif
        default:
            break;
    }

    throw std::runtime_error("HNSW search has no usable SIMD backend");
}

}  // namespace rabitqlib::hnsw
