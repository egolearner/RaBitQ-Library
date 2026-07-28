#pragma once

#include "rabitqlib/index/hnsw/hnsw.hpp"

namespace rabitqlib::hnsw::detail {

class HnswSearchAccess {
   public:
    template <class Kernel>
    static maxheap<std::pair<float, PID>> search(
        HierarchicalNSW& index, const float* rotated_query, size_t topk
    ) {
        return index.search_knn_direct<Kernel>(rotated_query, topk);
    }
};

#if defined(RABITQ_TARGET_X86_64)
maxheap<std::pair<float, PID>> search_knn_avx2(
    HierarchicalNSW&, const float*, size_t
);

maxheap<std::pair<float, PID>> search_knn_avx512_core(
    HierarchicalNSW&, const float*, size_t
);

maxheap<std::pair<float, PID>> search_knn_avx512_popcnt(
    HierarchicalNSW&, const float*, size_t
);
#endif

#if defined(RABITQ_TARGET_AARCH64)
maxheap<std::pair<float, PID>> search_knn_neon(
    HierarchicalNSW&, const float*, size_t
);
#endif

#if defined(RABITQ_ENABLE_TEST_BACKENDS)
maxheap<std::pair<float, PID>> search_knn_scalar(
    HierarchicalNSW&, const float*, size_t
);
#endif

}  // namespace rabitqlib::hnsw::detail
