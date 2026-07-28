#include "index/hnsw_search_backend.hpp"

#include "simd/backend.hpp"

namespace rabitqlib::hnsw::detail {

struct HnswScalarKernel {
    static inline float warmup_ip_x0_q_512(
        const uint64_t* data,
        const uint64_t* query,
        float delta,
        float vl,
        size_t padded_dim,
        size_t b_query
    ) {
        return rabitqlib::simd::warmup_ip_x0_q_512_scalar(
            data, query, delta, vl, padded_dim, b_query
        );
    }

    static inline float mask_ip_x0_q(
        const float* query, const uint64_t* data, size_t padded_dim
    ) {
        return rabitqlib::simd::mask_ip_x0_q_scalar(query, data, padded_dim);
    }
};

maxheap<std::pair<float, PID>> search_knn_scalar(
    HierarchicalNSW& index, const float* rotated_query, size_t topk
) {
    return HnswSearchAccess::search<HnswScalarKernel>(index, rotated_query, topk);
}

}  // namespace rabitqlib::hnsw::detail
