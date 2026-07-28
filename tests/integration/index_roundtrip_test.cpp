#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/index/symqg/qg.hpp"
#include "rabitqlib/index/symqg/qg_builder.hpp"

namespace {

constexpr size_t kDimension = 64;
constexpr size_t kNumPoints = 96;
constexpr size_t kNumClusters = 4;
constexpr size_t kTopK = 5;

struct DeterministicData {
    std::vector<float> points;
    std::vector<float> centroids;
    std::vector<rabitqlib::PID> cluster_ids;

    DeterministicData()
        : points(kNumPoints * kDimension)
        , centroids(kNumClusters * kDimension, 0.0F)
        , cluster_ids(kNumPoints) {
        std::array<size_t, kNumClusters> counts{};
        for (size_t point = 0; point < kNumPoints; ++point) {
            const size_t cluster = point % kNumClusters;
            cluster_ids[point] = static_cast<rabitqlib::PID>(cluster);
            ++counts[cluster];
            for (size_t dim = 0; dim < kDimension; ++dim) {
                const float value =
                    static_cast<float>(cluster * 20) +
                    static_cast<float>((point * 17 + dim * 13) % 101) / 101.0F;
                points[point * kDimension + dim] = value;
                centroids[cluster * kDimension + dim] += value;
            }
        }
        for (size_t cluster = 0; cluster < kNumClusters; ++cluster) {
            for (size_t dim = 0; dim < kDimension; ++dim) {
                centroids[cluster * kDimension + dim] /=
                    static_cast<float>(counts[cluster]);
            }
        }
    }
};

class TemporaryIndexFile {
   public:
    explicit TemporaryIndexFile(const char* name)
        : path_((std::filesystem::temp_directory_path() / name).string()) {
        std::remove(path_.c_str());
    }

    ~TemporaryIndexFile() {
        std::remove(path_.c_str());
    }

    const char* c_str() const {
        return path_.c_str();
    }

   private:
    std::string path_;
};

std::vector<rabitqlib::PID> sorted_ids(
    const std::vector<std::pair<float, rabitqlib::PID>>& result
) {
    std::vector<rabitqlib::PID> ids;
    ids.reserve(result.size());
    for (const auto& item : result) {
        ids.push_back(item.second);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

}  // namespace

TEST(IndexRoundTrip, IvfBuildSaveLoadAndQuery) {
    DeterministicData data;
    TemporaryIndexFile file("rabitq_ivf_arm_roundtrip.bin");
    rabitqlib::ivf::IVF index(
        kNumPoints,
        kDimension,
        kNumClusters,
        4,
        rabitqlib::METRIC_L2,
        rabitqlib::RotatorType::FhtKacRotator
    );
    index.construct(
        data.points.data(),
        data.centroids.data(),
        data.cluster_ids.data(),
        false,
        1
    );

    std::array<rabitqlib::PID, kTopK> expected{};
    std::array<rabitqlib::PID, kTopK> expected_hacc{};
    index.search(data.points.data(), kTopK, kNumClusters, expected.data(), false);
    index.search(data.points.data(), kTopK, kNumClusters, expected_hacc.data(), true);
    index.save(file.c_str());

    rabitqlib::ivf::IVF loaded;
    loaded.load(file.c_str());
    std::array<rabitqlib::PID, kTopK> actual{};
    std::array<rabitqlib::PID, kTopK> actual_hacc{};
    loaded.search(data.points.data(), kTopK, kNumClusters, actual.data(), false);
    loaded.search(data.points.data(), kTopK, kNumClusters, actual_hacc.data(), true);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(actual_hacc, expected_hacc);
}

TEST(IndexRoundTrip, HnswBuildSaveLoadAndQuery) {
    DeterministicData data;
    TemporaryIndexFile file("rabitq_hnsw_arm_roundtrip.bin");
    rabitqlib::hnsw::HierarchicalNSW index(
        kNumPoints, kDimension, 4, 8, 32, 100, rabitqlib::METRIC_L2
    );
    index.construct(
        kNumClusters,
        data.centroids.data(),
        kNumPoints,
        data.points.data(),
        data.cluster_ids.data(),
        1,
        false
    );
    const auto expected = index.search(data.points.data(), 3, kTopK, 32, 1);
    index.save(file.c_str());

    rabitqlib::hnsw::HierarchicalNSW loaded;
    loaded.load(file.c_str());
    const auto actual = loaded.search(data.points.data(), 3, kTopK, 32, 1);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t query = 0; query < actual.size(); ++query) {
        EXPECT_EQ(sorted_ids(actual[query]), sorted_ids(expected[query]));
    }
}

TEST(IndexRoundTrip, SymphonyQgBuildSaveLoadAndQuery) {
    DeterministicData data;
    TemporaryIndexFile file("rabitq_symqg_arm_roundtrip.bin");
    rabitqlib::symqg::QuantizedGraph<float> index(
        kNumPoints, kDimension, 32, rabitqlib::METRIC_L2
    );
    rabitqlib::symqg::QGBuilder builder(index, 64, data.points.data(), 1);
    builder.build(2);
    index.set_ef(64);

    std::array<rabitqlib::PID, kTopK> expected{};
    index.search(data.points.data(), kTopK, expected.data());
    index.save(file.c_str());

    rabitqlib::symqg::QuantizedGraph<float> loaded;
    loaded.load(file.c_str());
    loaded.set_ef(64);
    std::array<rabitqlib::PID, kTopK> actual{};
    loaded.search(data.points.data(), kTopK, actual.data());
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
}
