#include <iostream>
#include <vector>
#include <unordered_set>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::ivf::IVF;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

static size_t topk = 100;
static size_t test_round = 1;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4>\n"
                  << "arg1: path for index \n"
                  << "arg2: path for query file, format .fvecs\n"
                  << "arg3: path for groundtruth file format .ivecs\n"
                  << "arg4: whether use high accuracy fastscan, (\"true\" or \"false\"), "
                     "true by default\n\n";
        exit(1);
    }

    char* index_file = argv[1];
    char* query_file = argv[2];
    char* gt_file = argv[3];
    bool use_hacc = true;

    if (argc > 4) {
        std::string hacc_str(argv[4]);
        if (hacc_str == "false") {
            use_hacc = false;
            std::cout << "Do not use Hacc FastScan\n";
        }
    }

    data_type query;
    gt_type gt;
    rabitqlib::load_vecs<float, data_type>(query_file, query);
    rabitqlib::load_vecs<uint32_t, gt_type>(gt_file, gt);
    size_t nq = query.rows();
    size_t total_count = nq * topk;

    index_type ivf;
    ivf.load(index_file);

    std::vector<std::unordered_set<PID>> gt_sets(nq);
    for (size_t i = 0; i < nq; i++) {
        for (size_t k = 0; k < topk; k++) {
            gt_sets[i].insert(gt(i, k));
        }
    }

    std::vector<size_t> nprobes = {5, 10, 20, 40, 80, 120, 200, 300, 400, 600, 800, 1000};
    size_t length = nprobes.size();

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(length));

    rabitqlib::StopW stopw;

    for (size_t r = 0; r < test_round; r++) {
        for (size_t l = 0; l < length; ++l) {
            size_t nprobe = nprobes[l];
            if (nprobe > ivf.num_clusters()) {
                std::cout << "nprobe " << nprobe << " is larger than number of clusters, ";
                std::cout << "will use nprobe = num_cluster (" << ivf.num_clusters() << ").\n";
            }
            size_t total_correct = 0;
            float total_time = 0;
            std::vector<PID> results(topk);
            for (size_t i = 0; i < nq; i++) {
                stopw.reset();
                ivf.search(&query(i, 0), topk, nprobe, results.data(), use_hacc);
                total_time += stopw.get_elapsed_micro();
                const auto& gt_set = gt_sets[i];
                for (size_t j = 0; j < topk; j++) {
                    if (gt_set.count(results[j])) {
                        total_correct++;
                    }
                }
            }
            float qps = static_cast<float>(nq) / (total_time / 1e6F);
            float recall =
                static_cast<float>(total_correct) / static_cast<float>(total_count);

            all_qps[r][l] = qps;
            all_recall[r][l] = recall;
        }
    }

    auto avg_qps = rabitqlib::horizontal_avg(all_qps);
    auto avg_recall = rabitqlib::horizontal_avg(all_recall);

    std::cout << "nprobe\tQPS\trecall" << '\n';

    for (size_t i = 0; i < length; ++i) {
        size_t nprobe = nprobes[i];
        float qps = avg_qps[i];
        float recall = avg_recall[i];

        std::cout << nprobe << '\t' << qps << '\t' << recall << '\n';
    }

    return 0;
}
