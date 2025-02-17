#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_MUTUAL_INFORMATION_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_MUTUAL_INFORMATION_HPP

#include <random>
#include <dataset/dataset.hpp>
#include <learning/independences/independence.hpp>
#include <learning/independences/hybrid/ms/vptree.hpp>
#include <kdtree/kdtree.hpp>

using dataset::DataFrame, dataset::Copy;
using Eigen::MatrixXi;
using Array_ptr = std::shared_ptr<arrow::Array>;
using vptree::VPTree, kdtree::IndexComparator;

namespace learning::independences::hybrid {
DataFrame scale_data_min_max(const DataFrame& df, const bool min_max_scale);

double mi_general(VPTree& ztree,
                  DataFrame& df,
                  int k,
                  std::shared_ptr<arrow::DataType> datatype,
                  std::vector<bool>& is_discrete_column,
                  int tree_leafsize,
                  unsigned int seed);
double mi_pair(VPTree& ytree,
               DataFrame& df,
               int k,
               std::shared_ptr<arrow::DataType> datatype,
               std::vector<bool>& is_discrete_column,
               int tree_leafsize,
               unsigned int seed);

class MSKMutualInformation : public IndependenceTest {
public:
    MSKMutualInformation(DataFrame df,
                         int k,
                         unsigned int seed = std::random_device{}(),
                         int shuffle_neighbors = 5,
                         int samples = 1000,
                         bool min_max_scale = true,
                         int tree_leafsize = 16)
        : m_df(df),
          m_scaled_df(scale_data_min_max(df, min_max_scale)),
          m_datatype(),
          m_k(k),
          m_seed(seed),
          m_shuffle_neighbors(shuffle_neighbors),
          m_samples(samples),
          m_tree_leafsize(tree_leafsize) {
        m_datatype = m_scaled_df.same_type();
    }

    double pvalue(const std::string& x, const std::string& y) const override;
    double pvalue(const std::string& x, const std::string& y, const std::string& z) const override;
    double pvalue(const std::string& x, const std::string& y, const std::vector<std::string>& z) const override;

    double shuffled_pvalue(double original_mi,
                           DataFrame& x_df,
                           VPTree& ztree,
                           DataFrame& z_df,
                           DataFrame& shuffled_df,
                           std::vector<bool>& is_discrete_column) const;

    double mi(const std::string& x, const std::string& y) const;
    double mi(const std::string& x, const std::string& y, const std::string& z) const;
    double mi(const std::string& x, const std::string& y, const std::vector<std::string>& z) const;

    int num_variables() const override { return m_df->num_columns(); }

    std::vector<std::string> variable_names() const override { return m_df.column_names(); }

    const std::string& name(int i) const override { return m_df.name(i); }

    bool has_variables(const std::string& name) const override { return m_df.has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const override { return m_df.has_columns(cols); }

private:
    DataFrame m_df;
    DataFrame m_scaled_df;
    std::shared_ptr<arrow::DataType> m_datatype;
    int m_k;
    unsigned int m_seed;
    int m_shuffle_neighbors;
    int m_samples;
    int m_tree_leafsize;
};

template <typename CType, typename Random>
void shuffle_dataframe(const CType* original_x,
                       CType* shuffled_x,
                       const std::vector<size_t>& order,
                       std::vector<bool>& used,
                       MatrixXi& neighbors,
                       Random& rng) {
    for (int i = 0; i < neighbors.cols(); ++i) {
        auto begin = neighbors.data() + i * neighbors.rows();
        auto end = begin + neighbors.rows();
        std::shuffle(begin, end, rng);
    }

    std::uniform_real_distribution<float> tiebreaker(-0.5, 0.5);
    for (int i = 0; i < neighbors.cols(); ++i) {
        auto index = order[i];

        int neighbor_index = 0;
        for (int j = 0; j < neighbors.rows(); ++j) {
            neighbor_index = neighbors(j, index);
            if (!used[neighbor_index]) {
                break;
            }
        }
        if (used[neighbor_index]) {
            shuffled_x[index] = original_x[neighbor_index] + tiebreaker(rng);
        } else {
            shuffled_x[index] = original_x[neighbor_index];
            used[neighbor_index] = true;
        }
    }

    std::vector<size_t> sorted_indices(neighbors.cols());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

    IndexComparator comp(shuffled_x);
    std::sort(sorted_indices.begin(), sorted_indices.end(), comp);

    for (size_t i = 0; i < sorted_indices.size(); ++i) {
        shuffled_x[sorted_indices[i]] = static_cast<float>(i);
    }
}



// using DynamicMSKMutualInformation = DynamicIndependenceTestAdaptator<MSKMutualInformation>;

}  // namespace learning::independences::hybrid

#endif  // PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_MUTUAL_INFORMATION_HPP