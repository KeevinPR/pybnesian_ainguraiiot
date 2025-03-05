#include <learning/independences/hybrid/ms/knncmi.hpp>
#include <factors/discrete/discrete_indices.hpp>
#include <kdtree/kdtree.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <algorithm>

using Array_ptr = std::shared_ptr<arrow::Array>;

namespace learning::independences::hybrid {


template <typename ArrowType>
DataFrame scale_data(const DataFrame& df, const std::string& scaling) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using CType = typename ArrowType::c_type;
    using kdtree::IndexComparator;

    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);
    std::vector<Array_ptr> new_columns;

    arrow::NumericBuilder<ArrowType> builder;
    auto n_rows = df->num_rows();

    std::vector<size_t> indices(n_rows);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<CType> ranked_data(n_rows);

    for (int j = 0; j < df->num_columns(); ++j) {
        auto column = df.col(j);
        auto dt = column->type_id();
        switch (dt) {
            case Type::DICTIONARY: {
                auto column_cast = std::static_pointer_cast<arrow::DictionaryArray>(column);
                auto indices = std::static_pointer_cast<arrow::Int8Array>(column_cast->indices());
                for (int i = 0; i < n_rows; ++i) {
                    RAISE_STATUS_ERROR(builder.Append(static_cast<CType>(indices->Value(i))));
                }
                break;
            }
            // min-max transform only the continuous variables
            default: {
                
                if (scaling == "normalized_rank") {
                    auto dwn = df.downcast<ArrowType>(j);
                    auto raw_values = dwn->raw_values();

                    IndexComparator comp(raw_values);
                    std::sort(indices.begin(), indices.end(), comp);
                    
                    for (int i = 0; i < n_rows; ++i) {
                        ranked_data[indices[i]] = static_cast<CType>(i)/static_cast<CType>(n_rows - 1);
                    }

                    RAISE_STATUS_ERROR(builder.AppendValues(ranked_data.begin(), ranked_data.end()));

                } else if (scaling == "min_max") {
                    auto column_cast = std::static_pointer_cast<ArrayType>(column);
                    auto min = df.min<ArrowType>(j);
                    auto max = df.max<ArrowType>(j);
                    if (max != min) {
                        for (int i = 0; i < n_rows; ++i) {
                            auto normalized_value = (column_cast->Value(i) - min) / (max - min);
                            RAISE_STATUS_ERROR(builder.Append(normalized_value));
                        }
                    } else {
                        throw std::invalid_argument("Constant column in DataFrame.");
                    }

                } else {
                    throw std::invalid_argument("Invalid scaling option, must be normalized_rank or min_max.");
                }
            }
        }
        Array_ptr out;
        RAISE_STATUS_ERROR(builder.Finish(&out));
        new_columns.push_back(out);
        builder.Reset();

        auto f = arrow::field(df.name(j), out->type());
        RAISE_STATUS_ERROR(b.AddField(f));
    }

    RAISE_RESULT_ERROR(auto schema, b.Finish())

    auto rb = arrow::RecordBatch::Make(schema, n_rows, new_columns);
    return DataFrame(rb);
}

DataFrame scale_data(const DataFrame& df, const std::string& scaling) {
    // Check continuous columns dtype
    auto cont_cols = df.continuous_columns();
    std::shared_ptr<arrow::DataType> dt;
    if (cont_cols.size() > 0) {
        dt = df.loc(cont_cols).same_type();
    } else {
        dt = std::static_pointer_cast<arrow::FloatType>(arrow::float32());
    }
    switch (dt->id()) {
        case Type::DOUBLE:
            return scale_data<arrow::DoubleType>(df, scaling);
        case Type::FLOAT:
            return scale_data<arrow::FloatType>(df, scaling);
        default:
            throw std::invalid_argument("Wrong data type in MSKMutualInformation.");
    }
}

double mi_general(VPTree& ztree,
                  DataFrame& df,
                  int k,
                  std::shared_ptr<arrow::DataType> datatype,
                  std::vector<bool>& is_discrete_column,
                  int tree_leafsize,
                  unsigned int seed) {
    auto n_rows = df->num_rows();
    VPTree vptree(df, datatype, is_discrete_column, tree_leafsize, seed);
    auto knn_results = vptree.query(df, k + 1);  // excluding the reference point which is not a neighbor of itself

    VectorXd eps(n_rows);
    VectorXi k_hat(n_rows);
    for (auto i = 0; i < n_rows; ++i) {
        eps(i) = knn_results[i].first(k);
        k_hat(i) = knn_results[i].second.size();
    }

    auto [n_xz, n_yz, n_z] = ztree.count_ball_subspaces(df, eps, is_discrete_column);

    double res = 0;
    // for (int i = 0; i < n_rows; ++i) {
    //     res += boost::math::digamma(k_hat(i) - 1) + boost::math::digamma(n_z(i) - 1) -
    //            boost::math::digamma(n_xz(i) - 1) - boost::math::digamma(n_yz(i) - 1);
    // }

    auto log_or_digamma = [](int value_hat, int k) {
        return (value_hat > k) ? std::log(value_hat) : boost::math::digamma(value_hat);
    };
    for (int i = 0; i < n_rows; ++i) {
        res += log_or_digamma(k_hat(i) - 1, k) + log_or_digamma(n_z(i) - 1, k) - log_or_digamma(n_xz(i) - 1, k) -
               log_or_digamma(n_yz(i) - 1, k);
    }

    res /= n_rows;

    return res;
}

double mi_pair(VPTree& ytree,
               DataFrame& df,
               int k,
               std::shared_ptr<arrow::DataType> datatype,
               std::vector<bool>& is_discrete_column,
               int tree_leafsize,
               unsigned int seed) {
    auto n_rows = df->num_rows();
    VPTree xytree(df, datatype, is_discrete_column, tree_leafsize, seed);
    auto knn_results = xytree.query(df, k + 1);  // excluding the reference point which is not a neighbor of itself

    VectorXd eps(n_rows);
    VectorXi k_hat(n_rows);
    for (auto i = 0; i < n_rows; ++i) {
        eps(i) = knn_results[i].first[k];
        k_hat(i) = knn_results[i].second.size();
    }

    auto x_is_discrete_column = std::vector<bool>(is_discrete_column.begin(), is_discrete_column.begin() + 1);
    auto y_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 1, is_discrete_column.end());

    auto x_df = df.loc(0);
    auto y_df = df.loc(1);

    VPTree xtree(x_df, datatype, x_is_discrete_column, tree_leafsize, seed);

    auto n_x = xtree.count_ball_unconditional(x_df, eps, x_is_discrete_column);
    auto n_y = ytree.count_ball_unconditional(y_df, eps, y_is_discrete_column);

    double res = 0;

    // for (int i = 0; i < n_rows; ++i) {
    //     res += boost::math::digamma(k_hat(i) - 1) + boost::math::digamma(n_rows - 1) -
    //            boost::math::digamma(n_x(i) - 1) - boost::math::digamma(n_y(i) - 1);
    // }

    auto log_or_digamma = [](int value_hat, int k) {
        return (value_hat > k) ? std::log(value_hat) : boost::math::digamma(value_hat);
    };
    for (int i = 0; i < n_rows; ++i) {
        res += log_or_digamma(k_hat(i) - 1, k) + std::log(n_rows - 1) - log_or_digamma(n_x(i) - 1, k) -
               log_or_digamma(n_y(i) - 1, k);
    }

    res /= n_rows;

    return res;
}

int MSKMutualInformation::find_minimum_cluster_size(const std::vector<std::string>& discrete_vars) {
    auto dummy_vars = std::vector<std::string>(discrete_vars.begin() + 1, discrete_vars.end());

    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, discrete_vars);

    auto joint_counts = factors::discrete::joint_counts(m_df, discrete_vars[0], dummy_vars, cardinality, strides);

    int min_cluster_size = std::numeric_limits<int>::max();

    // find minimum positive cluster size
    for (int i = 0; i < joint_counts.size(); ++i) {
        if (joint_counts[i] > 0 && joint_counts[i] < min_cluster_size) {
            min_cluster_size = joint_counts[i];
        }
    }

    return min_cluster_size;
}

std::vector<std::string> check_discrete_cols(const DataFrame& df,
                                             std::vector<bool>& is_discrete_column,
                                             bool& discrete_present,
                                             const std::string& x,
                                             const std::string& y) {
    is_discrete_column.push_back(df.is_discrete(x));
    is_discrete_column.push_back(df.is_discrete(y));

    std::vector<std::string> discrete_vars;
    if (is_discrete_column[0]) {
        discrete_vars.push_back(x);
        discrete_present = true;
    }
    if (is_discrete_column[1]) {
        discrete_vars.push_back(y);
        discrete_present = true;
    }
    return discrete_vars;
}

std::vector<std::string> check_discrete_cols(const DataFrame& df,
                                             std::vector<bool>& is_discrete_column,
                                             bool& discrete_present,
                                             const std::string& x,
                                             const std::string& y,
                                             const std::string& z) {
    auto discrete_vars = check_discrete_cols(df, is_discrete_column, discrete_present, x, y);
    is_discrete_column.push_back(df.is_discrete(z));

    if (is_discrete_column.back()) {
        discrete_vars.push_back(z);
        discrete_present = true;
    }

    return discrete_vars;
}

std::vector<std::string> check_discrete_cols(const DataFrame& df,
                                             std::vector<bool>& is_discrete_column,
                                             bool& discrete_present,
                                             const std::string& x,
                                             const std::string& y,
                                             const std::vector<std::string>& z) {
    auto discrete_vars = check_discrete_cols(df, is_discrete_column, discrete_present, x, y);
    for (const auto& col : z) {
        is_discrete_column.push_back(df.is_discrete(col));
        if (is_discrete_column.back()) {
            discrete_vars.push_back(col);
            discrete_present = true;
        }
    }

    return discrete_vars;
}

double MSKMutualInformation::mi(const std::string& x, const std::string& y) const {
    auto subset_df = m_scaled_df.loc(x, y);
    std::vector<bool> is_discrete_column;
    bool discrete_present = false;
    int k = m_k;
    auto discrete_vars = check_discrete_cols(m_df, is_discrete_column, discrete_present, x, y);

    // if (discrete_present) {
    //     auto min_cluster_size = find_minimum_cluster_size(discrete_vars);
    //     k = std::min(k, min_cluster_size);
    // }

    auto y_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 1, is_discrete_column.end());
    auto y_df = subset_df.loc(1);
    VPTree ytree(y_df, m_datatype, y_is_discrete_column, m_tree_leafsize, m_seed);

    return mi_pair(ytree, subset_df, k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
}

double MSKMutualInformation::mi(const std::string& x, const std::string& y, const std::string& z) const {
    auto subset_df = m_scaled_df.loc(x, y, z);
    std::vector<bool> is_discrete_column;
    bool discrete_present = false;
    int k = m_k;
    auto discrete_vars = check_discrete_cols(m_df, is_discrete_column, discrete_present, x, y, z);

    // if (discrete_present) {
    //     auto min_cluster_size = find_minimum_cluster_size(discrete_vars);
    //     k = std::min(k, min_cluster_size);
    // }

    auto z_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 2, is_discrete_column.end());
    auto z_df = subset_df.loc(2);
    VPTree ztree(z_df, m_datatype, z_is_discrete_column, m_tree_leafsize, m_seed);

    return mi_general(ztree, subset_df, k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
}

double MSKMutualInformation::mi(const std::string& x, const std::string& y, const std::vector<std::string>& z) const {
    auto subset_df = m_scaled_df.loc(x, y, z);
    std::vector<bool> is_discrete_column;
    bool discrete_present = false;
    int k = m_k;
    auto discrete_vars = check_discrete_cols(m_df, is_discrete_column, discrete_present, x, y, z);

    // if (discrete_present) {
    //     auto min_cluster_size = find_minimum_cluster_size(discrete_vars);
    //     k = std::min(k, min_cluster_size);
    // }

    auto z_df = m_scaled_df.loc(z);
    auto z_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 2, is_discrete_column.end());

    VPTree ztree(z_df, m_datatype, z_is_discrete_column, m_tree_leafsize, m_seed);
    return mi_general(ztree, subset_df, k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
}

double compute_mean(const std::vector<double>& data) {
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double compute_variance(const std::vector<double>& data, double mean) {
    double variance = 0.0;
    for (double x : data) {
        variance += std::pow((x - mean), 2);
    }
    return variance / data.size();
}

double compute_skewness(const std::vector<double>& data, double mean, double variance) {
    double skewness = 0.0;
    for (double x : data) {
        skewness += std::pow(x - mean, 3);
    }
    return (skewness / data.size()) / std::pow(variance, 1.5);
}

double compute_gamma_approximation(double original_mi, std::vector<double>& permutation_stats) {
    double min_value = *std::min_element(permutation_stats.begin(), permutation_stats.end());
    double max_value = *std::max_element(permutation_stats.begin(), permutation_stats.end());

    if (original_mi > max_value) {
        return 0.0;
    } else if (original_mi <= min_value) {
        return 1.0;
    }

    double epsilon = std::numeric_limits<double>::epsilon();  // Small positive value to ensure positivity
    std::vector<double> shifted_data;
    for (auto i = 0; i < permutation_stats.size(); ++i) {
        permutation_stats[i] = permutation_stats[i] - min_value + epsilon;
    }

    double mean = compute_mean(permutation_stats);
    double variance = compute_variance(permutation_stats, mean);
    double skewness = compute_skewness(permutation_stats, mean, variance);

    double shape, scale;
    shape = (mean * mean) / variance;
    scale = variance / mean;

    boost::math::gamma_distribution<> gamma_dist(shape, scale);

    // Use the fitted gamma distribution to compute the p-value
    if (skewness > 0) {
        return 1 - boost::math::cdf(gamma_dist, original_mi - min_value + epsilon);
    }

    return boost::math::cdf(gamma_dist, original_mi - min_value + epsilon);

    // int count_greater = 0;

    // for (int i = 0; i < permutation_stats.size(); ++i) {

    //     if (permutation_stats[i] >= original_mi - min_value + epsilon) ++count_greater;
    // }

    // return static_cast<double>(count_greater) / permutation_stats.size();
}

double MSKMutualInformation::pvalue(const std::string& x, const std::string& y) const {
    std::mt19937 rng{m_seed};
    std::vector<bool> is_discrete_column;
    bool discrete_present = false;
    int k = m_k;
    auto discrete_vars = check_discrete_cols(m_df, is_discrete_column, discrete_present, x, y);

    if (discrete_present) {
        auto min_cluster_size = find_minimum_cluster_size(discrete_vars);
        k = std::min(k, min_cluster_size);
    }

    auto y_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 1, is_discrete_column.end());
    auto shuffled_df = m_scaled_df.loc(Copy(x), y);
    auto y_df = shuffled_df.loc(1);
    VPTree ytree(y_df, m_datatype, y_is_discrete_column, m_tree_leafsize, m_seed);

    auto original_mi = mi_pair(ytree, shuffled_df, k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
    std::vector<double> permutation_stats(m_samples);

    switch (m_datatype->id()) {
        case Type::FLOAT: {
            auto x_begin = shuffled_df.template mutable_data<arrow::FloatType>(0);
            auto x_end = x_begin + shuffled_df->num_rows();

            for (int i = 0; i < m_samples; ++i) {
                std::shuffle(x_begin, x_end, rng);
                auto shuffled_value =
                    mi_pair(ytree, shuffled_df, k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
                permutation_stats[i] = shuffled_value;
            }
            break;
        }

        default: {
            auto x_begin = shuffled_df.template mutable_data<arrow::DoubleType>(0);
            auto x_end = x_begin + shuffled_df->num_rows();

            for (int i = 0; i < m_samples; ++i) {
                std::shuffle(x_begin, x_end, rng);
                auto shuffled_value =
                    mi_pair(ytree, shuffled_df, k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
                permutation_stats[i] = shuffled_value;
            }
        }
    }

    return compute_gamma_approximation(original_mi, permutation_stats);
}

double MSKMutualInformation::pvalue(const std::string& x, const std::string& y, const std::string& z) const {
    auto subset_df = m_scaled_df.loc(x, y, z);
    std::vector<bool> is_discrete_column;
    bool discrete_present = false;
    int k = m_k;
    auto discrete_vars = check_discrete_cols(m_df, is_discrete_column, discrete_present, x, y, z);

    if (discrete_present) {
        auto min_cluster_size = find_minimum_cluster_size(discrete_vars);
        k = std::min(k, min_cluster_size);
    }

    auto x_df = subset_df.loc(0);

    auto z_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 2, is_discrete_column.end());
    auto shuffled_df = m_scaled_df.loc(Copy(x), y, z);
    auto z_df = shuffled_df.loc(2);
    VPTree ztree(z_df, m_datatype, z_is_discrete_column, m_tree_leafsize, m_seed);

    auto original_mi = mi_general(ztree, shuffled_df, k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);

    return shuffled_pvalue(original_mi, k, x_df, ztree, z_df, shuffled_df, is_discrete_column);
}

double MSKMutualInformation::pvalue(const std::string& x,
                                    const std::string& y,
                                    const std::vector<std::string>& z) const {
    auto subset_df = m_scaled_df.loc(x, y, z);
    std::vector<bool> is_discrete_column;
    bool discrete_present = false;
    int k = m_k;
    auto discrete_vars = check_discrete_cols(m_df, is_discrete_column, discrete_present, x, y, z);

    if (discrete_present) {
        auto min_cluster_size = find_minimum_cluster_size(discrete_vars);
        k = std::min(k, min_cluster_size);
    }

    auto x_df = subset_df.loc(0);

    auto z_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 2, is_discrete_column.end());
    auto shuffled_df = m_scaled_df.loc(Copy(x), y, z);
    auto z_df = shuffled_df.loc(z);
    VPTree ztree(z_df, m_datatype, z_is_discrete_column, m_tree_leafsize, m_seed);

    auto original_mi = mi_general(ztree, shuffled_df, k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);

    return shuffled_pvalue(original_mi, k, x_df, ztree, z_df, shuffled_df, is_discrete_column);
}

template <typename CType, typename Random>
void shuffle_dataframe(const CType* original_x,
                       CType* shuffled_x,
                       const std::vector<size_t>& order,
                       std::vector<bool>& used,
                       std::vector<VectorXi>& neighbors,
                       Random& rng) {
    for (auto& neighbor_list : neighbors) {
        auto begin = neighbor_list.data();
        auto end = begin + neighbor_list.size();
        std::shuffle(begin, end, rng);
    }

    for (long unsigned int i = 0; i < order.size(); ++i) {
        size_t index = order[i];
        int neighbor_index = 0;

        for (auto j = 0; j < neighbors[index].size(); ++j) {
            neighbor_index = neighbors[index][j];
            if (!used[neighbor_index]) {
                break;
            }
        }

        shuffled_x[index] = original_x[neighbor_index];
        used[neighbor_index] = true;
    }
}

double MSKMutualInformation::shuffled_pvalue(double original_mi,
                                             int k,
                                             DataFrame& x_df,
                                             VPTree& ztree,
                                             DataFrame& z_df,
                                             DataFrame& shuffled_df,
                                             std::vector<bool>& is_discrete_column) const {
    std::minstd_rand rng{m_seed};
    std::vector<VectorXi> neighbors(m_df->num_rows());

    auto zknn = ztree.query(z_df, m_shuffle_neighbors);

    for (size_t i = 0; i < zknn.size(); ++i) {
        neighbors[i] = zknn[i].second;
    }

    std::vector<size_t> order(m_df->num_rows());
    std::iota(order.begin(), order.end(), 0);

    std::vector<bool> used(m_df->num_rows(), false);
    std::vector<double> permutation_stats(m_samples);

    switch (m_datatype->id()) {
        case Type::FLOAT: {
            auto original_x = x_df.template data<arrow::FloatType>(0);
            auto shuffled_x = shuffled_df.template mutable_data<arrow::FloatType>(0);

            for (int i = 0; i < m_samples; ++i) {
                std::shuffle(order.begin(), order.end(), rng);
                shuffle_dataframe(original_x, shuffled_x, order, used, neighbors, rng);

                auto shuffled_value =
                    mi_general(ztree, shuffled_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);

                permutation_stats[i] = shuffled_value;

                std::fill(used.begin(), used.end(), false);
            }
            break;
        }

        default: {
            auto original_x = x_df.template data<arrow::DoubleType>(0);
            auto shuffled_x = shuffled_df.template mutable_data<arrow::DoubleType>(0);

            for (int i = 0; i < m_samples; ++i) {
                std::shuffle(order.begin(), order.end(), rng);
                shuffle_dataframe(original_x, shuffled_x, order, used, neighbors, rng);

                auto shuffled_value =
                    mi_general(ztree, shuffled_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);

                permutation_stats[i] = shuffled_value;

                std::fill(used.begin(), used.end(), false);
            }
        }
    }

    return compute_gamma_approximation(original_mi, permutation_stats);
}

}  // namespace learning::independences::hybrid