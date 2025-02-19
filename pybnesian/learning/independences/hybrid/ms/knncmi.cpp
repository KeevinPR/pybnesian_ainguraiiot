#include <learning/independences/hybrid/ms/knncmi.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <algorithm>

using Array_ptr = std::shared_ptr<arrow::Array>;

namespace learning::independences::hybrid {

template <typename ArrowType>
DataFrame scale_data_min_max(const DataFrame& df, const bool min_max_scale) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using CType = typename ArrowType::c_type;
    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);
    std::vector<Array_ptr> new_columns;

    arrow::NumericBuilder<ArrowType> builder;

    for (int j = 0; j < df->num_columns(); ++j) {
        auto column = df.col(j);
        auto dt = column->type_id();
        switch (dt) {
            case Type::DICTIONARY: {
                auto column_cast = std::static_pointer_cast<arrow::DictionaryArray>(column);
                auto indices = std::static_pointer_cast<arrow::Int8Array>(column_cast->indices());
                for (int i = 0; i < df->num_rows(); ++i) {
                    RAISE_STATUS_ERROR(builder.Append(static_cast<CType>(indices->Value(i))));
                }
                break;
            }
            // min-max transform only the continuous variables
            default: {
                auto min = df.min<ArrowType>(j);
                auto max = df.max<ArrowType>(j);
                if (max != min) {
                    auto column_cast = std::static_pointer_cast<ArrayType>(column);
                    if (min_max_scale) {
                        for (int i = 0; i < df->num_rows(); ++i) {
                            auto normalized_value = (column_cast->Value(i) - min) / (max - min);
                            RAISE_STATUS_ERROR(builder.Append(normalized_value));
                        }
                    } else {
                        for (int i = 0; i < df->num_rows(); ++i) {
                            auto value = column_cast->Value(i);
                            RAISE_STATUS_ERROR(builder.Append(value));
                        }
                    }
                } else {
                    throw std::invalid_argument("Constant column in DataFrame.");
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

    auto rb = arrow::RecordBatch::Make(schema, df->num_rows(), new_columns);
    return DataFrame(rb);
}

DataFrame scale_data_min_max(const DataFrame& df, const bool min_max_scale) {
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
            return scale_data_min_max<arrow::DoubleType>(df, min_max_scale);
        case Type::FLOAT:
            return scale_data_min_max<arrow::FloatType>(df, min_max_scale);
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
    for (int i = 0; i < n_rows; ++i) {
        res += boost::math::digamma(k_hat(i) - 1) + boost::math::digamma(n_z(i) - 1) -
               boost::math::digamma(n_xz(i) - 1) - boost::math::digamma(n_yz(i) - 1);
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
        eps(i) = knn_results[i].first(k);
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
    for (int i = 0; i < n_rows; ++i) {
        res += boost::math::digamma(k_hat(i) - 1) + boost::math::digamma(n_rows - 1) -
               boost::math::digamma(n_x(i) - 1) - boost::math::digamma(n_y(i) - 1);
    }

    res /= n_rows;

    return res;
}

double MSKMutualInformation::mi(const std::string& x, const std::string& y) const {
    auto subset_df = m_scaled_df.loc(x, y);
    std::vector<bool> is_discrete_column;
    is_discrete_column.push_back(m_df.is_discrete(x));
    is_discrete_column.push_back(m_df.is_discrete(y));

    auto y_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 1, is_discrete_column.end());
    auto y_df = subset_df.loc(1);
    VPTree ytree(y_df, m_datatype, y_is_discrete_column, m_tree_leafsize, m_seed);

    return mi_pair(ytree, subset_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
}

double MSKMutualInformation::mi(const std::string& x, const std::string& y, const std::string& z) const {
    auto subset_df = m_scaled_df.loc(x, y, z);
    std::vector<bool> is_discrete_column;
    is_discrete_column.push_back(m_df.is_discrete(x));
    is_discrete_column.push_back(m_df.is_discrete(y));
    is_discrete_column.push_back(m_df.is_discrete(z));

    auto z_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 2, is_discrete_column.end());
    auto z_df = subset_df.loc(2);
    VPTree ztree(z_df, m_datatype, z_is_discrete_column, m_tree_leafsize, m_seed);

    return mi_general(ztree, subset_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
}

double MSKMutualInformation::mi(const std::string& x, const std::string& y, const std::vector<std::string>& z) const {
    auto subset_df = m_scaled_df.loc(x, y, z);
    std::vector<bool> is_discrete_column;
    is_discrete_column.push_back(m_df.is_discrete(x));
    is_discrete_column.push_back(m_df.is_discrete(y));
    for (auto col_name : z) {
        is_discrete_column.push_back(m_df.is_discrete(col_name));
    }

    auto z_df = m_scaled_df.loc(z);
    auto z_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 2, is_discrete_column.end());

    VPTree ztree(z_df, m_datatype, z_is_discrete_column, m_tree_leafsize, m_seed);
    return mi_general(ztree, subset_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
}

double MSKMutualInformation::pvalue(const std::string& x, const std::string& y) const {
    std::mt19937 rng{m_seed};
    std::vector<bool> is_discrete_column;
    is_discrete_column.push_back(m_df.is_discrete(x));
    is_discrete_column.push_back(m_df.is_discrete(y));

    auto y_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 1, is_discrete_column.end());
    auto shuffled_df = m_scaled_df.loc(Copy(x), y);
    auto y_df = shuffled_df.loc(1);
    VPTree ytree(y_df, m_datatype, y_is_discrete_column, m_tree_leafsize, m_seed);

    auto value = mi_pair(ytree, shuffled_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);

    int count_greater = 0;

    switch (m_datatype->id()) {
        case Type::FLOAT: {
            auto x_begin = shuffled_df.template mutable_data<arrow::FloatType>(0);
            auto x_end = x_begin + shuffled_df->num_rows();

            for (int i = 0; i < m_samples; ++i) {
                std::shuffle(x_begin, x_end, rng);
                auto shuffled_value =
                    mi_pair(ytree, shuffled_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
                if (shuffled_value >= value) ++count_greater;
            }
            break;
        }

        default: {
            auto x_begin = shuffled_df.template mutable_data<arrow::DoubleType>(0);
            auto x_end = x_begin + shuffled_df->num_rows();

            for (int i = 0; i < m_samples; ++i) {
                std::shuffle(x_begin, x_end, rng);
                auto shuffled_value =
                    mi_pair(ytree, shuffled_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);
                if (shuffled_value >= value) ++count_greater;
            }
        }
    }

    return static_cast<double>(count_greater) / m_samples;
}

double MSKMutualInformation::pvalue(const std::string& x, const std::string& y, const std::string& z) const {
    auto subset_df = m_scaled_df.loc(x, y, z);
    std::vector<bool> is_discrete_column;
    is_discrete_column.push_back(m_df.is_discrete(x));
    is_discrete_column.push_back(m_df.is_discrete(y));
    is_discrete_column.push_back(m_df.is_discrete(z));

    auto x_df = subset_df.loc(0);

    auto z_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 2, is_discrete_column.end());
    auto shuffled_df = m_scaled_df.loc(Copy(x), y, z);
    auto z_df = shuffled_df.loc(2);
    VPTree ztree(z_df, m_datatype, z_is_discrete_column, m_tree_leafsize, m_seed);

    auto original_mi = mi_general(ztree, shuffled_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);

    return shuffled_pvalue(original_mi, x_df, ztree, z_df, shuffled_df, is_discrete_column);
}

double MSKMutualInformation::pvalue(const std::string& x,
                                    const std::string& y,
                                    const std::vector<std::string>& z) const {
    auto subset_df = m_scaled_df.loc(x, y, z);
    std::vector<bool> is_discrete_column;
    is_discrete_column.push_back(m_df.is_discrete(x));
    is_discrete_column.push_back(m_df.is_discrete(y));
    for (auto col_name : z) {
        is_discrete_column.push_back(m_df.is_discrete(col_name));
    }

    auto x_df = subset_df.loc(0);

    auto z_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 2, is_discrete_column.end());
    auto shuffled_df = m_scaled_df.loc(Copy(x), y, z);
    auto z_df = shuffled_df.loc(z);
    VPTree ztree(z_df, m_datatype, z_is_discrete_column, m_tree_leafsize, m_seed);

    auto original_mi = mi_general(ztree, shuffled_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);

    return shuffled_pvalue(original_mi, x_df, ztree, z_df, shuffled_df, is_discrete_column);
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
        int neighbor_index = -1;
        for (auto j = 0; j < neighbors[index].size(); ++j) {
            neighbor_index = neighbors[index][j];
            if (!used[neighbor_index]) {
                break;
            }
        }

        if (neighbor_index == -1 || used[neighbor_index]) {
            shuffled_x[index] = original_x[index];
        } else {
            shuffled_x[index] = original_x[neighbor_index];
            used[neighbor_index] = true;
        }
    }
}

double MSKMutualInformation::shuffled_pvalue(double original_mi,
                                             DataFrame& x_df,
                                             VPTree& ztree,
                                             DataFrame& z_df,
                                             DataFrame& shuffled_df,
                                             std::vector<bool>& is_discrete_column) const {
    std::mt19937 rng{m_seed};
    std::vector<VectorXi> neighbors(m_df->num_rows());

    auto zknn = ztree.query(z_df, m_shuffle_neighbors);

    for (size_t i = 0; i < zknn.size(); ++i) {
        neighbors.push_back(zknn[i].second);
    }

    std::vector<size_t> order(m_df->num_rows());
    std::iota(order.begin(), order.end(), 0);

    std::vector<bool> used(m_df->num_rows(), false);
    int count_greater = 0;

    switch (m_datatype->id()) {
        case Type::FLOAT: {
            auto original_x = x_df.template data<arrow::FloatType>(0);
            auto shuffled_x = shuffled_df.template mutable_data<arrow::FloatType>(0);

            for (int i = 0; i < m_samples; ++i) {
                std::shuffle(order.begin(), order.end(), rng);
                shuffle_dataframe(original_x, shuffled_x, order, used, neighbors, rng);

                auto shuffled_value =
                    mi_general(ztree, shuffled_df, m_k, m_datatype, is_discrete_column, m_tree_leafsize, m_seed);

                if (shuffled_value >= original_mi) ++count_greater;

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

                if (shuffled_value >= original_mi) ++count_greater;

                std::fill(used.begin(), used.end(), false);
            }
        }
    }

    return static_cast<double>(count_greater) / m_samples;
}

}  // namespace learning::independences::hybrid