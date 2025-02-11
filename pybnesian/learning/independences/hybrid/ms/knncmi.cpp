#include <learning/independences/hybrid/ms/knncmi.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>

#include <iomanip>
using Array_ptr = std::shared_ptr<arrow::Array>;

namespace learning::independences::hybrid {

template <typename ArrowType>
DataFrame scale_data_min_max(const DataFrame& df) {
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
                    for (int i = 0; i < df->num_rows(); ++i) {
                        auto normalized_value = (column_cast->Value(i) - min) / (max - min);
                        RAISE_STATUS_ERROR(builder.Append(normalized_value));
                    }

                }

                else {
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

DataFrame scale_data_min_max(const DataFrame& df) {
    // Check continuous columns dtype
    switch (df.loc(df.continuous_columns()).same_type()->id()) {
        case Type::DOUBLE:
            return scale_data_min_max<arrow::DoubleType>(df);
        case Type::FLOAT:
            return scale_data_min_max<arrow::FloatType>(df);
        default:
            throw std::invalid_argument("Wrong data type in MSKMutualInformation.");
    }
}

double mi_general(DataFrame& df,
                  int k,
                  std::shared_ptr<arrow::DataType> datatype,
                  std::vector<bool>& is_discrete_column) {
    auto n_rows = df->num_rows();
    VPTree vptree(df, datatype, is_discrete_column);
    auto knn_results = vptree.query(df, k + 1);  // excluding the reference point which is not a neighbor of itself

    VectorXd eps(n_rows);
    VectorXi k_hat(n_rows);
    for (auto i = 0; i < n_rows; ++i) {
        eps(i) = knn_results[i].first(k);
        k_hat(i) = knn_results[i].second.size();
    }

    std::vector<size_t> indices(df->num_columns() - 2);
    std::iota(indices.begin(), indices.end(), 2);
    auto z_df = df.loc(indices);
    auto z_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 2, is_discrete_column.end());
    VPTree ztree(z_df, datatype, z_is_discrete_column);

    auto [n_xz, n_yz, n_z] = ztree.count_ball_subspaces(df, eps, is_discrete_column);

    double res = 0;
    for (int i = 0; i < n_rows; ++i) {
        res += boost::math::digamma(k_hat(i) - 1) + boost::math::digamma(n_z(i) - 1) -
               boost::math::digamma(n_xz(i) - 1) - boost::math::digamma(n_yz(i) - 1);
    }

    res /= n_rows;

    return res;

    // auto n_rows = df->num_rows();

    // // Timer for VP-tree construction
    // auto start_vptree_construction = std::chrono::high_resolution_clock::now();
    // VPTree vptree(df, datatype, is_discrete_column);
    // auto end_vptree_construction = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> vptree_construction_time = end_vptree_construction - start_vptree_construction;
    // std::cout << "VP-tree construction time: " << vptree_construction_time.count() << " seconds\n";

    // // Timer for query function
    // auto start_query = std::chrono::high_resolution_clock::now();
    // auto knn_results = vptree.query(df, k + 1); // excluding the reference point which is not a neighbor of itself
    // auto end_query = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> query_time = end_query - start_query;
    // std::cout << "Query time: " << query_time.count() << " seconds\n";

    // VectorXd eps(n_rows);
    // VectorXi k_hat(n_rows);
    // for (auto i = 0; i < n_rows; ++i) {
    //     eps(i) = knn_results[i].first(k);
    //     k_hat(i) = knn_results[i].second.size();
    // }

    // std::vector<size_t> indices(df->num_columns() - 2);
    // std::iota(indices.begin(), indices.end(), 2);
    // auto z_df = df.loc(indices);
    // auto z_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 2, is_discrete_column.end());

    // // Timer for Z VP-tree construction
    // auto start_ztree_construction = std::chrono::high_resolution_clock::now();
    // VPTree ztree(z_df, datatype, z_is_discrete_column);
    // auto end_ztree_construction = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> ztree_construction_time = end_ztree_construction - start_ztree_construction;
    // std::cout << "Z VP-tree construction time: " << ztree_construction_time.count() << " seconds\n";

    // // Timer for count_ball function
    // auto start_count_ball = std::chrono::high_resolution_clock::now();
    // auto [n_xz, n_yz, n_z] = ztree.count_ball_subspaces(df, eps, is_discrete_column);
    // auto end_count_ball = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> count_ball_time = end_count_ball - start_count_ball;
    // std::cout << "Count ball time: " << count_ball_time.count() << " seconds\n";

    // double res = 0;
    // for (int i = 0; i < n_rows; ++i) {
    //     res += boost::math::digamma(k_hat(i) - 1) + boost::math::digamma(n_z(i) - 1) - boost::math::digamma(n_xz(i) -
    //     1) - boost::math::digamma(n_yz(i) - 1);
    // }

    // res /= n_rows;

    // return res;
}

double mi_pair(DataFrame& df, int k, std::shared_ptr<arrow::DataType> datatype, std::vector<bool>& is_discrete_column) {
    auto n_rows = df->num_rows();
    VPTree xytree(df, datatype, is_discrete_column);
    auto knn_results = xytree.query(df, k + 1);  // excluding the reference point which is not a neighbor of itself

    VectorXd eps(n_rows);
    VectorXi k_hat(n_rows);
    for (auto i = 0; i < n_rows; ++i) {
        eps(i) = knn_results[i].first(k);
        k_hat(i) = knn_results[i].second.size();
    }

    auto x_df = df.loc(0);
    auto y_df = df.loc(1);
    auto x_is_discrete_column = std::vector<bool>(is_discrete_column.begin(), is_discrete_column.begin() + 1);
    auto y_is_discrete_column = std::vector<bool>(is_discrete_column.begin() + 1, is_discrete_column.end());
    VPTree xtree(x_df, datatype, x_is_discrete_column);
    VPTree ytree(y_df, datatype, y_is_discrete_column);

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

// double MSKMutualInformation::mi(const std::string& x, const std::string& y) const {
//     const std::string z = "const_col";
//     auto subset_df = m_scaled_df.loc(x, y, z);
//     std::vector<bool> is_discrete_column;
//     is_discrete_column.push_back(m_df.is_discrete(x));
//     is_discrete_column.push_back(m_df.is_discrete(y));
//     is_discrete_column.push_back(false);
//     return mi_general(subset_df, m_k, m_datatype, is_discrete_column);
// }

double MSKMutualInformation::mi(const std::string& x, const std::string& y) const {
    auto subset_df = m_scaled_df.loc(x, y);
    std::vector<bool> is_discrete_column;
    is_discrete_column.push_back(m_df.is_discrete(x));
    is_discrete_column.push_back(m_df.is_discrete(y));

    return mi_pair(subset_df, m_k, m_datatype, is_discrete_column);
}

double MSKMutualInformation::mi(const std::string& x, const std::string& y, const std::string& z) const {
    auto subset_df = m_scaled_df.loc(x, y, z);
    std::vector<bool> is_discrete_column;
    is_discrete_column.push_back(m_df.is_discrete(x));
    is_discrete_column.push_back(m_df.is_discrete(y));
    is_discrete_column.push_back(m_df.is_discrete(z));
    return mi_general(subset_df, m_k, m_datatype, is_discrete_column);
}

double MSKMutualInformation::mi(const std::string& x, const std::string& y, const std::vector<std::string>& z) const {
    auto subset_df = m_scaled_df.loc(x, y, z);
    std::vector<bool> is_discrete_column;
    is_discrete_column.push_back(m_df.is_discrete(x));
    is_discrete_column.push_back(m_df.is_discrete(y));
    for (auto col_name : z) {
        is_discrete_column.push_back(m_df.is_discrete(col_name));
    }
    return mi_general(subset_df, m_k, m_datatype, is_discrete_column);
}

double MSKMutualInformation::pvalue(const std::string& x, const std::string& y) const {
    // auto value = mi(x, y);

    // auto shuffled_df = m_scaled_df.loc(Copy(x), y);

    // auto x_begin = shuffled_df.template mutable_data<arrow::FloatType>(0);
    // auto x_end = x_begin + shuffled_df->num_rows();
    // std::mt19937 rng{m_seed};

    // int count_greater = 0;
    // for (int i = 0; i < m_samples; ++i) {
    //     std::shuffle(x_begin, x_end, rng);
    //     auto shuffled_value = mi_pair(shuffled_df, m_k);

    //     if (shuffled_value >= value) ++count_greater;
    // }

    // return static_cast<double>(count_greater) / m_samples;
    return 1.0;
}

double MSKMutualInformation::pvalue(const std::string& x, const std::string& y, const std::string& z) const {
    // auto original_mi = mi(x, y, z);
    // auto z_df = m_df.loc(z);
    // auto shuffled_df = m_scaled_df.loc(Copy(x), y, z);
    // auto original_rank_x = m_scaled_df.template data<arrow::FloatType>(x);

    // return shuffled_pvalue(original_mi, original_rank_x, z_df, shuffled_df, MITriple{});
    return 1.0;
}

double MSKMutualInformation::pvalue(const std::string& x,
                                    const std::string& y,
                                    const std::vector<std::string>& z) const {
    // auto original_mi = mi(x, y, z);
    // auto z_df = m_df.loc(z);
    // auto shuffled_df = m_scaled_df.loc(Copy(x), y, z);
    // auto original_rank_x = m_scaled_df.template data<arrow::FloatType>(x);

    // return shuffled_pvalue(original_mi, original_rank_x, z_df, shuffled_df, MIGeneral{});
    return 1.0;
}

}  // namespace learning::independences::hybrid