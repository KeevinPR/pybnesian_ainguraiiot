#include <learning/independences/hybrid/ms/knncmi.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <algorithm>

#include <iomanip>
using Array_ptr = std::shared_ptr<arrow::Array>;


namespace learning::independences::hybrid {

DataFrame scale_data_min_max(const DataFrame& df) {
    // Check continuous columns dtype
    df.continuous_columns();
    std::vector<Array_ptr> new_columns;

    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);

    for (int j = 0; j < df->num_columns(); ++j) {
        auto column = df.col(j);
        auto dt = column->type_id();
        switch (dt) {
            case Type::DICTIONARY:{
                new_columns.push_back(column);
                auto f = arrow::field(df.name(j), arrow::dictionary(arrow::int32(), arrow::utf8()));
                RAISE_STATUS_ERROR(b.AddField(f));
                break;}
            // min-max transform only the continuous variables
            case Type::DOUBLE: {
                using ArrayType = typename arrow::TypeTraits<arrow::DoubleType>::ArrayType;
                arrow::NumericBuilder<arrow::DoubleType> builder;
                auto min = df.min<arrow::DoubleType>(j);
                auto max = df.max<arrow::DoubleType>(j);
                if (max != min) {
                    auto column_cast = std::static_pointer_cast<ArrayType>(column);
                    for (int i = 0; i < df->num_rows(); ++i) {
                        auto normalized_value = (column_cast->Value(i) - min)/(max - min);
                        RAISE_STATUS_ERROR(builder.Append(normalized_value));
                    }
                    Array_ptr out;
                    RAISE_STATUS_ERROR(builder.Finish(&out));
                    new_columns.push_back(out);
                    auto f = arrow::field(df.name(j), out->type());
                    RAISE_STATUS_ERROR(b.AddField(f));
                     
                }
                else {
                    throw std::invalid_argument("Constant column in DataFrame.");
                }
                break; 
                
            }
            default:
            throw std::invalid_argument("Wrong data type in MSKMutualInformation.");
        }
        
        
    }
    RAISE_RESULT_ERROR(auto schema, b.Finish())

    auto rb = arrow::RecordBatch::Make(schema, df->num_rows(), new_columns);
    return DataFrame(rb);
}
  
double mi_pair(const DataFrame& df, int k) {
    VPTree vptree(df);
    // auto knn_results = kdtree.query(df, k + 1, std::numeric_limits<double>::infinity()); // excluding the reference point which is not a neighbor of itself
    // double res = 0;

    // if (df.is_continuous(0) && df.is_continuous(1)) {
    //     for (int i = 0, rows = static_cast<int>(df->num_rows()); i < rows; ++i) {
    //         auto eps_i = static_cast<int>(knn_results[i].first(k););

            
    //         auto raw_values1 = df.data<arrow::FloatType>(0);
    //         auto raw_values2 = df.data<arrow::FloatType>(1);
    //         auto v1 = static_cast<int>(raw_values1[i]);
    //         auto v2 = static_cast<int>(raw_values2[i]);
            
    //         auto nv1 = std::min(1 + v1, eps_i) + std::min(rows - v1, eps_i) - 1;
    //         auto nv2 = std::min(1 + v2, eps_i) + std::min(rows - v2, eps_i) - 1;
    //         auto k_hat = k;
        
    //         res -= boost::math::digamma(nv1) + boost::math::digamma(nv2);
    //         res += boost::math::digamma(k_hat) + boost::math::digamma(df->num_rows());
    //     } 

    // } else if (df.is_continuous(0) && df.is_continuous(1) || df.is_continuous(1) && df.is_continuous(0)) {
    //     int c_idx = (df.is_continuous(0)) ? 0 : 1;
    //     for (int i = 0, rows = static_cast<int>(df->num_rows()); i < rows; ++i) {
    //         auto eps_i = static_cast<int>(knn_results[i].first(k););

            
    //         auto raw_values1 = df.data<arrow::FloatType>(c_idx);
    //         auto raw_values2 = df.data<arrow::FloatType>(1-c_idx);
    //         auto v1 = static_cast<int>(raw_values1[i]);
    //         auto v2 = static_cast<int>(raw_values2[i]);
            
    //         auto nv1 = std::min(1 + v1, eps_i) + std::min(rows - v1, eps_i) - 1;
    //         auto nv2 = std::min(1 + v2, eps_i) + std::min(rows - v2, eps_i) - 1;
    //         auto k_hat = k; // count ball subspaces first
        
    //         res -= boost::math::digamma(nv1) + boost::math::digamma(nv2);
    //         res += boost::math::digamma(k_hat) + boost::math::digamma(df->num_rows());
    //     } 
    // }

    // res /= df->num_rows();
    
    return 1.0;
}

// double mi_triple(const DataFrame& df, int k) {
//     VPTree vptree(df);
//     auto knn_results = HybridChebyshevDistance<arrow::DoubleType> dist(train, test, is_discrete_column, 1);

//     VectorXd eps(df->num_rows());
//     for (auto i = 0; i < df->num_rows(); ++i) {
//         eps(i) = knn_results[i].first(k);
//     }

//     VectorXi n_xz = VectorXi::Zero(df->num_rows());
//     VectorXi n_yz = VectorXi::Zero(df->num_rows());
//     VectorXi n_z(df->num_rows());

//     auto raw_x = df.data<arrow::FloatType>(0);
//     auto raw_y = df.data<arrow::FloatType>(1);
//     auto raw_z = df.data<arrow::FloatType>(2);

//     IndexComparator comp_z(raw_z);
//     std::vector<size_t> sort_z(df->num_rows());
//     std::iota(sort_z.begin(), sort_z.end(), 0);
//     std::sort(sort_z.begin(), sort_z.end(), comp_z);

//     for (int i = 0, rows = static_cast<int>(df->num_rows()); i < rows; ++i) {
//         auto eps_i = static_cast<int>(eps(i));
//         auto x_i = static_cast<int>(raw_x[i]);
//         auto y_i = static_cast<int>(raw_y[i]);
//         auto z_i = static_cast<int>(raw_z[i]);

//         n_z(i) = std::min(1 + z_i, eps_i) + std::min(rows - z_i, eps_i) - 1;

//         if (z_i < eps_i) {
//             for (int j = 0, end = z_i + eps_i; j < end; ++j) {
//                 auto index = sort_z[j];
//                 auto x_value = raw_x[index];
//                 auto y_value = raw_y[index];
//                 if (std::abs(x_i - x_value) < eps_i) ++n_xz(i);
//                 if (std::abs(y_i - y_value) < eps_i) ++n_yz(i);
//             }
//         } else if (z_i > (rows - eps_i)) {
//             for (int j = z_i - eps_i + 1, end = df->num_rows(); j < end; ++j) {
//                 auto index = sort_z[j];
//                 auto x_value = raw_x[index];
//                 auto y_value = raw_y[index];
//                 if (std::abs(x_i - x_value) < eps_i) ++n_xz(i);
//                 if (std::abs(y_i - y_value) < eps_i) ++n_yz(i);
//             }
//         } else {
//             for (int j = z_i - eps_i + 1, end = z_i + eps_i; j < end; ++j) {
//                 auto index = sort_z[j];
//                 auto x_value = raw_x[index];
//                 auto y_value = raw_y[index];
//                 if (std::abs(x_i - x_value) < eps_i) ++n_xz(i);
//                 if (std::abs(y_i - y_value) < eps_i) ++n_yz(i);
//             }
//         }
//     }

//     double res = 0;
//     for (int i = 0; i < df->num_rows(); ++i) {
//         res += boost::math::digamma(n_z(i)) - boost::math::digamma(n_xz(i)) - boost::math::digamma(n_yz(i));
//     }

//     res /= df->num_rows();
//     res += boost::math::digamma(k);

//     return res;
// }

// double mi_general(const DataFrame& df, int k) {
//     VPTree vptree(df);
//     auto knn_results = HybridChebyshevDistance<arrow::DoubleType> dist(train, test, is_discrete_column, 1);

//     VectorXd eps(df->num_rows());
//     for (auto i = 0; i < df->num_rows(); ++i) {
//         eps(i) = knn_results[i].first(k);
//     }

//     std::vector<size_t> indices(df->num_columns() - 2);
//     std::iota(indices.begin(), indices.end(), 2);
//     auto z_df = df.loc(indices);
//     VPTree ztree(z_df);
//     auto [n_xz, n_yz, n_z] = ztree.count_ball_subspaces(z_df, df.col(0), df.col(1), eps);

//     double res = 0;
//     for (int i = 0; i < df->num_rows(); ++i) {
//         res += boost::math::digamma(n_z(i)) - boost::math::digamma(n_xz(i)) - boost::math::digamma(n_yz(i));
//     }

//     res /= df->num_rows();
//     res += boost::math::digamma(k);

//     return res;
// }

double MSKMutualInformation::mi(const std::string& x, const std::string& y) const {
    auto subset_df = m_scaled_df.loc(x, y);
    return mi_pair(subset_df, m_k);
}

// double MSKMutualInformation::mi(const std::string& x, const std::string& y, const std::string& z) const {
//     auto subset_df = m_scaled_df.loc(x, y, z);
//     return mi_triple(subset_df, m_k);
// }

// double MSKMutualInformation::mi(const std::string& x, const std::string& y, const std::vector<std::string>& z) const {
//     auto subset_df = m_scaled_df.loc(x, y, z);
//     return mi_general(subset_df, m_k);
// }

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

double MSKMutualInformation::pvalue(const std::string& x, const std::string& y, const std::vector<std::string>& z) const {
    // auto original_mi = mi(x, y, z);
    // auto z_df = m_df.loc(z);
    // auto shuffled_df = m_scaled_df.loc(Copy(x), y, z);
    // auto original_rank_x = m_scaled_df.template data<arrow::FloatType>(x);

    // return shuffled_pvalue(original_mi, original_rank_x, z_df, shuffled_df, MIGeneral{});
    return 1.0;
}

}  // namespace learning::independences::continuous