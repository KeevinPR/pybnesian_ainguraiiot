#include <learning/independences/hybrid/ms/vptree.hpp>

namespace vptree {

std::unique_ptr<VPTreeNode> VPTree::build_vptree(const DataFrame& df, const std::vector<bool>& is_discrete_column, std::vector<size_t>& indices_parent) {
    // switch (df.same_type()->id()) {
    //     case Type::DOUBLE: {
    //         return vptree::build_vptree<arrow::DoubleType>(
    //             df, leafsize, m_indices.begin(), m_indices.end(), -1, true, m_maxes, m_mines);
    //     }
    //     case Type::FLOAT: {
    //         return vptree::build_vptree<arrow::FloatType>(df,
    //                                                       leafsize,
    //                                                       m_indices.begin(),
    //                                                       m_indices.end(),
    //                                                       -1,
    //                                                       true,
    //                                                       m_maxes.template cast<float>(),
    //                                                       m_mines.template cast<float>());
    //     }
    //     default:
    //         throw std::invalid_argument("Wrong data type to apply VPTree.");
    // }
    return vptree::build_vptree<arrow::DoubleType>(df, is_discrete_column, indices_parent);
}

// void VPTree::fit(DataFrame df) {
//     m_df = df;
//     m_column_names = df.column_names();
//     m_indices.resize(df->num_rows());
//     std::iota(m_indices.begin(), m_indices.end(), 0);
//     m_maxes = VectorXd(df->num_columns());
//     m_mines = VectorXd(df->num_columns());

//     m_root = build_vptree(df);
// }

template <typename ArrowType>
std::vector<std::pair<VectorXd, VectorXi>> VPTree::query(const DataFrame& test_df, int k) const {
    if (k >= m_df->num_rows()) {
        throw std::invalid_argument("\"k\" value equal or greater to training data size.");
    }

    test_df.raise_has_columns(m_column_names);

    std::vector<std::pair<VectorXd, VectorXi>> res;
    res.reserve(test_df->num_rows());

    HybridChebyshevDistance<ArrowType> dist(test_df, m_is_discrete_column);
    for (int i = 0; i < test_df->num_rows(); ++i) {
        auto t = query_instance<ArrowType>(test_df, i, k, dist);
        res.push_back(t);
    }
    
    return res;
}

// std::tuple<VectorXi, VectorXi, VectorXi> VPTree::count_ball_subspaces(const DataFrame& test_df,
//                                                                       const Array_ptr& x_data,
//                                                                       const Array_ptr& y_data,
//                                                                       const VectorXd& eps) const {
//     VectorXi count_xz(test_df->num_rows());
//     VectorXi count_yz(test_df->num_rows());
//     VectorXi count_z(test_df->num_rows());

//     std::vector<bool> is_discrete_column(m_df->num_columns(), false);
    

//     switch (m_datatype->id()) {
//         case Type::DOUBLE: {
//             auto train = m_df.downcast_vector<arrow::DoubleType>();
//             auto test = test_df.downcast_vector<arrow::DoubleType>();
//             HybridChebyshevDistance<arrow::DoubleType> dist(train, test, m_is_discrete_column);

//             auto x = std::static_pointer_cast<arrow::DoubleArray>(x_data)->raw_values();
//             auto y = std::static_pointer_cast<arrow::DoubleArray>(y_data)->raw_values();

//             for (int i = 0; i < test_df->num_rows(); ++i) {
//                 auto c = count_ball_subspaces_instance<arrow::DoubleType>(test, x, y, i, dist, eps(i));

//                 count_xz(i) = std::get<0>(c);
//                 count_yz(i) = std::get<1>(c);
//                 count_z(i) = std::get<2>(c);
//             }
//             break;
//         }
//         case Type::FLOAT: {
//             auto train = m_df.downcast_vector<arrow::FloatType>();
//             auto test = test_df.downcast_vector<arrow::FloatType>();
//             HybridChebyshevDistance<arrow::DoubleType> dist(train, test, m_is_discrete_column);

//             auto x = std::static_pointer_cast<arrow::FloatArray>(x_data)->raw_values();
//             auto y = std::static_pointer_cast<arrow::FloatArray>(y_data)->raw_values();

//             for (int i = 0; i < test_df->num_rows(); ++i) {
//                 auto c = count_ball_subspaces_instance<arrow::FloatType>(test, x, y, i, dist, eps(i));

//                 count_xz(i) = std::get<0>(c);
//                 count_yz(i) = std::get<1>(c);
//                 count_z(i) = std::get<2>(c);
//             }
//             break;
//         }
//         default:
//             throw std::invalid_argument("Wrong data type to apply VPTree.");
//     }
//     return std::make_tuple(count_xz, count_yz, count_z);
// }

}  // namespace vptree
