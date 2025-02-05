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

std::vector<std::pair<VectorXd, VectorXi>> VPTree::query(const DataFrame& test_df, int k) const {
    if (k >= m_df->num_rows()) {
        throw std::invalid_argument("\"k\" value equal or greater to training data size.");
    }

    test_df.raise_has_columns(m_column_names);

    std::vector<std::pair<VectorXd, VectorXi>> res;
    res.reserve(test_df->num_rows());

    switch (m_datatype->id()) {
        case Type::FLOAT:{
        HybridChebyshevDistance<arrow::FloatType> dist(test_df, m_is_discrete_column);
        for (int i = 0; i < test_df->num_rows(); ++i) {
            auto t = query_instance<arrow::FloatType>(i, k, dist);
            res.push_back(t);
        }
        break;
        }

        default:{
        HybridChebyshevDistance<arrow::DoubleType> dist(test_df, m_is_discrete_column);
        for (int i = 0; i < test_df->num_rows(); ++i) {
            auto t = query_instance<arrow::DoubleType>(i, k, dist);
            res.push_back(t);
        }}
    }
    
    return res;
}
 


std::tuple<VectorXi, VectorXi, VectorXi> VPTree::count_ball_subspaces(const DataFrame& test_df,
                                                                      const VectorXd& eps,
                                                                      std::vector<bool>& is_discrete_column) const {

    test_df.raise_has_columns(m_column_names);

    auto n_rows = test_df->num_rows();
    VectorXi count_xz(n_rows);
    VectorXi count_yz(n_rows);
    VectorXi count_z(n_rows);


    switch (m_datatype->id()) {
        case Type::FLOAT:{
            HybridChebyshevDistance<arrow::FloatType> distance_xyz(test_df, is_discrete_column);

            for (int i = 0; i < n_rows; ++i) {
                auto c = count_ball_subspaces_instance<arrow::FloatType>(i, distance_xyz, eps(i));

                count_xz(i) = std::get<0>(c);
                count_yz(i) = std::get<1>(c);
                count_z(i) = std::get<2>(c);
            }
            break;
            }
        default:{
            HybridChebyshevDistance<arrow::DoubleType> distance_xyz(test_df, is_discrete_column);

            for (int i = 0; i < n_rows; ++i) {
                auto c = count_ball_subspaces_instance<arrow::DoubleType>(i, distance_xyz, eps(i));

                count_xz(i) = std::get<0>(c);
                count_yz(i) = std::get<1>(c);
                count_z(i) = std::get<2>(c);
            }}
    }
  
    return std::make_tuple(count_xz, count_yz, count_z);
}
 
}  // namespace vptree
