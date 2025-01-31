#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_VPTREE_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_VPTREE_HPP

#include <dataset/dataset.hpp>
#include <queue>

using dataset::DataFrame;
using Eigen::Matrix, Eigen::Dynamic, Eigen::VectorXd, Eigen::VectorXi;

template <typename ArrowType>
using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;

template <typename ArrowType>
using DowncastArray_ptr = std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>;

template <typename ArrowType>
using DowncastArray_vector = std::vector<DowncastArray_ptr<ArrowType>>;

namespace vptree {

template <typename ArrowType>
class HybridChebyshevDistance {
public:

    using CType = typename ArrowType::c_type;
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

    HybridChebyshevDistance(const DataFrame& df,
                            const std::vector<bool>& is_discrete_column)
        : m_df(df), m_is_discrete_column(is_discrete_column) {}

    inline CType distance(size_t p1_index, size_t p2_index) const {
        CType d = 0, t = 0;
        for (int col_idx = 0; col_idx < m_df.num_columns(); col_idx++) {
            if (m_is_discrete_column[col_idx]) {
                auto column = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(col_idx));
                auto indices = std::static_pointer_cast<arrow::Int32Array>(column->indices());
                t = (indices->Value(p2_index) == indices->Value(p1_index)) ? 0 : 1;
            } else {
                auto column = std::static_pointer_cast<ArrayType>(m_df.col(col_idx));
                t = (column->Value(p2_index) - column->Value(p1_index));
            }

            d = std::max(d, abs(t));
            // d += t*t;
        }

        return d;
        // return std::sqrt(d);
    }

    inline CType distance_p(CType difference) const { return std::abs(difference); }

    inline CType normalize(CType nonnormalized) const { return nonnormalized; }

    inline CType update_component_distance(CType distance, CType, CType new_component) const {
        return std::max(distance, new_component);
    }

private:
    const DataFrame& m_df;
    const std::vector<bool>& m_is_discrete_column;
};

// template <typename T>
// struct IndexComparator {
//     const T* data;

//     IndexComparator(const T* data) : data(data) {};

//     inline bool operator()(size_t a, size_t b) { return data[a] < data[b]; }
// };

template <typename ArrowType>
using Neighbor = std::pair<typename ArrowType::c_type, size_t>;

template <typename ArrowType>
struct NeighborComparator {
    inline bool operator()(const Neighbor<ArrowType>& a, const Neighbor<ArrowType>& b) { return a.first < b.first; }
};

template <typename ArrowType>
using NeighborQueue =
    std::priority_queue<Neighbor<ArrowType>, std::vector<Neighbor<ArrowType>>, NeighborComparator<ArrowType>>;

struct VPTreeNode {
    size_t index;
    double threshold;
    std::unique_ptr<VPTreeNode> left;
    std::unique_ptr<VPTreeNode> right;
    bool is_leaf;
};

template <typename ArrowType>
struct QueryNode {
    VPTreeNode* node;
    typename ArrowType::c_type min_distance;
};

template <typename ArrowType>
struct QueryNodeComparator {
    inline bool operator()(const QueryNode<ArrowType>& a, const QueryNode<ArrowType>& b) {
        auto d = a.min_distance - b.min_distance;
        if (d != 0) {
            return d > 0;
        } else {
            return a.node->is_leaf < b.node->is_leaf;
        }
    }
};

template <typename ArrowType>
using QueryQueue =
    std::priority_queue<QueryNode<ArrowType>, std::vector<QueryNode<ArrowType>>, QueryNodeComparator<ArrowType>>;

template <typename ArrowType>
std::unique_ptr<VPTreeNode> build_vptree(const DataFrame& df, const std::vector<bool>& is_discrete_column, std::vector<size_t>& indices_parent) {
    using CType = typename ArrowType::c_type;

    if (indices_parent.empty()) return nullptr;

    size_t rand_selection = rand() % indices_parent.size();
    size_t vp_index = indices_parent[rand_selection];
    indices_parent.erase(indices_parent.begin() + rand_selection);

    if (indices_parent.empty()) {
        auto leaf = std::make_unique<VPTreeNode>();
        leaf->index = vp_index;
        leaf->threshold = 0.0;
        leaf->is_leaf = true;
        return leaf;
    }

    std::vector<CType> distances(indices_parent.size(), 0);

    HybridChebyshevDistance<ArrowType> distance(df, is_discrete_column);
    double aux = 0;
    for (size_t i = 0; i < distances.size(); ++i) {
        distances[i] = distance.distance(indices_parent[i], vp_index);
        aux = std::max(distances[i], aux);
    }
    
    std::nth_element(distances.begin(), distances.begin() + distances.size()/2, distances.end());
    double threshold = distances[distances.size()/2];

    std::vector<size_t> indices_left, indices_right;

    for (size_t i = 0; i < distances.size(); ++i) {
        if (distances[i] <= threshold) {
            indices_left.push_back(indices_parent[i]);
        } else {
            indices_right.push_back(indices_parent[i]);
        }
    }

    auto node = std::make_unique<VPTreeNode>();

    node->index = vp_index;
    node->threshold = threshold;   
    node->left = build_vptree<ArrowType>(df, is_discrete_column, indices_left);
    node->right = build_vptree<ArrowType>(df, is_discrete_column, indices_right);
    
    // if (node->left && node->right) {
    //     printf("%lf %lf %lf\n", node->left->threshold, threshold, node->right->threshold);
    //     fflush(stdout);
    // }
    node->is_leaf = false;

    return node;
}


class VPTree {
public:

    VPTree(DataFrame df) : m_df(df), m_column_names(df.column_names()), m_root(), m_is_discrete_column() {
        std::vector<bool> is_discrete_column(m_df->num_columns(), false);
        for (int c=0; c<m_df->num_columns(); c++){
            is_discrete_column[c] = m_df.is_discrete(c);
        }
        m_is_discrete_column = is_discrete_column;
        std::vector<size_t> indices(m_df->num_rows());
        std::iota(indices.begin(), indices.end(), 0);
        m_root = build_vptree(m_df, m_is_discrete_column, indices);
    }

    void fit(DataFrame df);

    template <typename ArrowType>
    std::vector<std::pair<VectorXd, VectorXi>> query(const DataFrame& test_df, int k = 1) const;

    template <typename ArrowType>
    std::pair<VectorXd, VectorXi> query_instance(const DowncastArray_vector<ArrowType>& test_df,
                                                 size_t i,
                                                 int k,
                                                 const HybridChebyshevDistance<ArrowType>& distance) const;

    std::tuple<VectorXi, VectorXi, VectorXi> count_ball_subspaces(const DataFrame& test_df,
                                                                  const Array_ptr& x_data,
                                                                  const Array_ptr& y_data,
                                                                  const VectorXd& eps) const;

    template <typename ArrowType, typename DistanceType>
    std::tuple<int, int, int> count_ball_subspaces_instance(const DowncastArray_vector<ArrowType>& test_df,
                                                            const typename ArrowType::c_type* x_data,
                                                            const typename ArrowType::c_type* y_data,
                                                            size_t i,
                                                            const DistanceType& distance,
                                                            const typename ArrowType::c_type eps_value) const;

    const DataFrame& scaled_data() const { return m_df; }

private:
    std::unique_ptr<VPTreeNode> build_vptree(const DataFrame& df, const std::vector<bool>& is_discrete_column, std::vector<size_t>& indices_parent);

    DataFrame m_df;
    std::vector<std::string> m_column_names;
    std::unique_ptr<VPTreeNode> m_root;
    std::vector<bool> m_is_discrete_column;
};

template <typename ArrowType>
std::pair<VectorXd, VectorXi> VPTree::query_instance(const DowncastArray_vector<ArrowType>& test_df,
                                                     size_t i,
                                                     int k,
                                                     const HybridChebyshevDistance<ArrowType>& distance) const {
    using CType = typename ArrowType::c_type;
   
    NeighborQueue<ArrowType> neighborhood;

    std::vector<Neighbor<ArrowType>> neighborhood_star;

    CType distance_upper_bound = std::numeric_limits<CType>::infinity();

    QueryQueue<ArrowType> query_nodes;
    CType min_distance = 0;

    query_nodes.push(QueryNode<ArrowType>{/*.node = */ m_root.get(),
                                          /*.min_distance = */ min_distance});

    while (!query_nodes.empty()) {
        auto& query = query_nodes.top();
        auto node = query.node;

        query_nodes.pop();

        if (query.min_distance > distance_upper_bound) continue;

        auto dist = distance.distance(node.index, i);

        if (dist <= distance_upper_bound) {
            if (neighborhood.size() == k){
                if (dist < distance_upper_bound){
                    neighborhood.pop();
                    neighborhood.push(std::make_pair(dist, node.index));
                }
                else {
                    neighborhood_star.push(std::make_pair(dist, node.index));
                }
            }
            else {
                neighborhood.push(std::make_pair(dist, node.index));
            }
        }

        if (neighborhood.size() == k){
            distance_upper_bound = neighborhood.top().first;
            if (neighborhood_star.front().first > distance_upper_bound) {
                neighborhood_star.clear();
            }
        }

        if (dist < node.threshold) {
            // Process left child first (min_distance is 0)
            if (node.left) {
                query_nodes.push(QueryNode<ArrowType>{node.left.get(), 0});
            }
            CType right_min_distance = node.threshold - dist;
            if (node.right && right_min_distance <= distance_upper_bound) {
                query_nodes.push(QueryNode<ArrowType>{node.right.get(), right_min_distance});
            }
        } else {
            // Process right child first (min_distance is 0)
            if (node.right) {
                query_nodes.push(QueryNode<ArrowType>{node.right.get(), 0});
            }
            CType left_min_distance = dist - node.threshold;
            if (node.left && left_min_distance <= distance_upper_bound) {
                query_nodes.push(QueryNode<ArrowType>{node.left.get(), left_min_distance});
            }
        }
    
        }
    

    auto k_hat = k + neighborhood_star.size();
    VectorXd distances(k_hat);
    VectorXi indices(k_hat);

    auto u = k_hat - 1;

    while (!neighborhood_star.empty()) {
        auto& neigh = neighborhood_star.back();
        distances(u) = neigh.first;
        indices(u) = neigh.second;
        neighborhood_star.pop_back();
        --u;
    }

    while (!neighborhood.empty()) {
        auto& neigh = neighborhood.top();
        distances(u) = neigh.first;
        indices(u) = neigh.second;
        neighborhood.pop();
        --u;
    }

    return std::make_pair(distances, indices);
}

// template <typename ArrowType, typename DistanceType>
// std::tuple<int, int, int> VPTree::count_ball_subspaces_instance(const DowncastArray_vector<ArrowType>& test_downcast,
//                                                                 const typename ArrowType::c_type* x_data,
//                                                                 const typename ArrowType::c_type* y_data,
//                                                                 size_t i,
//                                                                 const DistanceType& distance,
//                                                                 const typename ArrowType::c_type eps_value) const {
//     using CType = typename ArrowType::c_type;
//     using VectorType = Matrix<typename ArrowType::c_type, Dynamic, 1>;

//     VectorType side_distance(test_downcast.size());
//     CType min_distance = 0;

//     for (size_t j = 0; j < test_downcast.size(); ++j) {
//         auto p = test_downcast[j]->Value(i);
//         side_distance(j) = std::max(0., std::max(p - m_maxes(j), m_mines(j) - p));
//         side_distance(j) = distance.distance_p(side_distance(j));
//         min_distance = distance.update_component_distance(min_distance, 0, side_distance(j));
//     }

//     int count_xz = 0, count_yz = 0, count_z = 0;

//     QueryQueue<ArrowType> query_nodes;

//     if (min_distance < eps_value) {
//         query_nodes.push(QueryNode<ArrowType>{/*.node = */ m_root.get(),
//                                               /*.min_distance = */ min_distance,
//                                               /*.side_distance = */ side_distance});
//     }

//     while (!query_nodes.empty()) {
//         auto& query = query_nodes.top();
//         auto node = query.node;

//         if (node->is_leaf) {
//             for (auto it = node->indices_begin; it != node->indices_end; ++it) {
//                 auto d = distance.distance(*it, i);

//                 if (d < eps_value) {
//                     ++count_z;
//                     if (std::abs(x_data[*it] - x_data[i]) < eps_value) ++count_xz;
//                     if (std::abs(y_data[*it] - y_data[i]) < eps_value) ++count_yz;
//                 }
//             }

//             query_nodes.pop();
//         } else {
//             VPTreeNode* near_node;
//             VPTreeNode* far_node;

//             auto p = test_downcast[node->split_id]->Value(i);
//             if (p < node->split_value) {
//                 near_node = node->left.get();
//                 far_node = node->right.get();
//             } else {
//                 near_node = node->right.get();
//                 far_node = node->left.get();
//             }

//             QueryNode<ArrowType> near_query{/*.node = */ near_node,
//                                             /*.min_distance = */ query.min_distance,
//                                             /*.side_distance = */ query.side_distance};

//             CType far_dimension_distance = distance.distance_p(node->split_value - p);
//             CType far_node_distance = distance.update_component_distance(
//                 query.min_distance, query.side_distance(node->split_id), far_dimension_distance);

//             query_nodes.pop();
//             query_nodes.push(near_query);

//             if (far_node_distance < eps_value) {
//                 VectorType far_side_distance = near_query.side_distance;
//                 far_side_distance(node->split_id) = far_dimension_distance;
//                 query_nodes.push(QueryNode<ArrowType>{/*.node = */ far_node,
//                                                       /*.min_distance = */ far_node_distance,
//                                                       /*.side_distance = */ far_side_distance});
//             }
//         }
//     }

//     return std::make_tuple(count_xz, count_yz, count_z);
// }

}  // namespace vptree

#endif  // PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_VPTREE_HPP