#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_VPTREE_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_VPTREE_HPP

#include <dataset/dataset.hpp>
#include <queue>
#include <random>
#include <algorithm>

using dataset::DataFrame;
using Eigen::Matrix, Eigen::Dynamic, Eigen::VectorXd, Eigen::VectorXi;

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
    using OperationFunc = std::function<CType(size_t, size_t)>;

    HybridChebyshevDistance(const std::vector<std::shared_ptr<ArrayType>>& data,
                            const std::vector<bool>& is_discrete_column)
        : m_data(data) {
        m_operations_coords.reserve(m_data.size());
        for (size_t i = 0; i < m_data.size(); ++i) {
            if (is_discrete_column[i]) {
                // For discrete columns, Hamming distance
                m_operations_coords.push_back([this, i](size_t p1_index, size_t p2_index) -> CType {
                    return (m_data[i]->Value(p1_index) != m_data[i]->Value(p2_index));
                });
            } else {
                // For continuous columns, Manhattan distance
                m_operations_coords.push_back([this, i](size_t p1_index, size_t p2_index) -> CType {
                    return std::abs(m_data[i]->Value(p1_index) - m_data[i]->Value(p2_index));
                });
            }
        }
    }

    inline CType distance(size_t p1_index, size_t p2_index) const {
        CType d = 0;
        for (auto it_operation = m_operations_coords.begin(), it_end = m_operations_coords.end();
             it_operation != it_end;
             ++it_operation) {
            d = std::max(d, (*it_operation)(p1_index, p2_index));
        }

        return d;
    }

    inline CType distance_coords(size_t p1_index, size_t p2_index, std::vector<int>& coords) const {
        CType d = 0;
        for (auto it_col_idx = coords.begin(); it_col_idx != coords.end(); it_col_idx++) {
            d = std::max(d, m_operations_coords[*it_col_idx](p1_index, p2_index));
        }

        return d;
    }

    inline CType distance_p(CType difference) const { return std::abs(difference); }

    inline CType normalize(CType nonnormalized) const { return nonnormalized; }

    inline CType update_component_distance(CType distance, CType, CType new_component) const {
        return std::max(distance, new_component);
    }

private:
    const std::vector<std::shared_ptr<ArrayType>>& m_data;
    std::vector<OperationFunc> m_operations_coords;
};

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
    std::vector<size_t> leaf_indices;
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
        return d > 0;
    }
};

template <typename ArrowType>
using QueryQueue =
    std::priority_queue<QueryNode<ArrowType>, std::vector<QueryNode<ArrowType>>, QueryNodeComparator<ArrowType>>;

template <typename ArrowType>
std::unique_ptr<VPTreeNode> build_vptree(const HybridChebyshevDistance<ArrowType>& distance,
                                         std::vector<size_t>& indices_parent,
                                         int leafsize) {
    using CType = typename ArrowType::c_type;

    if (indices_parent.empty()) return nullptr;

    if (indices_parent.size() <= static_cast<std::size_t>(leafsize)) {
        auto leaf = std::make_unique<VPTreeNode>();
        leaf->threshold = 0.0;
        leaf->is_leaf = true;
        leaf->leaf_indices = indices_parent;
        return leaf;
    }

    size_t rand_selection = rand() % indices_parent.size();
    std::iter_swap(indices_parent.begin() + rand_selection, indices_parent.begin());
    size_t vp_index = indices_parent[0];

    std::vector<std::pair<CType, size_t>> distances_indices(indices_parent.size() - 1);

    CType min = 0, max = std::numeric_limits<CType>::infinity();

    for (size_t i = 1; i < indices_parent.size(); ++i) {
        auto dist = distance.distance(indices_parent[i], vp_index);
        distances_indices[i - 1] = std::make_pair(dist, indices_parent[i]);
        if (dist > min) min = dist;
        if (dist < max) max = dist;

    }

    if (min == max) {
        auto leaf = std::make_unique<VPTreeNode>();
        leaf->threshold = 0.0;
        leaf->is_leaf = true;
        leaf->leaf_indices = indices_parent;
        return leaf;
    }


    std::nth_element(
        distances_indices.begin(),
        distances_indices.begin() + distances_indices.size() / 2,
        distances_indices.end(),
        [](const std::pair<CType, size_t>& a, const std::pair<CType, size_t>& b) { return a.first < b.first; });
    double threshold = distances_indices[distances_indices.size() / 2].first;

    std::vector<size_t> indices_left, indices_right;

    for (size_t i = 0; i < distances_indices.size(); ++i) {
        if (distances_indices[i].first <= threshold) {
            indices_left.push_back(distances_indices[i].second);
        } else {
            indices_right.push_back(distances_indices[i].second);
        }
    }

    auto node = std::make_unique<VPTreeNode>();

    node->index = vp_index;
    node->threshold = threshold;
    node->is_leaf = false;

    node->left = build_vptree<ArrowType>(distance, indices_left, leafsize);
    node->right = build_vptree<ArrowType>(distance, indices_right, leafsize);

    return node;
}

class VPTree {
public:
    VPTree(DataFrame& df,
           std::shared_ptr<arrow::DataType> datatype,
           std::vector<bool>& is_discrete_column,
           int leafsize = 16)
        : m_df(df),
          m_datatype(datatype),
          m_is_discrete_column(is_discrete_column),
          m_column_names(df.column_names()),
          m_root() {
        m_root = build_vptree(m_df, m_datatype, m_is_discrete_column, leafsize);
    }

    std::vector<std::pair<VectorXd, VectorXi>> query(const DataFrame& test_df, int k) const;

    template <typename ArrowType>
    std::pair<VectorXd, VectorXi> query_instance(size_t i,
                                                 int k,
                                                 const HybridChebyshevDistance<ArrowType>& distance) const;

    std::tuple<VectorXi, VectorXi, VectorXi> count_ball_subspaces(const DataFrame& test_df,
                                                                  const VectorXd& eps,
                                                                  std::vector<bool>& is_discrete_column) const;

    template <typename ArrowType>
    std::tuple<int, int, int> count_ball_subspaces_instance(size_t i,
                                                            const typename ArrowType::c_type eps_value,
                                                            const HybridChebyshevDistance<ArrowType>& distance) const;

    VectorXi count_ball_unconditional(const DataFrame& test_df,
                                      const VectorXd& eps,
                                      std::vector<bool>& is_discrete_column) const;

    template <typename ArrowType>
    int count_ball_unconditional_instance(size_t i,
                                          const typename ArrowType::c_type eps_value,
                                          const HybridChebyshevDistance<ArrowType>& distance) const;

    const DataFrame& scaled_data() const { return m_df; }

private:
    std::unique_ptr<VPTreeNode> build_vptree(const DataFrame& df,
                                             const std::shared_ptr<arrow::DataType> datatype,
                                             const std::vector<bool>& is_discrete_column,
                                             int leafsize);

    DataFrame& m_df;
    std::shared_ptr<arrow::DataType> m_datatype;
    std::vector<bool>& m_is_discrete_column;
    std::vector<std::string> m_column_names;
    std::unique_ptr<VPTreeNode> m_root;
};

template <typename ArrowType>
std::pair<VectorXd, VectorXi> VPTree::query_instance(size_t i,
                                                     int k,
                                                     const HybridChebyshevDistance<ArrowType>& distance) const {
    using CType = typename ArrowType::c_type;

    NeighborQueue<ArrowType> neighborhood;

    std::vector<Neighbor<ArrowType>> neighborhood_star;

    CType distance_upper_bound = std::numeric_limits<CType>::infinity(), distance_neigh = 0;

    QueryQueue<ArrowType> query_nodes;
    CType min_distance = 0;

    query_nodes.push(QueryNode<ArrowType>{/*.node = */ m_root.get(),
                                          /*.min_distance = */ min_distance});

    while (!query_nodes.empty()) {
        auto& query = query_nodes.top();
        auto node = query.node;

        query_nodes.pop();

        if (query.min_distance > distance_upper_bound) continue;

        std::vector<size_t> eval_neighbors(1, node->index);

        if (node->is_leaf) {
            eval_neighbors = node->leaf_indices;
        }

        for (auto it_neigh = eval_neighbors.begin(), neigh_end = eval_neighbors.end(); it_neigh != neigh_end;
             ++it_neigh) {
            distance_neigh = distance.distance(*it_neigh, i);

            if (neighborhood.size() == static_cast<std::size_t>(k)) {
                if (distance_neigh < distance_upper_bound) {
                    neighborhood.pop();
                    neighborhood.push(std::make_pair(distance_neigh, *it_neigh));
                    if (!neighborhood_star.empty() && neighborhood_star.front().first > neighborhood.top().first) {
                        neighborhood_star.clear();
                    }
                } else if (distance_neigh == distance_upper_bound) {
                    neighborhood_star.push_back(std::make_pair(distance_neigh, *it_neigh));
                }
            } else {
                neighborhood.push(std::make_pair(distance_neigh, *it_neigh));
            }

            if (neighborhood.size() == static_cast<std::size_t>(k)) {
                distance_upper_bound = neighborhood.top().first;
            }
        }

        CType left_min_distance = std::max(distance_neigh - node->threshold, 0.0);

        if (node->left && left_min_distance <= distance_upper_bound) {
            query_nodes.push(QueryNode<ArrowType>{node->left.get(), left_min_distance});
        }

        CType right_min_distance = std::max(node->threshold - distance_neigh, 0.0);

        if (node->right && right_min_distance <= distance_upper_bound) {
            query_nodes.push(QueryNode<ArrowType>{node->right.get(), right_min_distance});
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

template <typename ArrowType>
std::tuple<int, int, int> VPTree::count_ball_subspaces_instance(
    size_t i,
    const typename ArrowType::c_type eps_value,
    const HybridChebyshevDistance<ArrowType>& distance_xyz) const {
    using CType = typename ArrowType::c_type;

    CType min_distance = 0, d_z = 0;

    int count_xz = 0, count_yz = 0, count_z = 0;

    QueryQueue<ArrowType> query_nodes;

    query_nodes.push(QueryNode<ArrowType>{/*.node = */ m_root.get(),
                                          /*.min_distance = */ min_distance});

    std::vector<int> z_indices(m_df->num_columns());
    std::iota(z_indices.begin(), z_indices.end(), 2);

    std::vector<int> x_index(1, 0);
    std::vector<int> y_index(1, 1);

    while (!query_nodes.empty()) {
        auto& query = query_nodes.top();
        auto node = query.node;

        query_nodes.pop();

        std::vector<size_t> eval_neighbors(1, node->index);

        if (node->is_leaf) {
            eval_neighbors = node->leaf_indices;
        }

        for (auto it_neigh = eval_neighbors.begin(), neigh_end = eval_neighbors.end(); it_neigh != neigh_end;
             ++it_neigh) {
            d_z = distance_xyz.distance_coords(*it_neigh, i, z_indices);

            if (d_z <= eps_value) {
                ++count_z;
                if (distance_xyz.distance_coords(*it_neigh, i, x_index) <= eps_value) ++count_xz;
                if (distance_xyz.distance_coords(*it_neigh, i, y_index) <= eps_value) ++count_yz;
            }
        }

        CType left_min_distance = std::max(d_z - node->threshold, 0.0);

        if (node->left && left_min_distance <= eps_value) {
            query_nodes.push(QueryNode<ArrowType>{node->left.get(), left_min_distance});
        }

        CType right_min_distance = std::max(node->threshold - d_z, 0.0);

        if (node->right && right_min_distance <= eps_value) {
            query_nodes.push(QueryNode<ArrowType>{node->right.get(), right_min_distance});
        }
    }

    return std::make_tuple(count_xz, count_yz, count_z);
}

template <typename ArrowType>
int VPTree::count_ball_unconditional_instance(
    size_t i,
    const typename ArrowType::c_type eps_value,
    const HybridChebyshevDistance<ArrowType>& distance) const {
    using CType = typename ArrowType::c_type;

    CType min_distance = 0, distance_neigh = 0;

    int count_n = 0;

    QueryQueue<ArrowType> query_nodes;

    query_nodes.push(QueryNode<ArrowType>{/*.node = */ m_root.get(),
                                          /*.min_distance = */ min_distance});

    while (!query_nodes.empty()) {
        auto& query = query_nodes.top();
        auto node = query.node;

        query_nodes.pop();

        std::vector<size_t> eval_neighbors(1, node->index);

        if (node->is_leaf) {
            eval_neighbors = node->leaf_indices;
        }

        for (auto it_neigh = eval_neighbors.begin(), neigh_end = eval_neighbors.end(); it_neigh != neigh_end;
             ++it_neigh) {
            distance_neigh = distance.distance(*it_neigh, i);

            if (distance_neigh <= eps_value) {
                ++count_n;
            }
        }

        CType left_min_distance = std::max(distance_neigh - node->threshold, 0.0);

        if (node->left && left_min_distance <= eps_value) {
            query_nodes.push(QueryNode<ArrowType>{node->left.get(), left_min_distance});
        }

        CType right_min_distance = std::max(node->threshold - distance_neigh, 0.0);

        if (node->right && right_min_distance <= eps_value) {
            query_nodes.push(QueryNode<ArrowType>{node->right.get(), right_min_distance});
        }
    }

    return count_n;
}

}  // namespace vptree

#endif  // PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_VPTREE_HPP