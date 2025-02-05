#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_VPTREE_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_VPTREE_HPP

#include <dataset/dataset.hpp>
#include <queue>

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

    HybridChebyshevDistance(const DataFrame& df,
                            const std::vector<bool>& is_discrete_column)
        : m_df(df), m_is_discrete_column(is_discrete_column) {}

    inline CType distance(size_t p1_index, size_t p2_index) const {
        CType d = 0, t = 0;
        for (int col_idx = 0; col_idx < m_df->num_columns(); col_idx++) {
            if (m_is_discrete_column[col_idx]) {
                auto column = std::static_pointer_cast<ArrayType>(m_df.col(col_idx));
                t = (column->Value(p2_index) == column->Value(p1_index)) ? 0 : 1;
            } else {
                auto column = std::static_pointer_cast<ArrayType>(m_df.col(col_idx));
                t = (column->Value(p2_index) - column->Value(p1_index));
            }

            d = std::max(d, abs(t));
        }

        return d;
    }
    

    inline CType distance_coord(size_t p1_index, size_t p2_index, int coord) const {
        CType d = 0;

        if (m_is_discrete_column[coord]) {
                auto column = std::static_pointer_cast<ArrayType>(m_df.col(coord));
                d = (column->Value(p2_index) == column->Value(p1_index)) ? 0 : 1;
            } else {
                auto column = std::static_pointer_cast<ArrayType>(m_df.col(coord));
                d = abs(column->Value(p2_index) - column->Value(p1_index));
            }

        return d;
    }

    inline CType distance_coords(size_t p1_index, size_t p2_index, std::vector<int>& coords) const {
        CType d = 0, t = 0;
        for (auto it_col_idx = coords.begin(); it_col_idx != coords.end(); it_col_idx++) {
            if (m_is_discrete_column[*it_col_idx]) {
                auto column = std::static_pointer_cast<ArrayType>(m_df.col(*it_col_idx));
                t = (column->Value(p2_index) == column->Value(p1_index)) ? 0 : 1;
            } else {
                auto column = std::static_pointer_cast<ArrayType>(m_df.col(*it_col_idx));
                t = (column->Value(p2_index) - column->Value(p1_index));
            }

            d = std::max(d, abs(t));
        }

        return d;
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

    if (indices_parent.size() == 1) {
        auto leaf = std::make_unique<VPTreeNode>();
        leaf->index = indices_parent.front();
        leaf->threshold = 0.0;
        leaf->is_leaf = true;
        return leaf;
    }

    size_t rand_selection = rand() % indices_parent.size();
    // size_t rand_selection = 0;
    size_t vp_index = indices_parent[rand_selection];
    indices_parent.erase(indices_parent.begin() + rand_selection);

    std::vector<std::pair<CType, size_t>> distances_indices(indices_parent.size());

    HybridChebyshevDistance<ArrowType> distance(df, is_discrete_column);

    for (size_t i = 0; i < distances_indices.size(); ++i) {
        distances_indices[i] = std::make_pair(distance.distance(indices_parent[i], vp_index), indices_parent[i]);
    }
    
    std::nth_element(distances_indices.begin(), distances_indices.begin() + distances_indices.size()/2, distances_indices.end(), [](const std::pair<CType, size_t>& a, const std::pair<CType, size_t>& b) {
        return a.first < b.first;
    });
    double threshold = distances_indices[distances_indices.size()/2].first;

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
    node->left = build_vptree<ArrowType>(df, is_discrete_column, indices_left);
    node->right = build_vptree<ArrowType>(df, is_discrete_column, indices_right);
    
    node->is_leaf = false;

    return node;
}


class VPTree {
public:

    VPTree(DataFrame df, std::shared_ptr<arrow::DataType> datatype, std::vector<bool>& is_discrete_column) : m_df(df), m_is_discrete_column(is_discrete_column), m_datatype(datatype), m_column_names(df.column_names()), m_root() {
        std::vector<size_t> indices(m_df->num_rows());
        std::iota(indices.begin(), indices.end(), 0);
        m_root = build_vptree(m_df, m_is_discrete_column, indices);
    }

    void fit(DataFrame df);

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
                                                                const HybridChebyshevDistance<ArrowType>& distance_xyz,
                                                                const typename ArrowType::c_type eps_value) const;

    const DataFrame& scaled_data() const { return m_df; }

private:
    std::unique_ptr<VPTreeNode> build_vptree(const DataFrame& df, const std::vector<bool>& is_discrete_column, std::vector<size_t>& indices_parent);

    DataFrame m_df;
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

        auto dist = distance.distance(node->index, i);

        if (dist <= distance_upper_bound) {
            if (neighborhood.size() == static_cast<std::size_t>(k)){
                if (dist < distance_upper_bound){
                    neighborhood.pop();
                    neighborhood.push(std::make_pair(dist, node->index));
                    if (!neighborhood_star.empty() && neighborhood_star.front().first > neighborhood.top().first) {
                        neighborhood_star.clear();
                    }
                }
                else {
                    neighborhood_star.push_back(std::make_pair(dist, node->index));
                }
            }
            else {
                neighborhood.push(std::make_pair(dist, node->index));
            }
        }

        if (neighborhood.size() == static_cast<std::size_t>(k)){
            distance_upper_bound = neighborhood.top().first;
        }

        CType left_min_distance = std::max(dist - node->threshold, 0.0);

        if (node->left && left_min_distance <= distance_upper_bound) {
       
            query_nodes.push(QueryNode<ArrowType>{node->left.get(), left_min_distance});
            
        } 
        
        CType right_min_distance = std::max(node->threshold - dist, 0.0);;
          
        if (node->right && right_min_distance <= distance_upper_bound){
            
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
std::tuple<int, int, int> VPTree::count_ball_subspaces_instance(size_t i,
                                                                const HybridChebyshevDistance<ArrowType>& distance_xyz,
                                                                const typename ArrowType::c_type eps_value) const {
    using CType = typename ArrowType::c_type;
    
    CType min_distance = 0;

    int count_xz = 0, count_yz = 0, count_z = 0;

    QueryQueue<ArrowType> query_nodes;

    query_nodes.push(QueryNode<ArrowType>{/*.node = */ m_root.get(),
                                            /*.min_distance = */ min_distance});

    std::vector<int> z_indices(m_df->num_columns());
    std::iota(z_indices.begin(), z_indices.end(), 2);
    

    while (!query_nodes.empty()) {
        auto& query = query_nodes.top();
        auto node = query.node;

        query_nodes.pop();

        auto d_z = distance_xyz.distance_coords(node->index, i, z_indices);

        if (d_z <= eps_value) {
            ++count_z;
            if (distance_xyz.distance_coord(node->index, i, 0) <= eps_value) ++count_xz;
            if (distance_xyz.distance_coord(node->index, i, 1) <= eps_value) ++count_yz;
        }

        CType left_min_distance = std::max(d_z - node->threshold, 0.0);

        if (node->left && left_min_distance <= eps_value) {
       
            query_nodes.push(QueryNode<ArrowType>{node->left.get(), left_min_distance});
            
        }
        
        CType right_min_distance = std::max(node->threshold - d_z, 0.0);;
          
        if (node->right&& right_min_distance <= eps_value){
            
            query_nodes.push(QueryNode<ArrowType>{node->right.get(), right_min_distance});
           
        }
    }

    return std::make_tuple(count_xz, count_yz, count_z);
}

}  // namespace vptree

#endif  // PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_VPTREE_HPP