#include <learning/independences/hybrid/ms/vptree.hpp>

namespace vptree {

template <typename ArrowType>
using Neighbor = std::pair<typename ArrowType::c_type, size_t>;

template <typename ArrowType>
struct NeighborComparator {
    inline bool operator()(const Neighbor<ArrowType>& a, const Neighbor<ArrowType>& b) { return a.first < b.first; }
};

template <typename ArrowType>
using NeighborQueue =
    std::priority_queue<Neighbor<ArrowType>, std::vector<Neighbor<ArrowType>>, NeighborComparator<ArrowType>>;

template <typename ArrowType>
struct QueryNode {
    VPTreeNode* node;
    typename ArrowType::c_type min_distance;
};

template <typename ArrowType>
struct QueryNodeComparator {
    inline bool operator()(const QueryNode<ArrowType>& a, const QueryNode<ArrowType>& b) {
        return a.min_distance > b.min_distance;
        ;
    }
};

template <typename ArrowType>
using QueryQueue =
    std::priority_queue<QueryNode<ArrowType>, std::vector<QueryNode<ArrowType>>, QueryNodeComparator<ArrowType>>;

template <typename ArrowType, typename Random>
std::unique_ptr<VPTreeNode> build_vptree(const HybridChebyshevDistance<ArrowType>& distance,
                                         std::vector<size_t>& indices_parent,
                                         int leafsize,
                                         Random& rng) {
    using CType = typename ArrowType::c_type;

    if (indices_parent.empty()) return nullptr;

    if (indices_parent.size() <= static_cast<std::size_t>(leafsize)) {
        auto leaf = std::make_unique<VPTreeNode>();
        leaf->threshold = 0.0;
        leaf->is_leaf = true;
        leaf->leaf_indices = indices_parent;
        return leaf;
    }

    size_t rand_selection = std::uniform_int_distribution<size_t>(0, indices_parent.size() - 1)(rng);
    std::iter_swap(indices_parent.begin() + rand_selection, indices_parent.begin());
    size_t vp_index = indices_parent[0];

    std::vector<std::pair<CType, size_t>> distances_indices(indices_parent.size() - 1);

    CType max = 0;

    for (size_t i = 1; i < indices_parent.size(); ++i) {
        auto dist = distance.distance(indices_parent[i], vp_index);
        distances_indices[i - 1] = std::make_pair(dist, indices_parent[i]);
        if (dist > max) max = dist;
    }

    if (max == 0) {
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
        [](const std::pair<CType, size_t>& a, const std::pair<CType, size_t>& b) { return a.first > b.first; });
    double threshold = distances_indices[distances_indices.size() / 2].first;

    std::vector<size_t> indices_left, indices_right;

    for (size_t i = 0; i < distances_indices.size(); ++i) {
        if (distances_indices[i].first < threshold) {
            indices_left.push_back(distances_indices[i].second);
        } else {
            indices_right.push_back(distances_indices[i].second);
        }
    }

    auto node = std::make_unique<VPTreeNode>();

    node->index = vp_index;
    node->threshold = threshold;
    node->is_leaf = false;

    node->left = build_vptree<ArrowType>(distance, indices_left, leafsize, rng);
    node->right = build_vptree<ArrowType>(distance, indices_right, leafsize, rng);

    return node;
}

std::unique_ptr<VPTreeNode> VPTree::build_vptree(const DataFrame& df,
                                                 const std::shared_ptr<arrow::DataType> datatype,
                                                 const std::vector<bool>& is_discrete_column,
                                                 int leafsize,
                                                 unsigned int seed) {
    std::vector<size_t> indices(m_df->num_rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng{seed};
    switch (datatype->id()) {
        case Type::DOUBLE: {
            auto data = df.downcast_vector<arrow::DoubleType>();

            HybridChebyshevDistance<arrow::DoubleType> distance(data, is_discrete_column);
            return vptree::build_vptree<arrow::DoubleType>(distance, indices, leafsize, rng);
        }
        case Type::FLOAT: {
            auto data = df.downcast_vector<arrow::FloatType>();

            HybridChebyshevDistance<arrow::FloatType> distance(data, is_discrete_column);
            return vptree::build_vptree<arrow::FloatType>(distance, indices, leafsize, rng);
        }
        default:
            throw std::invalid_argument("Wrong data type to apply VPTree.");
    }
}

std::vector<std::pair<VectorXd, VectorXi>> VPTree::query(const DataFrame& test_df, int k) const {
    if (k >= m_df->num_rows()) {
        throw std::invalid_argument("\"k\" value equal or greater to training data size.");
    }

    test_df.raise_has_columns(m_column_names);

    std::vector<std::pair<VectorXd, VectorXi>> res(test_df->num_rows());

    switch (m_datatype->id()) {
        case Type::FLOAT: {
            auto test = test_df.downcast_vector<arrow::FloatType>();
            HybridChebyshevDistance<arrow::FloatType> dist(test, m_is_discrete_column);

            auto hash_keys = hash_query_keys<arrow::FloatType>(test, m_column_names);

            for (int i = 0; i < test_df->num_rows(); ++i) {
                auto key = hash_keys[i];

                auto it = m_query_cache.find(key);
                if (it != m_query_cache.end()) {
                    res[i] = it->second;
                    continue;  // Skip the query, use cached result
                }

                auto t = query_instance<arrow::FloatType>(i, k, dist);

                res[i] = t;
                m_query_cache[key] = t;
            }

            break;
        }

        default: {
            auto test = test_df.downcast_vector<arrow::DoubleType>();

            HybridChebyshevDistance<arrow::DoubleType> dist(test, m_is_discrete_column);

            auto hash_keys = hash_query_keys<arrow::DoubleType>(test, m_column_names);
            for (int i = 0; i < test_df->num_rows(); ++i) {
                auto key = hash_keys[i];

                auto it = m_query_cache.find(key);
                if (it != m_query_cache.end()) {
                    res[i] = it->second;
                    continue;  // Skip the query, use cached result
                }

                auto t = query_instance<arrow::DoubleType>(i, k, dist);
                res[i] = t;

                m_query_cache[key] = t;
            }
        }
    }

    m_query_cache.clear();

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
        case Type::FLOAT: {
            auto test = test_df.downcast_vector<arrow::FloatType>();
            HybridChebyshevDistance<arrow::FloatType> distance_xyz(test, is_discrete_column);

            // auto hash_keys = hash_query_keys<arrow::FloatType>(test, test_df.column_names());

            for (int i = 0; i < n_rows; ++i) {
                // auto key = hash_keys[i];

                // boost::hash_combine(key, eps(i));

                // auto it = m_count_cache.find(key);
                // if (it != m_count_cache.end()) {
                //     count_xz(i) = std::get<0>(it->second);
                //     count_yz(i) = std::get<1>(it->second);
                //     count_z(i) = std::get<2>(it->second);
                //     continue;  // Skip the query, use cached result
                // }

                auto c = count_ball_subspaces_instance<arrow::FloatType>(i, eps(i), distance_xyz);

                count_xz(i) = std::get<0>(c);
                count_yz(i) = std::get<1>(c);
                count_z(i) = std::get<2>(c);

                // m_count_cache[key] = c;
            }
            break;
        }
        default: {
            auto test = test_df.downcast_vector<arrow::DoubleType>();
            HybridChebyshevDistance<arrow::DoubleType> distance_xyz(test, is_discrete_column);

            // auto hash_keys = hash_query_keys<arrow::DoubleType>(test, test_df.column_names());

            for (int i = 0; i < n_rows; ++i) {
                // auto key = hash_keys[i];

                // boost::hash_combine(key, eps(i));

                // auto it = m_count_cache.find(key);
                // if (it != m_count_cache.end()) {
                //     count_xz(i) = std::get<0>(it->second);
                //     count_yz(i) = std::get<1>(it->second);
                //     count_z(i) = std::get<2>(it->second);
                //     continue;  // Skip the query, use cached result
                // }

                auto c = count_ball_subspaces_instance<arrow::DoubleType>(i, eps(i), distance_xyz);

                count_xz(i) = std::get<0>(c);
                count_yz(i) = std::get<1>(c);
                count_z(i) = std::get<2>(c);

                // m_count_cache[key] = c;
            }
        }
    }

    // m_count_cache.clear();

    return std::make_tuple(count_xz, count_yz, count_z);
}

template <typename ArrowType>
std::vector<size_t> VPTree::hash_query_keys(
    const std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>>& data,
    std::vector<std::string> column_names) const {
    int num_rows = data.empty() ? 0 : data[0]->length();
    std::vector<size_t> row_hashes(num_rows, 0);

    size_t colnames_hash = boost::hash_range(column_names.begin(), column_names.end());
    ;

    for (auto j = 0; j < data.size(); ++j) {
        for (int i = 0; i < num_rows; ++i) {
            auto value = data[j]->Value(i);

            boost::hash_combine(row_hashes[i], value);
        }
    }

    for (int i = 0; i < num_rows; ++i) {
        boost::hash_combine(row_hashes[i], colnames_hash);
    }

    return row_hashes;
}

VectorXi VPTree::count_ball_unconditional(const DataFrame& test_df,
                                          const VectorXd& eps,
                                          std::vector<bool>& is_discrete_column) const {
    test_df.raise_has_columns(m_column_names);

    auto n_rows = test_df->num_rows();
    VectorXi count_n(n_rows);

    switch (m_datatype->id()) {
        case Type::FLOAT: {
            auto test = test_df.downcast_vector<arrow::FloatType>();
            HybridChebyshevDistance<arrow::FloatType> distance(test, is_discrete_column);

            auto hash_keys = hash_query_keys<arrow::FloatType>(test, test_df.column_names());

            for (int i = 0; i < n_rows; ++i) {
                auto key = hash_keys[i];

                boost::hash_combine(key, eps(i));

                auto it = m_count_cache_unconditional.find(key);
                if (it != m_count_cache_unconditional.end()) {
                    count_n(i) = it->second;
                    continue;  // Skip the query, use cached result
                }

                auto c = count_ball_unconditional_instance<arrow::FloatType>(i, eps(i), distance);

                count_n(i) = c;

                m_count_cache_unconditional[key] = c;
            }

            break;
        }
        default: {
            auto test = test_df.downcast_vector<arrow::DoubleType>();
            HybridChebyshevDistance<arrow::DoubleType> distance(test, is_discrete_column);

            auto hash_keys = hash_query_keys<arrow::DoubleType>(test, test_df.column_names());

            for (int i = 0; i < n_rows; ++i) {
                auto key = hash_keys[i];

                boost::hash_combine(key, eps(i));

                auto it = m_count_cache_unconditional.find(key);
                if (it != m_count_cache_unconditional.end()) {
                    count_n(i) = it->second;
                    continue;  // Skip the query, use cached result
                }

                auto c = count_ball_unconditional_instance<arrow::DoubleType>(i, eps(i), distance);

                count_n(i) = c;

                m_count_cache_unconditional[key] = c;
            }
        }
    }

    return count_n;
}

template <typename ArrowType>
std::pair<VectorXd, VectorXi> VPTree::query_instance(size_t i,
                                                     int k,
                                                     const HybridChebyshevDistance<ArrowType>& distance) const {
    using CType = typename ArrowType::c_type;

    NeighborQueue<ArrowType> neighborhood;

    std::pair<CType, std::vector<size_t>> neighborhood_star;

    CType distance_upper_bound = neighborhood_star.first = std::numeric_limits<CType>::infinity(), distance_neigh = 0;

    QueryQueue<ArrowType> query_nodes;
    CType min_distance = 0;

    query_nodes.push(QueryNode<ArrowType>{/*.node = */ m_root.get(),
                                          /*.min_distance = */ min_distance});

    while (!query_nodes.empty()) {
        auto& query = query_nodes.top();
        auto node = query.node;

        query_nodes.pop();

        // if (query.min_distance > distance_upper_bound) continue;

        std::vector<size_t> eval_neighbors(1, node->index);

        if (node->is_leaf) {
            eval_neighbors = node->leaf_indices;
        }

        auto num_neighbors = eval_neighbors.size();

        for (auto it_neigh = eval_neighbors.begin(), neigh_end = eval_neighbors.end(); it_neigh != neigh_end;
             ++it_neigh) {
            distance_neigh = distance.distance(*it_neigh, i);

            if (neighborhood.size() == static_cast<std::size_t>(k)) {
                if (distance_neigh < distance_upper_bound) {
                    neighborhood.pop();
                    neighborhood.push(std::make_pair(distance_neigh, *it_neigh));
                    if (neighborhood_star.first > neighborhood.top().first) {
                        neighborhood_star.second.clear();
                    }
                } else if (distance_neigh == distance_upper_bound) {
                    if (num_neighbors > static_cast<std::size_t>(m_leafsize)) {
                        neighborhood_star.second.reserve(neighborhood_star.second.size() +
                                                         std::distance(it_neigh, neigh_end));
                        neighborhood_star.second.insert(neighborhood_star.second.end(), it_neigh, neigh_end);

                        distance_upper_bound = neighborhood_star.first = distance_neigh;
                        break;
                    } else {
                        neighborhood_star.second.push_back(*it_neigh);
                        neighborhood_star.first = distance_neigh;
                    }
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

    auto k_hat = k + neighborhood_star.second.size();
    VectorXd distances(k);
    VectorXi indices(k_hat);

    std::copy(neighborhood_star.second.begin(),
              neighborhood_star.second.end(),
              indices.data() + (k_hat - neighborhood_star.second.size()));

    auto u = k - 1;
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

        auto num_neighbors = eval_neighbors.size();

        for (auto it_neigh = eval_neighbors.begin(), neigh_end = eval_neighbors.end(); it_neigh != neigh_end;
             ++it_neigh) {
            d_z = distance_xyz.distance_coords(*it_neigh, i, z_indices);

            if (d_z <= eps_value) {
                if (num_neighbors <= static_cast<std::size_t>(m_leafsize)) {
                    ++count_z;
                    if (distance_xyz.distance_coords(*it_neigh, i, x_index) <= eps_value) ++count_xz;
                    if (distance_xyz.distance_coords(*it_neigh, i, y_index) <= eps_value) ++count_yz;
                } else {
                    count_z += num_neighbors;
                    for (; it_neigh != neigh_end; ++it_neigh) {
                        if (distance_xyz.distance_coords(*it_neigh, i, x_index) <= eps_value) ++count_xz;
                        if (distance_xyz.distance_coords(*it_neigh, i, y_index) <= eps_value) ++count_yz;
                    }
                    break;
                }
            } else if (num_neighbors > static_cast<std::size_t>(m_leafsize))
                break;
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
int VPTree::count_ball_unconditional_instance(size_t i,
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

        auto num_neighbors = eval_neighbors.size();

        for (auto it_neigh = eval_neighbors.begin(), neigh_end = eval_neighbors.end(); it_neigh != neigh_end;
             ++it_neigh) {
            distance_neigh = distance.distance(*it_neigh, i);

            if (distance_neigh <= eps_value) {
                if (num_neighbors <= static_cast<std::size_t>(m_leafsize)) {
                    ++count_n;
                } else {
                    count_n += num_neighbors;
                    break;
                }
            } else if (num_neighbors > static_cast<std::size_t>(m_leafsize))
                break;
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
