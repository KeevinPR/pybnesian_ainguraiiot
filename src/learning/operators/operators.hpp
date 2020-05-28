#include <queue>
#include <Eigen/Dense>
#include <models/BayesianNetwork.hpp>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::Matrix, Eigen::Dynamic;
using MatrixXb = Matrix<bool, Dynamic, Dynamic>;

using models::BayesianNetwork, models::BayesianNetworkType;
using graph::arc_vector;

namespace learning::operators {

    

    template<typename Model>
    class Operator {
    public:
        Operator(double delta) : m_delta(delta) {}

        virtual void apply(Model& m) = 0;
        
        double delta() { return m_delta; }
    private:
        double m_delta;
    };

    template<typename Model>
    class AddArc : public Operator<Model> {
    public:
        AddArc(typename Model::node_descriptor source, 
               typename Model::node_descriptor dest,
               double delta) : m_source(source), m_dest(dest), Operator<Model>(delta) {}
        

        void apply(Model& m) override {
            m.add_edge(m_source, m_dest);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model>
    class RemoveArc : public Operator<Model> {
    public:
        RemoveArc(typename Model::node_descriptor source, 
                  typename Model::node_descriptor dest,
                  double delta) : m_source(source), m_dest(dest), Operator<Model>(delta) {}
        
        void apply(Model& m) override {
            m.remove_edge(m_source, m_dest);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model>
    class FlipArc : public Operator<Model> {
    public:
        FlipArc(typename Model::node_descriptor source, 
                  typename Model::node_descriptor dest,
                  double delta) : m_source(source), m_dest(dest), Operator<Model>(delta){}

       
        void apply(Model& m) override {
            m.remove_edge(m_source, m_dest);
            m.add_edge(m_dest, m_source);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model, typename Score>
    class ArcOperatorsType {
    public:
        ArcOperatorsType(const DataFrame& df, const Model& model, arc_vector whitelist, arc_vector blacklist);

        void cache_scores(const Model& m);
        void update(Model& model, std::unique_ptr<Operator<Model>> new_op);
        void arc_update(Model& model, typename Model::node_descriptor dest_node);
        std::unique_ptr<Operator<Model>> find_max(Model& m);

    private:
        MatrixXd delta;
        std::vector<int> sorted_idx;
        MatrixXb valid_op;
        VectorXd local_score;
        const DataFrame& df;
    };

    template<typename Model, typename Score>
    ArcOperatorsType<Model, Score>::ArcOperatorsType(const DataFrame& df, const Model& model, arc_vector whitelist, arc_vector blacklist) :
                                                    delta(),
                                                    sorted_idx(),
                                                    valid_op(model.num_nodes(), model.num_nodes()), 
                                                    local_score(model.num_nodes()), 
                                                    df(df)
    {
        auto nnodes = model.num_nodes();
        auto val_ptr = valid_op.data();

        std::fill(val_ptr, val_ptr + nnodes*nnodes, true);

        auto indices = model.indices();
        auto valid_ops = (nnodes * nnodes) - 2*whitelist.size() - blacklist.size() - nnodes;

        sorted_idx.reserve(valid_ops);

        for(auto whitelist_edge : whitelist) {
            auto source_index = indices[whitelist_edge.first];
            auto dest_index = indices[whitelist_edge.second];

            valid_op(source_index, dest_index) = false;
            valid_op(dest_index, source_index) = false;
            delta(source_index, dest_index) = std::numeric_limits<double>::lowest();
            delta(dest_index, source_index) = std::numeric_limits<double>::lowest();
        }
        
        for(auto blacklist_edge : blacklist) {
            auto source_index = indices[blacklist_edge.first];
            auto dest_index = indices[blacklist_edge.second];

            valid_op(source_index, dest_index) = false;
            delta(source_index, dest_index) = std::numeric_limits<double>::lowest();
        }

        for (auto i = 0; i < model.num_nodes(); ++i) {
            valid_op(i, i) = false;
            delta(i, i) = std::numeric_limits<double>::lowest();
        }

        for (auto i = 0; i < nnodes; ++i) {
            for (auto j = 0; j < nnodes; ++j) {
                if (valid_op(i, j)) {
                    sorted_idx.push_back(i + j * nnodes);
                }
            }
        }

    }

    template<typename Model, typename Score>
    void ArcOperatorsType<Model, Score>::cache_scores(const Model& m) {
        std::cout << "valid_op:" << std::endl;
        std::cout << valid_op << std::endl;
        
        for (int i = 0; i < m.num_nodes(); ++i) {
            auto parents = m.get_parent_indices(i);
            local_score(i) = Score::local_score(df, i, parents);
        }

        for (auto dest = 0; dest < m.num_nodes(); ++dest) {
            std::vector<int> new_parents_dest = m.get_parent_indices(dest);
            
            for (auto source = 0; source < m.num_nodes(); ++source) {
                if(valid_op(source, dest)) {

                    if (m.has_edge(source, dest)) {
                        std::iter_swap(std::find(new_parents_dest.begin(), new_parents_dest.end(), source), new_parents_dest.end() - 1);
                        double d = Score::local_score(df, dest, new_parents_dest.begin(), new_parents_dest.end() - 1) - local_score(dest);
                        delta(source, dest) = d;
                    } else if (m.has_edge(dest, source)) {
                        auto new_parents_source = m.get_parent_indices(source);
                        std::iter_swap(std::find(new_parents_source.begin(), new_parents_source.end(), dest), new_parents_source.end() - 1);
                        
                        new_parents_dest.push_back(source);
                        double d = Score::local_score(df, source, new_parents_source.begin(), new_parents_source.end() - 1) + 
                                   Score::local_score(df, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                   - local_score(source) - local_score(dest);
                        new_parents_dest.pop_back();
                        delta(dest, source) = d;
                    } else {
                        new_parents_dest.push_back(source);
                        double d = Score::local_score(df, dest, new_parents_dest) - local_score(dest);
                        new_parents_dest.pop_back();
                        delta(source, dest) = d;
                    }
                }
            }
        }

    }

    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> ArcOperatorsType<Model, Score>::find_max(Model& m) {

        auto delta_ptr = delta.data();

        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(); it != sorted_idx.end(); ++it) {
            auto idx = *it;
            auto source = idx % m.num_nodes();
            auto dest = idx / m.num_nodes();


            if(m.has_edge(source, dest)) {
                return std::make_unique<RemoveArc>(m.node(source), m.node(dest), delta(source, dest));
            } else if (m.has_edge(dest, source) && can_flip_edge(m, m.node(dest), m.node(source))) {
                return std::make_unique<FlipArc>(m.node(dest), m.node(source), delta(dest, source));
            } else if (can_add_edge(m, m.node(source), m.node(dest))) {
                return std::make_unique<AddArc>(m.node(source), m.node(dest), delta(source, dest));
            }

        }
    }

    template<typename Model>
    bool can_add_edge(Model& m, typename Model::node_descriptor source, typename Model::node_descriptor dest) {
        if (m.num_parents(source) == 0 || m.num_children(dest) == 0 || !m.has_path(dest, source)) {
            return true;
        }

        return false;
    }

    template<typename Model>
    bool can_flip_edge(Model& m, typename Model::node_descriptor source, typename Model::node_descriptor dest) {
        if (m.num_parents(dest) == 0 || m.num_children(source) == 0) {
            return true;
        } else {
            m.remove_edge(source, dest);
            bool has_path = m.has_path(source, dest);
            m.add_edge(source, dest);
            if (has_path) {
                return false;
            } else {
                return true;
            }
        }
    }

    template<typename Model, typename Score>
    void ArcOperatorsType<Model, Score>::update(Model& model, std::unique_ptr<Operator<Model>> op) {
    
    }


    template<typename Model, typename Score>
    void ArcOperatorsType<Model, Score>::arc_update(Model& model, typename Model::node_descriptor dest_node) {

        auto parents = model.get_parent_indices(dest_node);
        auto dest_idx = model.index(dest_node);
        local_score(dest_idx) = Score::local_score(df, dest_idx, parents);
        
        for (int i = 0; i < model.num_nodes(); ++i) {
            if (valid_op(i, dest_idx)) {

                if (model.has_edge(i, dest_idx)) {
                    std::iter_swap(std::find(parents.begin(), parents.end(), i), parents.end() - 1);
                    double d = Score::local_score(df, dest_idx, parents.begin(), parents.end() - 1) - local_score(dest_idx);
                    delta(i, dest_idx) = d;
                } else if (model.has_edge(dest_idx, i)) {
                    auto new_parents_i = model.get_parent_indices(source);
                    std::iter_swap(std::find(new_parents_i.begin(), new_parents_i.end(), dest_idx), new_parents_i.end() - 1);
                        
                    parents.push_back(i);
                    double d = Score::local_score(df, i, new_parents_i.begin(), new_parents_i.end() - 1) + 
                                Score::local_score(df, dest_idx, parents.begin(), parents.end()) 
                                - local_score(i) - local_score(dest_idx);
                    parents.pop_back();
                    delta(dest_idx, i) = d;
                } else {
                    parents.push_back(i);
                    double d = Score::local_score(df, dest_idx, parents) - local_score(dest_idx);
                    parents.pop_back();
                    delta(i, dest_idx) = d;
                }

            }
        }
    }

    template<typename Model, typename Score>
    struct default_operator {};

    template<BayesianNetworkType bn_type, typename Score>
    struct default_operator<BayesianNetwork<bn_type>, Score> {
        using default_operator_t = ArcOperatorsType<BayesianNetwork<bn_type>, Score>;
    };
    

    // template<typename Model, typename Score>
    // class OperatorPool {
    // public:

    //     OperatorPool(std::vector<std::unique_ptr<OperatorType<Model>>>&& ops) : operator_types(std::move(ops)) {}

    //     void cache_scores(const Model& m);
    //     void update_scores(const Model& m, std::unique_ptr<Operator<Model>> new_op);
    //     std::unique_ptr<Operator<Model>> find_max(const Model& m);

    // private:
    //     std::vector<std::unique_ptr<OperatorType<Model>>> operator_types;
    // };


    // template<typename Model, typename Score>
    // class DefaultOperatorPool {
    // public:
    //     using default_operator_t = typename default_operator<Model, Score>::default_operator_t;

    //     DefaultOperatorPool(int n_nodes) : operator_type(n_nodes) {}

    //     void cache_scores(const Model& m);
    //     void update_scores(const Model& m, std::unique_ptr<Operator<Model>> new_op);
    //     std::unique_ptr<Operator<Model>> find_max(const Model& m);

    // private:
    //     default_operator_t operator_type;
    // };



}