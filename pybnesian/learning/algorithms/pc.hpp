#ifndef PYBNESIAN_LEARNING_ALGORITHMS_PC_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_PC_HPP

#include <graph/generic_graph.hpp>
#include <learning/independences/independence.hpp>
#include <learning/algorithms/constraint.hpp>

using graph::PartiallyDirectedGraph, graph::ConditionalPartiallyDirectedGraph;
using learning::algorithms::SepList;
using learning::independences::IndependenceTest;
using util::ArcStringVector;
using learning::algorithms::direct_unshielded_triples_interactive, learning::algorithms::vstructure;

namespace learning::algorithms {

class PC {
public:
    PartiallyDirectedGraph estimate(const IndependenceTest& test,
                                    const std::vector<std::string>& nodes,
                                    const ArcStringVector& arc_blacklist,
                                    const ArcStringVector& arc_whitelist,
                                    const EdgeStringVector& edge_blacklist,
                                    const EdgeStringVector& edge_whitelist,
                                    double alpha,
                                    bool use_sepsets,
                                    double ambiguous_threshold,
                                    bool allow_bidirected,
                                    int verbose) const;

    ConditionalPartiallyDirectedGraph estimate_conditional(const IndependenceTest& test,
                                                           const std::vector<std::string>& nodes,
                                                           const std::vector<std::string>& interface_nodes,
                                                           const ArcStringVector& arc_blacklist,
                                                           const ArcStringVector& arc_whitelist,
                                                           const EdgeStringVector& edge_blacklist,
                                                           const EdgeStringVector& edge_whitelist,
                                                           double alpha,
                                                           bool use_sepsets,
                                                           double ambiguous_threshold,
                                                           bool allow_bidirected,
                                                           int verbose) const;

    SepList compute_sepsets_of_size(PartiallyDirectedGraph& g,
                                    const IndependenceTest& test,
                                    const ArcStringVector& varc_blacklist,
                                    const ArcStringVector& varc_whitelist,
                                    const EdgeStringVector& vedge_blacklist,
                                    const EdgeStringVector& vedge_whitelist,
                                    int sepset_size) const;

    PartiallyDirectedGraph apply_adjacency_search(PartiallyDirectedGraph& pdag, const IndependenceTest& test,
                                    const ArcStringVector& arc_blacklist,
                                    const ArcStringVector& arc_whitelist,
                                    const EdgeStringVector& edge_blacklist,
                                    const EdgeStringVector& edge_whitelist,
                                    double alpha) const;

    std::vector<vstructure> compute_v_structures(PartiallyDirectedGraph& pdag,
                                                                  const IndependenceTest& test,
                                                                  double alpha,
                                                                  const std::optional<SepSet>& sepset,
                                                                  bool use_sepsets,
                                                                  double ambiguous_threshold) {
                
        auto vstructures = direct_unshielded_triples_interactive(pdag, test, alpha, sepset, use_sepsets, ambiguous_threshold);
        return vstructures;
    }
    PartiallyDirectedGraph estimate_from_initial_pdag(PartiallyDirectedGraph& pdag, const IndependenceTest& test,
                                    const ArcStringVector& arc_blacklist,
                                    const ArcStringVector& arc_whitelist,
                                    const EdgeStringVector& edge_blacklist,
                                    const EdgeStringVector& edge_whitelist,
                                    double alpha,
                                    double ambiguous_threshold,
                                    int phase_number) const;

};

}  // namespace learning::algorithms

#endif  // PYBNESIAN_LEARNING_ALGORITHMS_PC_HPP