#ifndef PYBNESIAN_LEARNING_SCORES_CV_LIKELIHOOD_HPP
#define PYBNESIAN_LEARNING_SCORES_CV_LIKELIHOOD_HPP

#include <dataset/crossvalidation_adaptator.hpp>
#include <learning/scores/scores.hpp>

using dataset::CrossValidation;
using learning::scores::Score, learning::scores::ScoreSPBN;
using factors::FactorType;
using models::BayesianNetworkBase, models::BayesianNetworkType, 
      models::SemiparametricBNBase;

namespace learning::scores {

    class CVLikelihood : public ScoreSPBN {
    public:
        CVLikelihood(const DataFrame& df, int k) : m_cv(df, k) {}
        CVLikelihood(const DataFrame& df, int k, unsigned int seed) : m_cv(df, k, seed) {}

        double local_score(const BayesianNetworkBase& model, int variable) const override {
            return local_score(model, model.name(variable));
        }

        double local_score(const BayesianNetworkBase& model, const std::string& variable) const override {
            auto parents = model.parents(variable);
            return local_score(model, variable, parents);
        }

        double local_score(const BayesianNetworkBase& model,
                           int variable,
                           const std::vector<int>& evidence) const override {
            std::vector<std::string> evidence_str;
            for (auto ev : evidence) {
                evidence_str.push_back(model.name(ev));
            }

            return local_score(model, model.name(variable), evidence_str);
        }

        double local_score(const BayesianNetworkBase& model,
                           const std::string& variable,
                           const std::vector<std::string>& evidence) const override;

        double local_score(FactorType variable_type,
                           const std::string& variable,
                           const std::vector<std::string>& evidence) const override;

        template<typename VarType, typename EvidenceIter>
        double local_score(FactorType variable_type,
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

        const CrossValidation& cv() { return m_cv; }

        std::string ToString() const override {
            return "CVLikelihood";
        }

        bool is_decomposable() const override {
            return true;
        }

        ScoreType type() const override {
            return ScoreType::PREDICTIVE_LIKELIHOOD;
        }

        bool compatible_bn(const BayesianNetworkBase& model) const override{
            return m_cv.data().has_columns(model.nodes());
        }

        bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
            return m_cv.data().has_columns(model.all_nodes());
        }
    private:
        CrossValidation m_cv;
    };

    using DynamicCVLikelihood = DynamicScoreAdaptator<CVLikelihood>;
}

#endif //PYBNESIAN_CV_LIKELIHOOD_HPP