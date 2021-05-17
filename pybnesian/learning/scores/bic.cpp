#include <learning/scores/bic.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/discrete/DiscreteFactor.hpp>

using factors::continuous::LinearGaussianCPDType, factors::continuous::LinearGaussianCPD;
using factors::discrete::DiscreteFactorType;

using models::GaussianNetworkType;

namespace learning::scores {

double BIC::bic_lineargaussian(const std::string& variable, const std::vector<std::string>& parents) const {
    MLE<LinearGaussianCPD> mle;

    auto mle_params = mle.estimate(m_df, variable, parents);

    if (mle_params.variance < util::machine_tol) {
        return -std::numeric_limits<double>::infinity();
    }

    auto rows = m_df.valid_rows(variable, parents);
    auto num_parents = parents.size();
    auto loglik = 0.5 * (1 + static_cast<double>(num_parents) - static_cast<double>(rows)) -
                  0.5 * rows * std::log(2 * util::pi<double>) - rows * 0.5 * std::log(mle_params.variance);

    return loglik - std::log(rows) * 0.5 * (num_parents + 2);
}

double BIC::bic_clg(const std::string& variable,
                    const std::vector<std::string>& discrete_parents,
                    const std::vector<std::string>& continuous_parents) const {
    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, discrete_parents);

    auto num_configs = cardinality.prod();
    auto slice_builders = factors::discrete::discrete_slice_indices(m_df, discrete_parents, strides, num_configs);

    MLE<LinearGaussianCPD> mle;

    double loglik = 0;

    auto num_continuous_parents = continuous_parents.size();

    for (auto i = 0; i < num_configs; ++i) {
        auto num_config_rows = slice_builders[i].length();
        if (num_config_rows > 0) {
            Array_ptr take_indices;
            RAISE_STATUS_ERROR(slice_builders[i].Finish(&take_indices));

            // Calling take() can be slower than fitting all the linear regressions at the same time (as bnlearn)
            auto df_filtered = m_df.loc(variable, continuous_parents).take(take_indices);

            auto num_valid_config = df_filtered.valid_rows(variable, continuous_parents);
            auto mle_params = mle.estimate(df_filtered, variable, continuous_parents);

            loglik += 0.5 * (1 + static_cast<double>(num_continuous_parents) - static_cast<double>(num_valid_config)) -
                      0.5 * num_valid_config * std::log(2 * util::pi<double>) -
                      num_valid_config * 0.5 * std::log(mle_params.variance);
        }
    }

    auto valid_rows = m_df.valid_rows(variable, discrete_parents, continuous_parents);

    return loglik - std::log(valid_rows) * 0.5 * num_configs * (num_continuous_parents + 2);
}

double BIC::bic_discrete(const std::string& variable, const std::vector<std::string>& parents) const {
    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, variable, parents);
    auto joint_counts = factors::discrete::joint_counts(m_df, variable, parents, cardinality, strides);

    auto parent_configurations = cardinality.tail(parents.size()).prod();

    double ll = 0;

    for (auto k = 0; k < parent_configurations; ++k) {
        auto offset = k * cardinality(0);

        int sum_configuration = 0;
        for (auto i = 0; i < cardinality(0); ++i) {
            sum_configuration += joint_counts(offset + i);
        }

        if (sum_configuration > 0) {
            auto inv_dbl_sum = static_cast<double>(1. / sum_configuration);

            for (auto i = 0; i < cardinality(0); ++i) {
                if (joint_counts(offset + i) > 0) {
                    auto dbl_count = static_cast<double>(joint_counts(offset + i));
                    ll += dbl_count * std::log(dbl_count * inv_dbl_sum);
                }
            }
        }
    }

    double sum_count = static_cast<double>(joint_counts.sum());
    return ll - std::log(sum_count) * 0.5 * (cardinality(0) - 1) * parent_configurations;
}

double BIC::local_score(const BayesianNetworkBase& model,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const {
    const auto& node_type = *model.node_type(variable);
    if (node_type == LinearGaussianCPDType::get_ref()) {
        std::vector<std::string> discrete_parents;
        std::vector<std::string> continuous_parents;

        for (const auto& p : parents) {
            if (*model.node_type(p) == DiscreteFactorType::get_ref()) {
                discrete_parents.push_back(p);
            } else {
                continuous_parents.push_back(p);
            }
        }

        if (discrete_parents.empty())
            return bic_lineargaussian(variable, parents);
        else
            return bic_clg(variable, discrete_parents, continuous_parents);
    }

    if (node_type == DiscreteFactorType::get_ref()) {
        return bic_discrete(variable, parents);
    }

    throw std::invalid_argument("Bayesian network type \"" + model.type_ref().ToString() +
                                "\" not valid for score BIC");
}

double BIC::local_score(const BayesianNetworkBase&,
                        const FactorType& node_type,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const {
    if (node_type == LinearGaussianCPDType::get_ref()) {
        return bic_lineargaussian(variable, parents);
    }

    if (node_type == DiscreteFactorType::get_ref()) {
        return bic_discrete(variable, parents);
    }

    throw std::invalid_argument("Node type \"" + node_type.ToString() + "\" not valid for score BIC");
}

}  // namespace learning::scores