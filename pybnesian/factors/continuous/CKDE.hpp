#ifndef PYBNESIAN_FACTORS_CONTINUOUS_CKDE_HPP
#define PYBNESIAN_FACTORS_CONTINUOUS_CKDE_HPP

#include <random>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <dataset/dataset.hpp>
#include <factors/factors.hpp>
#include <factors/discrete/DiscreteAdaptator.hpp>
#include <kde/BandwidthSelector.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <kde/KDE.hpp>
#include <util/math_constants.hpp>

namespace py = pybind11;
namespace pyarrow = arrow::py;
using dataset::DataFrame;
using Eigen::VectorXd, Eigen::VectorXi;
using factors::FactorType, factors::discrete::DiscreteAdaptator;
using kde::KDE, kde::BandwidthSelector, kde::NormalReferenceRule, kde::UnivariateKDE, kde::MultivariateKDE;

namespace factors::continuous {

class CKDEType : public FactorType {
public:
    CKDEType(const CKDEType&) = delete;
    void operator=(const CKDEType&) = delete;

    static std::shared_ptr<CKDEType> get() {
        static std::shared_ptr<CKDEType> singleton = std::shared_ptr<CKDEType>(new CKDEType);
        return singleton;
    }

    static CKDEType& get_ref() {
        static CKDEType& ref = *CKDEType::get();
        return ref;
    }

    std::shared_ptr<Factor> new_factor(const BayesianNetworkBase&,
                                       const std::string&,
                                       const std::vector<std::string>&,
                                       py::args = py::args{},
                                       py::kwargs = py::kwargs{}) const override;
    std::shared_ptr<Factor> new_factor(const ConditionalBayesianNetworkBase&,
                                       const std::string&,
                                       const std::vector<std::string>&,
                                       py::args = py::args{},
                                       py::kwargs = py::kwargs{}) const override;

    std::string ToString() const override { return "CKDEFactor"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<CKDEType> __setstate__(py::tuple&) { return CKDEType::get(); }

private:
    CKDEType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

class CKDE : public Factor {
public:
    using FactorTypeClass = CKDEType;

    CKDE() = default;
    CKDE(std::string variable, std::vector<std::string> evidence)
        : CKDE(variable, evidence, std::make_shared<NormalReferenceRule>()) {}
    CKDE(std::string variable, std::vector<std::string> evidence, std::shared_ptr<BandwidthSelector> b_selector)
        : Factor(variable, evidence),
          m_variables(),
          m_fitted(false),
          m_bselector(b_selector),
          m_training_type(arrow::float64()),
          m_joint(),
          m_marg() {
        if (b_selector == nullptr) throw std::runtime_error("Bandwidth selector procedure must be non-null.");

        m_variables.reserve(evidence.size() + 1);
        m_variables.push_back(variable);
        for (auto it = evidence.begin(); it != evidence.end(); ++it) {
            m_variables.push_back(*it);
        }

        m_joint = KDE(m_variables, b_selector);
        if (!this->evidence().empty()) {
            m_marg = KDE(this->evidence(), b_selector);
        }
    }

    std::shared_ptr<FactorType> type() const override { return CKDEType::get(); }

    FactorType& type_ref() const override { return CKDEType::get_ref(); }

    std::shared_ptr<arrow::DataType> data_type() const override {
        check_fitted();
        return m_training_type;
    }

    int num_instances() const {
        check_fitted();
        return N;
    }

    KDE& kde_joint() {
        check_fitted();
        return m_joint;
    }
    KDE& kde_marg() {
        check_fitted();
        return m_marg;
    }

    bool fitted() const override { return m_fitted; }

    std::shared_ptr<BandwidthSelector> bandwidth_type() const { return m_bselector; }

    void fit(const DataFrame& df) override;
    VectorXd logl(const DataFrame& df) const override;
    double slogl(const DataFrame& df) const override;

    Array_ptr sample(int n,
                     const DataFrame& evidence_values,
                     unsigned int seed = std::random_device{}()) const override;

    VectorXd cdf(const DataFrame& df) const;

    std::string ToString() const override;

    py::tuple __getstate__() const override;
    static CKDE __setstate__(py::tuple& t);
    static CKDE __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    void check_fitted() const {
        if (!fitted()) throw std::invalid_argument("CKDE factor not fitted.");
    }
    template <typename ArrowType>
    void _fit(const DataFrame& df);

    template <typename ArrowType>
    VectorXd _logl(const DataFrame& df) const;

    template <typename ArrowType>
    double _slogl(const DataFrame& df) const;

    template <typename ArrowType>
    Array_ptr _sample(int n, const DataFrame& evidence_values, unsigned int seed) const;

    template <typename ArrowType>
    Array_ptr _sample_multivariate(int n, const DataFrame& evidence_values, unsigned int seed) const;

    template <typename ArrowType>
    VectorXi _sample_indices_multivariate(Matrix<typename ArrowType::c_type, Dynamic, 1>& random_prob,
                                          const DataFrame& evidence_values,
                                          int n) const;

    template <typename ArrowType, typename KDEType>
    Matrix<int, Dynamic, 1> _sample_indices_from_weights(typename ArrowType::c_type* random_prob, typename ArrowType::c_type* test_buffer, int n) const;

    template <typename ArrowType>
    VectorXd _cdf(const DataFrame& df) const;

    template <typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> _cdf_univariate(typename ArrowType::c_type* test_buffer, uint m) const;

    template <typename ArrowType, typename KDEType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> _cdf_multivariate(typename ArrowType::c_type* variable_test_buffer, typename ArrowType::c_type* evidence_test_buffer, uint m) const;

    template <typename ArrowType>
    py::tuple __getstate__() const;

    std::vector<std::string> m_variables;
    bool m_fitted;
    std::shared_ptr<BandwidthSelector> m_bselector;
    std::shared_ptr<arrow::DataType> m_training_type;
    size_t N;
    KDE m_joint;
    KDE m_marg;
};

template <typename ArrowType>
void CKDE::_fit(const DataFrame& df) {
    m_joint.fit(df);
    N = m_joint.num_instances();

    if (!this->evidence().empty()) {
        auto& joint_bandwidth = m_joint.bandwidth();
        auto d = m_variables.size();
        auto marg_bandwidth = joint_bandwidth.bottomRightCorner(d - 1, d - 1);

        m_marg.fit<ArrowType>(marg_bandwidth, &m_joint.training_raw<ArrowType>()[N], m_joint.data_type(), N);
    }
}

template <typename ArrowType>
VectorXd CKDE::_logl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto logl_joint = m_joint.logl_buffer<ArrowType>(df);
    auto combined_bitmap = df.combined_bitmap(m_variables);
    auto m = df->num_rows();
    if (combined_bitmap) m = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

    if (!this->evidence().empty()) {
        VectorType logl_marg(m);
        if (combined_bitmap)
            logl_marg = m_marg.logl_buffer<ArrowType>(df, combined_bitmap);
        else
            logl_marg = m_marg.logl_buffer<ArrowType>(df);
        Kernel<CType>::instance().substract_vectors(logl_joint.data(), logl_marg.data(), m);
    }

    if (combined_bitmap) {
        VectorType read_data(m);
        std::memcpy(read_data.data(), logl_joint.data(), m*sizeof(CType));

        auto bitmap_data = combined_bitmap->data();

        VectorXd res(df->num_rows());

        for (int i = 0, k = 0; i < df->num_rows(); ++i) {
            if (util::bit_util::GetBit(bitmap_data, i)) {
                res(i) = static_cast<double>(read_data[k++]);
            } else {
                res(i) = util::nan<double>;
            }
        }

        return res;
    } else {
        VectorType read_data(df->num_rows());
        std::memcpy(read_data.data(), logl_joint.data(), df->num_rows()*sizeof(CType));

        if constexpr (!std::is_same_v<CType, double>)
            return read_data.template cast<double>();
        else
            return read_data;
    }
}

template <typename ArrowType>
double CKDE::_slogl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto logl_joint = m_joint.logl_buffer<ArrowType>(df);
    auto combined_bitmap = df.combined_bitmap(m_variables);
    auto m = df->num_rows();
    if (combined_bitmap) m = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

    if (!this->evidence().empty()) {
        VectorType logl_marg(m);
        if (combined_bitmap)
            logl_marg = m_marg.logl_buffer<ArrowType>(df, combined_bitmap);
        else
            logl_marg = m_marg.logl_buffer<ArrowType>(df);
        Kernel<CType>::instance().substract_vectors(logl_joint.data(), logl_marg.data(), m);
    }

    CType result = -1;
    Kernel<CType>::instance().sum1d(logl_joint.data(), m, &result);
    return static_cast<double>(result);
}

template <typename ArrowType>
Array_ptr CKDE::_sample(int n, const DataFrame& evidence_values, unsigned int seed) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    if (this->evidence().empty()) {
        arrow::NumericBuilder<ArrowType> builder;
        RAISE_STATUS_ERROR(builder.Resize(n));
        std::mt19937 rng{seed};
        std::uniform_int_distribution<> uniform(0, N - 1);

        std::normal_distribution<CType> normal(0, std::sqrt(m_joint.bandwidth()(0, 0)));
        VectorType training_data(N);
        std::memcpy(training_data.data(), m_joint.training_raw<ArrowType>(), N*sizeof(CType));

        for (auto i = 0; i < n; ++i) {
            auto index = uniform(rng);
            builder.UnsafeAppend(training_data(index) + normal(rng));
        }

        Array_ptr out;
        RAISE_STATUS_ERROR(builder.Finish(&out));
        return out;
    } else {
        return _sample_multivariate<ArrowType>(n, evidence_values, seed);
    }
}

template <typename ArrowType>
Array_ptr CKDE::_sample_multivariate(int n, const DataFrame& evidence_values, unsigned int seed) const {
    using CType = typename ArrowType::c_type;
    using ArrowArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using VectorType = Matrix<CType, Dynamic, 1>;
    using MatrixType = Matrix<CType, Dynamic, Dynamic>;

    const auto& e = this->evidence();

    if (!evidence_values.has_columns(e)) throw std::domain_error("Evidence values not present for sampling.");

    VectorType random_prob(n);
    std::mt19937 rng{seed};
    std::uniform_real_distribution<CType> uniform(0, 1);
    for (auto i = 0; i < n; ++i) {
        random_prob(i) = uniform(rng);
    }

    VectorXi sample_indices = _sample_indices_multivariate<ArrowType>(random_prob, evidence_values, n);

    const auto& bandwidth = m_joint.bandwidth();
    const auto& marg_bandwidth = m_marg.bandwidth();

    auto cholesky = marg_bandwidth.llt();
    auto matrixL = cholesky.matrixL();

    auto d = e.size();
    MatrixXd inverseL = MatrixXd::Identity(d, d);

    // Solves and saves the result in inverseL
    matrixL.solveInPlace(inverseL);
    auto R = inverseL * bandwidth.bottomLeftCorner(d, 1);
    auto cond_var = bandwidth(0, 0) - R.squaredNorm();
    auto transform = (R.transpose() * inverseL).transpose().template cast<CType>();

    MatrixType training_dataset(N, m_variables.size());
    std::memcpy(training_dataset.data(), m_joint.training_raw<ArrowType>(), N*m_variables.size()*sizeof(CType));
    MatrixType evidence_substract(n, e.size());
    for (size_t j = 0; j < e.size(); ++j) {
        auto evidence = evidence_values->GetColumnByName(e[j]);
        auto dwn_evidence = std::static_pointer_cast<ArrowArrayType>(evidence);
        auto raw_values = dwn_evidence->raw_values();
        for (auto i = 0; i < n; ++i) {
            evidence_substract(i, j) = raw_values[i] - training_dataset(sample_indices(i), j + 1);
        }
    }

    auto cond_mean = (evidence_substract * transform).eval();

    std::normal_distribution<CType> normal(0, std::sqrt(cond_var));
    arrow::NumericBuilder<ArrowType> builder;
    RAISE_STATUS_ERROR(builder.Resize(n));

    for (auto i = 0; i < n; ++i) {
        cond_mean(i) += training_dataset(sample_indices(i), 0) + normal(rng);
    }

    RAISE_STATUS_ERROR(builder.AppendValues(cond_mean.data(), n));

    Array_ptr out;
    RAISE_STATUS_ERROR(builder.Finish(&out));

    return out;
}

template <typename ArrowType>
VectorXi CKDE::_sample_indices_multivariate(Matrix<typename ArrowType::c_type, Dynamic, 1>& random_prob,
                                            const DataFrame& evidence_values,
                                            int n) const {
    using CType = typename ArrowType::c_type;
    using ArrowArray = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using MatrixType = Matrix<CType, Dynamic, Dynamic>;

    const auto& e = this->evidence();
    MatrixType test_matrix(n, e.size());

    for (size_t i = 0; i < e.size(); ++i) {
        auto evidence = evidence_values->GetColumnByName(e[i]);
        auto dwn_evidence = std::static_pointer_cast<ArrowArray>(evidence);
        auto raw_evidence = dwn_evidence->raw_values();
        std::memcpy(test_matrix.data() + i * n, raw_evidence, sizeof(CType) * n);
    }

    Matrix<int, Dynamic, 1> indices(n);
    if (e.size() == 1)
        indices = _sample_indices_from_weights<ArrowType, UnivariateKDE>(random_prob.data(), test_matrix.data(), n);
    else 
        indices = _sample_indices_from_weights<ArrowType, MultivariateKDE>(random_prob.data(), test_matrix.data(), n);
    VectorXi res(n);
    for(int i = 0; i < n; ++i)
        res[i] = indices[i];
    return res;
}

template <typename ArrowType, typename KDEType>
Matrix<int, Dynamic, 1> CKDE::_sample_indices_from_weights(typename ArrowType::c_type* random_prob, typename ArrowType::c_type* test_buffer, int n) const {
    using CType = typename ArrowType::c_type;
    Kernel<CType> kernels = Kernel<CType>::instance();

    Matrix<int, Dynamic, 1> res(n);

    for(int i = 0; i < n; ++i)
        res.data()[i] = N - 1;

    auto allocated_m = std::min(n, 64);
    auto iterations = static_cast<int>(std::ceil(static_cast<double>(n) / static_cast<double>(allocated_m)));

    CType* out = (CType*)malloc(N * allocated_m * sizeof(CType));
    CType* total_sum = (CType*)malloc(allocated_m*sizeof(CType));

    for (auto i = 0; i < (iterations - 1); ++i) {
        KDEType::template execute_logl_mat<ArrowType>(m_marg.training_raw<ArrowType>(),
                                                      N,
                                                      test_buffer,
                                                      n,
                                                      i * allocated_m,
                                                      allocated_m,
                                                      this->evidence().size(),
                                                      m_marg.cholesky_raw<ArrowType>(),
                                                      m_marg.lognorm_const(),
                                                    //   tmp,
                                                      out);
        kernels.exp_elementwise(out, N * allocated_m);
        kernels.accum_sum_cols(out, N, allocated_m, total_sum);
        kernels.normalize_accum_sum_mat_cols(out, N, total_sum, N - 1, allocated_m);
        kernels.find_random_indices(out, N, i * allocated_m, random_prob, res.data(), N - 1, allocated_m);
    }
    auto offset = (iterations - 1) * allocated_m;
    auto remaining_m = n - offset;

    KDEType::template execute_logl_mat<ArrowType>(m_marg.training_raw<ArrowType>(),
                                                  N,
                                                  test_buffer,
                                                  n,
                                                  offset,
                                                  remaining_m,
                                                  this->evidence().size(),
                                                  m_marg.cholesky_raw<ArrowType>(),
                                                  m_marg.lognorm_const(),
                                                //   tmp,
                                                  out);
    kernels.exp_elementwise(out, N * remaining_m);
    kernels.accum_sum_cols(out, N, remaining_m, total_sum);
    kernels.normalize_accum_sum_mat_cols(out, N, total_sum, N - 1, remaining_m);
    kernels.find_random_indices(out, N, offset, random_prob, res.data(), N - 1, remaining_m);

    free(total_sum);
    free(out);

    return res;
}

template <typename ArrowType>
VectorXd CKDE::_cdf(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
    auto m = test_matrix->rows();

    VectorType res_tmp(m);
    if (this->evidence().empty()) {
        res_tmp = _cdf_univariate<ArrowType>(test_matrix->data(), m);
    } else {
        if (this->evidence().size() == 1)
            res_tmp = _cdf_multivariate<ArrowType, UnivariateKDE>(test_matrix->data(), test_matrix->data() + m, m);
        else
            res_tmp = _cdf_multivariate<ArrowType, MultivariateKDE>(test_matrix->data(), test_matrix->data() + m, m);
    }

    if (df.null_count(m_variables) == 0) {
        VectorType read_data(df->num_rows());
        std::memcpy(read_data.data(), res_tmp.data(), df->num_rows()*sizeof(CType));
        if constexpr (!std::is_same_v<CType, double>)
            return read_data.template cast<double>();
        else
            return read_data;
    } else {
        auto bitmap = df.combined_bitmap(m_variables);
        auto bitmap_data = bitmap->data();
        VectorXd res(df->num_rows());

        for (int i = 0, k = 0; i < df->num_rows(); ++i) {
            if (util::bit_util::GetBit(bitmap_data, i)) {
                res(i) = static_cast<double>(res_tmp[k++]);
            } else {
                res(i) = util::nan<double>;
            }
        }

        return res;
    }
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> CKDE::_cdf_univariate(typename ArrowType::c_type* test_buffer, uint m) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    Kernel<CType> kernels = Kernel<CType>::instance();

    auto allocated_m = std::min(m, 64u);
    auto iterations = std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m));

    CType* mu = (CType*)malloc(N*allocated_m*sizeof(CType));

    VectorType res(m);
    for (auto i = 0; i < (iterations - 1); ++i) {
        kernels.univariate_normal_cdf(m_joint.training_raw<ArrowType>(),
                                        N,
                                        test_buffer,
                                        static_cast<unsigned int>(i * allocated_m),
                                        static_cast<CType>(1.0 / std::sqrt(m_joint.bandwidth()(0, 0))),
                                        static_cast<CType>(1.0 / N),
                                        mu,
                                        allocated_m);
        kernels.sum_cols_offset(mu, N, allocated_m, res.data(), i * allocated_m);
    }

    auto offset = (iterations - 1) * allocated_m;
    auto remaining_m = m - offset;

    kernels.univariate_normal_cdf(m_joint.training_raw<ArrowType>(),
                                    N,
                                    test_buffer,
                                    offset,
                                    static_cast<CType>(1.0 / std::sqrt(m_joint.bandwidth()(0, 0))),
                                    static_cast<CType>(1.0 / N),
                                    mu,
                                    remaining_m);
    kernels.sum_cols_offset(mu, N, remaining_m, res.data(), offset);

    return res;
}

template <typename ArrowType, typename KDEType>
Matrix<typename ArrowType::c_type, Dynamic, 1> CKDE::_cdf_multivariate(typename ArrowType::c_type* variable_test_buffer, typename ArrowType::c_type* evidence_test_buffer, uint m) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    Kernel<CType> kernels = Kernel<CType>::instance();

    const auto& bandwidth = m_joint.bandwidth();
    const auto& marg_bandwidth = m_marg.bandwidth();

    auto cholesky = marg_bandwidth.llt();
    auto matrixL = cholesky.matrixL();

    auto d = this->evidence().size();
    MatrixXd inverseL = MatrixXd::Identity(d, d);

    matrixL.solveInPlace(inverseL);
    auto R = inverseL * bandwidth.bottomLeftCorner(d, 1);
    auto cond_var = bandwidth(0, 0) - R.squaredNorm();
    auto transform = (R.transpose() * inverseL).template cast<CType>().eval();

    auto allocated_m = std::min(m, 64u);
    auto iterations = static_cast<int>(std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m)));

    auto new_lognorm_marg = m_marg.lognorm_const() + std::log(N);

    auto tmp_mat_size = this->evidence().size() * (N>allocated_m?N:allocated_m);

    VectorType res(m);

    CType* tmp = (CType*)malloc(tmp_mat_size*sizeof(CType));
    CType* W = (CType*)malloc(N*allocated_m*sizeof(CType));
    CType* sum_W = (CType*)malloc(allocated_m*sizeof(CType));
    CType* mu = (CType*)malloc(N*allocated_m*sizeof(CType));

    for (auto i = 0; i < (iterations - 1); ++i) {
        // Computes Weigths
        KDEType::template execute_logl_mat<ArrowType>(m_marg.training_raw<ArrowType>(),
                                                      N,
                                                      evidence_test_buffer,
                                                      m,
                                                      i * allocated_m,
                                                      allocated_m,
                                                      this->evidence().size(),
                                                      m_marg.cholesky_raw<ArrowType>(),
                                                      new_lognorm_marg,
                                                    //   tmp,
                                                      W);
        kernels.exp_elementwise(W, N * allocated_m);
        kernels.sum_cols_offset(W, N, allocated_m, sum_W, 0);

        // Computes conditional mu.
        KDEType::template execute_conditional_means<ArrowType>(m_joint.training_raw<ArrowType>(),
                                                               m_marg.training_raw<ArrowType>(),
                                                               N,
                                                               evidence_test_buffer,
                                                               m,
                                                               i * allocated_m,
                                                               allocated_m,
                                                               this->evidence().size(),
                                                               transform.data(),
                                                               tmp,
                                                               mu);
        kernels.normal_cdf(mu, N, variable_test_buffer, i * allocated_m, static_cast<CType>(1.0 / std::sqrt(cond_var)), N * allocated_m);
        kernels.product_elementwise(mu, W, N * allocated_m);
        kernels.sum_cols_offset(mu, N, allocated_m, res.data(), i * allocated_m);
        kernels.division_elementwise(res.data(), i * allocated_m, sum_W, allocated_m);
    }
    auto offset = (iterations - 1) * allocated_m;
    auto remaining_m = m - offset;

    // Computes Weigths
    KDEType::template execute_logl_mat<ArrowType>(m_marg.training_raw<ArrowType>(),
                                                  N,
                                                  evidence_test_buffer,
                                                  m,
                                                  offset,
                                                  remaining_m,
                                                  this->evidence().size(),
                                                  m_marg.cholesky_raw<ArrowType>(),
                                                  new_lognorm_marg,
                                                //   tmp,
                                                  W);
    kernels.exp_elementwise(W, N * remaining_m);
    kernels.sum_cols_offset(W, N, remaining_m, sum_W, 0);

    // Computes conditional mu.
    KDEType::template execute_conditional_means<ArrowType>(m_joint.training_raw<ArrowType>(),
                                                           m_marg.training_raw<ArrowType>(),
                                                           N,
                                                           evidence_test_buffer,
                                                           m,
                                                           offset,
                                                           remaining_m,
                                                           this->evidence().size(),
                                                           transform.data(),
                                                           tmp,
                                                           mu);
    kernels.normal_cdf(mu, N, variable_test_buffer, offset, static_cast<CType>(1.0 / std::sqrt(cond_var)), N * remaining_m);
    kernels.product_elementwise(mu, W, N * remaining_m);
    kernels.sum_cols_offset(mu, N, remaining_m, res.data(), offset);
    kernels.division_elementwise(res.data(), offset, sum_W, remaining_m);

    free(mu);
    free(sum_W);
    free(W);
    free(tmp);
    return res;
}

template <typename ArrowType>
py::tuple CKDE::__getstate__() const {
    py::tuple joint_tuple;
    if (m_fitted) {
        joint_tuple = m_joint.__getstate__();
    }

    return py::make_tuple(this->variable(), this->evidence(), m_fitted, joint_tuple);
}

// Fix const name: https://stackoverflow.com/a/15862594
struct HCKDEName {
    inline constexpr static auto* str = "HCKDE";
};

struct CKDEFitter {
    static bool fit(const std::shared_ptr<Factor>& factor, const DataFrame& df) {
        try {
            factor->fit(df);
            return true;
        } catch (util::singular_covariance_data& e) {
            return false;
        } catch (py::error_already_set& e) {
            auto t = py::module_::import("pybnesian").attr("SingularCovarianceData");
            if (e.matches(t)) {
                return false;
            } else {
                throw;
            }
        }
    }
};

using HCKDE = DiscreteAdaptator<CKDE, CKDEFitter, HCKDEName>;

}  // namespace factors::continuous

#endif  // PYBNESIAN_FACTORS_CONTINUOUS_CKDE_HPP
