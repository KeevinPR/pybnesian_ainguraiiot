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
#include <opencl/opencl_config.hpp>
#include <util/math_constants.hpp>

namespace py = pybind11;
namespace pyarrow = arrow::py;
using dataset::DataFrame;
using Eigen::VectorXd, Eigen::VectorXi;
using factors::FactorType, factors::discrete::DiscreteAdaptator;
using kde::KDE, kde::BandwidthSelector, kde::NormalReferenceRule, kde::UnivariateKDE, kde::MultivariateKDE;
using opencl::OpenCLConfig, opencl::OpenCL_kernel_traits;

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
    Matrix<typename ArrowType::c_type, Dynamic, 1> _cdf_univariate(typename ArrowType::c_type* test_buffer, int m) const;

    template <typename ArrowType, typename KDEType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> _cdf_multivariate(typename ArrowType::c_type* variable_test_buffer, typename ArrowType::c_type* evidence_test_buffer, int m) const;

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

        using CType = typename ArrowType::c_type;

        Matrix<CType, Dynamic, 1> training_mat(N * (d - 1));
        auto* training_buffer_raw = training_mat.data();
        auto* aux = m_joint.training_raw<ArrowType>();
        for(int i = 0; i < N * (d - 1); ++i){
            training_buffer_raw[i] = aux[N + i];
        }

        m_marg.fit<ArrowType>(marg_bandwidth, training_mat.data(), m_joint.data_type(), N);
    }
}

template <typename T>
void substract_vectors(T* v1, T* v2, uint m) {
    for(uint idx = 0; idx < m; ++idx)
        v1[idx] -= v2[idx];
}

template <typename ArrowType>
VectorXd CKDE::_logl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto logl_joint_tmp = m_joint.logl_buffer<ArrowType>(df);
    auto combined_bitmap = df.combined_bitmap(m_variables);
    auto m = df->num_rows();
    if (combined_bitmap) m = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

    if (!this->evidence().empty()) {
        VectorType logl_marg_tmp(m);
        if (combined_bitmap)
            logl_marg_tmp = m_marg.logl_buffer<ArrowType>(df, combined_bitmap);
        else
            logl_marg_tmp = m_marg.logl_buffer<ArrowType>(df);
        substract_vectors<CType>(logl_joint_tmp.data(), logl_marg_tmp.data(), m);
    }

    if (combined_bitmap) {
        VectorType read_data(m);
        for(int i = 0; i < m; ++i)
            read_data[i] = logl_joint_tmp.data()[i];

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
        for(int i = 0; i < df->num_rows(); ++i)
            read_data[i] = logl_joint_tmp.data()[i];

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

    auto logl_joint_tmp = m_joint.logl_buffer<ArrowType>(df);
    auto combined_bitmap = df.combined_bitmap(m_variables);
    auto m = df->num_rows();
    if (combined_bitmap) m = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

    auto& opencl = OpenCLConfig::get();

    if (!this->evidence().empty()) {
        VectorType logl_marg_tmp(m);
        if (combined_bitmap)
            logl_marg_tmp = m_marg.logl_buffer<ArrowType>(df, combined_bitmap);
        else
            logl_marg_tmp = m_marg.logl_buffer<ArrowType>(df);
        substract_vectors<CType>(logl_joint_tmp.data(), logl_marg_tmp.data(), m);
    }

    CType result;
    opencl.sum1d<ArrowType>(logl_joint_tmp.data(), m, &result);
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
        auto& opencl = OpenCLConfig::get();

        auto* aux = m_joint.training_raw<ArrowType>();
        for(int i = 0; i < N; ++i){
            training_data.data()[i] = aux[i];
        }

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
    auto* aux = m_joint.training_raw<ArrowType>();
    for(int i = 0; i < N * m_variables.size(); ++i){
        training_dataset.data()[i] = aux[i];
    }

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
    using VectorType = Matrix<CType, Dynamic, 1>;

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

template <typename T>
void exp_elementwise(T* mat, uint size) {
    for(uint idx = 0; idx < size; ++idx){
        mat[idx] = exp(mat[idx]);
    }
}

template <typename T>
void normalize_accum_sum_mat_cols(  T* mat,
                                    uint mat_rows,
                                    T* sums,
                                    uint size_dim1,
                                    uint size_dim2) {
    #define IDX(i, j, rows) (i) + ((j)*(rows))
    for(uint row_id = 0; row_id < size_dim1; ++row_id)
        for(uint col_id = 0; col_id < size_dim2; ++col_id)
            mat[IDX(row_id + 1, col_id, mat_rows)] /= sums[col_id];

}

template <typename T>
void find_random_indices(   T* mat,
                            uint mat_rows,
                            uint mat_offset,
                            T* random_numbers,
                            int* indices,
                            uint size_dim1,
                            uint size_dim2) {
    #define IDX(i, j, rows) (i) + ((j)*(rows))
    for(uint row_id = 0; row_id < size_dim1; ++row_id){
        for(uint col_id = 0; col_id < size_dim2; ++col_id){
            T rn = random_numbers[mat_offset + col_id];
            if (mat[IDX(row_id, col_id, mat_rows)] <= rn && rn < mat[IDX(row_id+1, col_id, mat_rows)])
                indices[mat_offset + col_id] = row_id;
        }
    }
}

template <typename ArrowType, typename KDEType>
Matrix<int, Dynamic, 1> CKDE::_sample_indices_from_weights(typename ArrowType::c_type* random_prob, typename ArrowType::c_type* test_buffer, int n) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto& opencl = OpenCLConfig::get();

    Matrix<int, Dynamic, 1> res_tmp(n);
    auto res_raw = res_tmp.data();

    for(int i = 0; i < n; ++i)
        res_tmp.data()[i] = N - 1;

    auto allocated_m = std::min(n, 64);
    auto iterations = static_cast<int>(std::ceil(static_cast<double>(n) / static_cast<double>(allocated_m)));

    auto tmp_mat_size = this->evidence().size() * (N>allocated_m?N:allocated_m);
    CType* tmp_mat_raw= (CType*)malloc(tmp_mat_size * sizeof(CType));

    CType* out_mat_raw = (CType*)malloc(N * allocated_m * sizeof(CType));

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
                                                      tmp_mat_raw,
                                                      out_mat_raw);

        exp_elementwise<CType>(out_mat_raw, N * allocated_m);
        auto total_sum_raw = opencl.accum_sum_cols<ArrowType>(out_mat_raw, N, allocated_m);
        normalize_accum_sum_mat_cols<CType>(out_mat_raw, N, total_sum_raw, N - 1, allocated_m);
        free(total_sum_raw);
        find_random_indices<CType>(out_mat_raw, N, i * allocated_m, random_prob, res_raw, N - 1, allocated_m);
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
                                                  tmp_mat_raw,
                                                  out_mat_raw);
    free(tmp_mat_raw);
    exp_elementwise<CType>(out_mat_raw, N * remaining_m);
    auto total_sum_raw = opencl.accum_sum_cols<ArrowType>(out_mat_raw, N, remaining_m);
    normalize_accum_sum_mat_cols<CType>(out_mat_raw, N, total_sum_raw, N - 1, remaining_m);
    free(total_sum_raw);
    find_random_indices<CType>(out_mat_raw, N, offset, random_prob, res_raw, N - 1, remaining_m);
    free(out_mat_raw);

    return res_tmp;
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
        for(int i = 0; i < df->num_rows(); ++i){
            read_data.data()[i] = res_tmp[i];
        }
        if constexpr (!std::is_same_v<CType, double>)
            return read_data.template cast<double>();
        else
            return read_data;
    } else {
        auto valid = df.valid_rows(m_variables);
        VectorType read_data(valid);
        auto bitmap = df.combined_bitmap(m_variables);
        auto bitmap_data = bitmap->data();

        for(int i = 0; i < valid; ++i)
            read_data.data()[i] = res_tmp[i];

        VectorXd res(df->num_rows());

        for (int i = 0, k = 0; i < df->num_rows(); ++i) {
            if (util::bit_util::GetBit(bitmap_data, i)) {
                res(i) = static_cast<double>(read_data[k++]);
            } else {
                res(i) = util::nan<double>;
            }
        }

        return res;
    }
}

template <typename T>
void univariate_normal_cdf( const T* means,
                            uint means_physical_rows,
                            const T* x,
                            uint x_offset,
                            T inv_std,
                            T inv_N,
                            T* cdf_mat,
                            uint m) {
    #define ROW(idx, rows) (idx) % (rows)
    #define COL(idx, rows) (idx) / (rows)

    for(uint i = 0; i < means_physical_rows * m; ++i){
        int means_idx = ROW(i, means_physical_rows);
        int x_idx = COL(i, means_physical_rows);

        cdf_mat[i] = inv_N*(0.5*erfc(M_SQRT1_2l * inv_std * -(x[x_offset + x_idx] - means[means_idx])));
    }
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> CKDE::_cdf_univariate(typename ArrowType::c_type* test_buffer, int m) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto& opencl = OpenCLConfig::get();

    auto allocated_m = std::min(m, 64);
    auto iterations = std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m));

    VectorType mu_tmp(N * allocated_m);
    auto mu_raw = mu_tmp.data();

    VectorType res_tmp(m);
    auto res_raw = res_tmp.data();

    for (auto i = 0; i < (iterations - 1); ++i) {
        univariate_normal_cdf<CType>(   m_joint.training_raw<ArrowType>(),
                                        N,
                                        test_buffer,
                                        static_cast<unsigned int>(i * allocated_m),
                                        static_cast<CType>(1.0 / std::sqrt(m_joint.bandwidth()(0, 0))),
                                        static_cast<CType>(1.0 / N),
                                        mu_raw,
                                        allocated_m);

        opencl.sum_cols_offset<ArrowType>(mu_raw, N, allocated_m, res_raw, i * allocated_m, m);
    }
    auto offset = (iterations - 1) * allocated_m;
    auto remaining_m = m - offset;

    univariate_normal_cdf<CType>(   m_joint.training_raw<ArrowType>(),
                                    N,
                                    test_buffer,
                                    offset,
                                    static_cast<CType>(1.0 / std::sqrt(m_joint.bandwidth()(0, 0))),
                                    static_cast<CType>(1.0 / N),
                                    mu_raw,
                                    remaining_m);
    opencl.sum_cols_offset<ArrowType>(mu_raw, N, remaining_m, res_raw, offset, m);

    return res_tmp;
}

template <typename T>
void normal_cdf(T* means,
                uint means_physical_rows,
                T* x,
                uint x_offset,
                T inv_std,
                uint size) {
    #define COL(idx, rows) (idx) / (rows)
    for(uint i = 0; i < size; ++i){
        int col_idx = COL(i, means_physical_rows);
        means[i] = 0.5*erfc(M_SQRT1_2 * inv_std * (means[i] - x[x_offset + col_idx]));
    }
}

template <typename T>
void product_elementwise(T* mat1, T* mat2, uint size) {
    for(uint i = 0; i < size; ++i)
        mat1[i] *= mat2[i];
}

template <typename T>
void division_elementwise(  T* mat1,
                            uint mat1_offset,
                            T* mat2,
                            uint size) {
    for(uint i = 0; i < size; ++i)
        mat1[mat1_offset + i] /= mat2[i];
}

template <typename ArrowType, typename KDEType>
Matrix<typename ArrowType::c_type, Dynamic, 1> CKDE::_cdf_multivariate(typename ArrowType::c_type* variable_test_buffer, typename ArrowType::c_type* evidence_test_buffer, int m) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

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

    auto& opencl = OpenCLConfig::get();
 
    auto allocated_m = std::min(m, 64);
    auto iterations = static_cast<int>(std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m)));

    auto new_lognorm_marg = m_marg.lognorm_const() + std::log(N);

    auto tmp_mat_size = this->evidence().size() * (N>allocated_m?N:allocated_m);
    VectorType tmp_mat_tmp(tmp_mat_size);
    auto tmp_mat_raw = tmp_mat_tmp.data();

    VectorType output_mat_tmp(N * allocated_m);
    auto out_mat_raw = output_mat_tmp.data();

    VectorType res_tmp(m);
    auto res_raw = res_tmp.data();

    VectorType sum_W_tmp(allocated_m);
    auto sum_W_raw = sum_W_tmp.data();

    VectorType transform_buffer_tmp(this->evidence().size());
    auto transform_buffer_tmp_raw = transform.data();

    VectorType mu_tmp(N * allocated_m);
    auto mu_raw = mu_tmp.data();

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
                                                      tmp_mat_raw,
                                                      out_mat_raw);

        auto W_raw = out_mat_raw;

        exp_elementwise<CType>(W_raw, N * allocated_m);

        opencl.sum_cols_offset<ArrowType>(W_raw, N, allocated_m, sum_W_raw, 0, allocated_m);

        // Computes conditional mu.
        KDEType::template execute_conditional_means<ArrowType>(m_joint.training_raw<ArrowType>(),
                                                               m_marg.training_raw<ArrowType>(),
                                                               N,
                                                               evidence_test_buffer,
                                                               m,
                                                               i * allocated_m,
                                                               allocated_m,
                                                               this->evidence().size(),
                                                               transform_buffer_tmp_raw,
                                                               tmp_mat_raw,
                                                               mu_raw);

        normal_cdf<CType>(mu_raw, N, variable_test_buffer, i * allocated_m, static_cast<CType>(1.0 / std::sqrt(cond_var)), N * allocated_m);
        product_elementwise<CType>(mu_raw, W_raw, N * allocated_m);

        opencl.sum_cols_offset<ArrowType>(mu_raw, N, allocated_m, res_raw, i * allocated_m, m);

        division_elementwise<CType>(res_raw, i * allocated_m, sum_W_raw, allocated_m);
    }
    auto offset = (iterations - 1) * allocated_m;
    auto remaining_m = m - offset;

    tmp_mat_size = this->evidence().size() * (N>remaining_m?N:remaining_m);
    tmp_mat_tmp(tmp_mat_size);
    tmp_mat_raw = tmp_mat_tmp.data();

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
                                                  tmp_mat_raw,
                                                  out_mat_raw);

    auto W_raw = out_mat_raw;

    exp_elementwise<CType>(W_raw, N * remaining_m);

    opencl.sum_cols_offset<ArrowType>(W_raw, N, remaining_m, sum_W_raw, 0, allocated_m);

    // Computes conditional mu.
    KDEType::template execute_conditional_means<ArrowType>(m_joint.training_raw<ArrowType>(),
                                                           m_marg.training_raw<ArrowType>(),
                                                           N,
                                                           evidence_test_buffer,
                                                           m,
                                                           offset,
                                                           remaining_m,
                                                           this->evidence().size(),
                                                           transform_buffer_tmp_raw,
                                                           tmp_mat_raw,
                                                           mu_raw);

    normal_cdf<CType>(mu_raw, N, variable_test_buffer, offset, static_cast<CType>(1.0 / std::sqrt(cond_var)), N * remaining_m);
    product_elementwise<CType>(mu_raw, W_raw, N * remaining_m);

    opencl.sum_cols_offset<ArrowType>(mu_raw, N, remaining_m, res_raw, offset, m);

    division_elementwise<CType>(res_raw, offset, sum_W_raw, remaining_m);

    return res_tmp;
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