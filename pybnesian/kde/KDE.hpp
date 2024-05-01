#ifndef PYBNESIAN_KDE_KDE_HPP
#define PYBNESIAN_KDE_KDE_HPP

#include <iostream>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <kde/BandwidthSelector.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <util/math_constants.hpp>
#include <util/pickle.hpp>
#include <kernels/kernel.hpp>
#include <omp.h>

namespace kde {

struct UnivariateKDE {
    template <typename ArrowType>
    void static execute_logl_mat(const typename ArrowType::c_type* training_vec,
                                 const unsigned int training_length,
                                 const typename ArrowType::c_type* test_vec,
                                 const unsigned int,
                                 const unsigned int test_offset,
                                 const unsigned int test_length,
                                 const unsigned int,
                                 const typename ArrowType::c_type* cholesky,
                                 const typename ArrowType::c_type lognorm_const,
                                //  typename ArrowType::c_type*,
                                 typename ArrowType::c_type* output_mat);

    template <typename ArrowType>
    static void execute_conditional_means(const typename ArrowType::c_type* joint_training,
                                          const typename ArrowType::c_type*,
                                          const unsigned int training_rows,
                                          const typename ArrowType::c_type* evidence_test,
                                          const unsigned int test_physical_rows,
                                          const unsigned int test_offset,
                                          const unsigned int test_length,
                                          const unsigned int,
                                          const typename ArrowType::c_type* transform_mean,
                                          typename ArrowType::c_type*,
                                          typename ArrowType::c_type* output_mat);
};

template <typename ArrowType>
void UnivariateKDE::execute_logl_mat(const typename ArrowType::c_type* training_vec,
                                     const unsigned int training_length,
                                     const typename ArrowType::c_type* test_vec,
                                     const unsigned int,
                                     const unsigned int test_offset,
                                     const unsigned int test_length,
                                     const unsigned int,
                                     const typename ArrowType::c_type* cholesky,
                                     const typename ArrowType::c_type lognorm_const,
                                    //  typename ArrowType::c_type*,
                                     typename ArrowType::c_type* output_mat) {
    using CType = typename ArrowType::c_type;

    Kernel<CType>::instance().logl_values_1d_mat(training_vec, training_length, test_vec, test_offset, cholesky, lognorm_const, output_mat, test_length);
}

template <typename ArrowType>
void UnivariateKDE::execute_conditional_means(const typename ArrowType::c_type* joint_training,
                                              const typename ArrowType::c_type*,
                                              const unsigned int training_rows,
                                              const typename ArrowType::c_type* evidence_test,
                                              const unsigned int test_physical_rows,
                                              const unsigned int test_offset,
                                              const unsigned int test_length,
                                              const unsigned int,
                                              const typename ArrowType::c_type* transform_mean,
                                              typename ArrowType::c_type*,
                                              typename ArrowType::c_type* output_mat) {
    using CType = typename ArrowType::c_type;

    Kernel<CType>::instance().conditional_means_1d(joint_training, training_rows, evidence_test, test_physical_rows, test_offset, transform_mean, output_mat, test_length);    
}

struct MultivariateKDE {
    template <typename ArrowType>
    static void execute_logl_mat(const typename ArrowType::c_type* training_mat,
                                 const unsigned int training_rows,
                                 const typename ArrowType::c_type* test_mat,
                                 const unsigned int test_physical_rows,
                                 const unsigned int test_offset,
                                 const unsigned int test_length,
                                 const unsigned int matrices_cols,
                                 const typename ArrowType::c_type* cholesky,
                                 const typename ArrowType::c_type lognorm_const,
                                //  typename ArrowType::c_type* tmp_mat,
                                 typename ArrowType::c_type* output_mat);

    template <typename ArrowType>
    static void execute_conditional_means(const typename ArrowType::c_type* joint_training,
                                          const typename ArrowType::c_type* marg_training,
                                          const unsigned int training_rows,
                                          const typename ArrowType::c_type* evidence_test,
                                          const unsigned int test_physical_rows,
                                          const unsigned int test_offset,
                                          const unsigned int test_length,
                                          const unsigned int evidence_cols,
                                          const typename ArrowType::c_type* transform_mean,
                                          typename ArrowType::c_type* tmp_mat,
                                          typename ArrowType::c_type* output_mat);
};

template <typename ArrowType>
void MultivariateKDE::execute_logl_mat(const typename ArrowType::c_type* training_mat,
                                            const unsigned int training_rows,
                                            const typename ArrowType::c_type* test_mat,
                                            const unsigned int test_physical_rows,
                                           const unsigned int test_offset,
                                            const unsigned int test_length,
                                            const unsigned int matrices_cols,
                                            const typename ArrowType::c_type* cholesky,
                                            const typename ArrowType::c_type lognorm_const,
                                            // typename ArrowType::c_type* tmp_mat,
                                            typename ArrowType::c_type* output_mat) {
    using CType = typename ArrowType::c_type;

    auto tmp_mat_size = matrices_cols * (training_rows>test_length?training_rows:test_length);
#pragma omp parallel
#pragma omp single
{
    Kernel<CType> kernels = Kernel<CType>::instance();
    int n_tasks = omp_get_num_threads();
    if (training_rows > test_length) {
#pragma omp taskloop num_tasks(n_tasks)
        for (uint i = 0; i < test_length; ++i) {
                CType* tmp_mat_raw = (CType*)malloc(tmp_mat_size * sizeof(CType));
                kernels.substract_domain_specific_new(training_mat, training_rows, 0, training_rows, test_mat, test_physical_rows, test_offset, i, matrices_cols, tmp_mat_raw);
                kernels.solve_specific_new(tmp_mat_raw, training_rows, matrices_cols, cholesky);
                kernels.square_inplace_new(tmp_mat_raw, training_rows * matrices_cols);
                kernels.logl_values_mat_column_new(tmp_mat_raw, matrices_cols, output_mat, training_rows, i, lognorm_const, training_rows);
                free(tmp_mat_raw);
        }
    } else {
#pragma omp taskloop num_tasks(n_tasks)
        for (uint i = 0; i < training_rows; ++i) {
                CType* tmp_mat_raw = (CType*)malloc(tmp_mat_size * sizeof(CType));
                kernels.substract_domain_specific_new(test_mat, test_physical_rows, test_offset, test_length, training_mat, training_rows, 0, i, matrices_cols, tmp_mat_raw);
                kernels.solve_specific_new(tmp_mat_raw, test_length, matrices_cols, cholesky);
                kernels.square_inplace_new(tmp_mat_raw, test_length * matrices_cols);
                kernels.logl_values_mat_row_new(tmp_mat_raw, matrices_cols, output_mat, training_rows, i, lognorm_const, test_length);
                free(tmp_mat_raw);
        }
    }
}
}

template <typename ArrowType>
void MultivariateKDE::execute_conditional_means(const typename ArrowType::c_type* joint_training,
                                                const typename ArrowType::c_type* marg_training,
                                                const unsigned int training_rows,
                                                const typename ArrowType::c_type* evidence_test,
                                                const unsigned int test_physical_rows,
                                                const unsigned int test_offset,
                                                const unsigned int test_length,
                                                const unsigned int evidence_cols,
                                                const typename ArrowType::c_type* transform_mean,
                                                typename ArrowType::c_type* tmp_mat,
                                                typename ArrowType::c_type* output_mat) {
    using CType = typename ArrowType::c_type;
    Kernel<CType> kernels = Kernel<CType>::instance();

    if (training_rows > test_length) {
        for (unsigned int i = 0; i < test_length; ++i) {
            kernels.substract_domain_specific_new(marg_training, training_rows, 0u, training_rows, evidence_test, test_physical_rows, test_offset, i, evidence_cols, tmp_mat);
            kernels.conditional_means_column(joint_training, training_rows, tmp_mat, training_rows, transform_mean, evidence_cols, output_mat, i, training_rows);
        }
    } else {
        for (unsigned int i = 0; i < training_rows; ++i) {
            kernels.substract_domain_specific_new(evidence_test, test_physical_rows, test_offset, test_length, marg_training, training_rows, 0, i, evidence_cols, tmp_mat);
            kernels.conditional_means_row(joint_training, training_rows, tmp_mat, test_length, transform_mean, evidence_cols, output_mat, i, training_rows);
        }
    }
}

class KDE {
public:
    KDE()
        : m_variables(),
          m_fitted(false),
          m_bselector(std::make_shared<NormalReferenceRule>()),
          m_bandwidth(),
          m_lognorm_const(0),
          N(0),
          m_training_type(arrow::float64()) {}

    KDE(std::vector<std::string> variables) : KDE(variables, std::make_shared<NormalReferenceRule>()) {}

    KDE(std::vector<std::string> variables, std::shared_ptr<BandwidthSelector> b_selector)
        : m_variables(variables),
          m_fitted(false),
          m_bselector(b_selector),
          m_bandwidth(),
          m_lognorm_const(0),
          N(0),
          m_training_type(arrow::float64()) {
        if (b_selector == nullptr) throw std::runtime_error("Bandwidth selector procedure must be non-null.");

        if (m_variables.empty()) {
            throw std::invalid_argument("Cannot create a KDE model with 0 variables");
        }
    }

    const std::vector<std::string>& variables() const { return m_variables; }
    void fit(const DataFrame& df);

    template <typename ArrowType, typename EigenMatrix>
    void fit(EigenMatrix bandwidth,
             typename ArrowType::c_type* training_data,
             std::shared_ptr<arrow::DataType> training_type,
             int training_instances);

    const MatrixXd& bandwidth() const { return m_bandwidth; }
    void setBandwidth(MatrixXd& new_bandwidth) {
        if (new_bandwidth.rows() != new_bandwidth.cols() ||
            static_cast<size_t>(new_bandwidth.rows()) != m_variables.size())
            throw std::invalid_argument(
                "The bandwidth matrix must be a square matrix with shape "
                "(" +
                std::to_string(m_variables.size()) + ", " + std::to_string(m_variables.size()) + ")");

        m_bandwidth = new_bandwidth;
        if (m_bandwidth.rows() > 0) copy_bandwidth();
    }

    template <typename ArrowType>
    typename ArrowType::c_type* training_raw() { 
        using CType = typename ArrowType::c_type;
        if constexpr (std::is_same_v<CType, double>) {
            return m_training_double.data(); 
        } else {
            return m_training_float.data();
        }
    }

    template <typename ArrowType>
    const typename ArrowType::c_type* training_raw() const { 
        using CType = typename ArrowType::c_type;
        if constexpr (std::is_same_v<CType, double>) {
            return m_training_double.data(); 
        } else {
            return m_training_float.data();
        }
    }

    template <typename ArrowType>
    typename ArrowType::c_type* cholesky_raw() { 
        using CType = typename ArrowType::c_type;
        if constexpr (std::is_same_v<CType, double>) {
            return m_H_cholesky_double.data(); 
        } else {
            return m_H_cholesky_float.data();
        }
    }

    template <typename ArrowType>
    const typename ArrowType::c_type* cholesky_raw() const { 
        using CType = typename ArrowType::c_type;
        if constexpr (std::is_same_v<CType, double>) {
            return m_H_cholesky_double.data(); 
        } else {
            return m_H_cholesky_float.data();
        }
    }

    double lognorm_const() const { return m_lognorm_const; }

    DataFrame training_data() const;

    int num_instances() const {
        check_fitted();
        return N;
    }
    int num_variables() const { return m_variables.size(); }
    bool fitted() const { return m_fitted; }

    std::shared_ptr<arrow::DataType> data_type() const {
        check_fitted();
        return m_training_type;
    }

    std::shared_ptr<BandwidthSelector> bandwidth_type() const { return m_bselector; }

    VectorXd logl(const DataFrame& df) const;

    template <typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> logl_buffer(const DataFrame& df) const;
    template <typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> logl_buffer(const DataFrame& df, Buffer_ptr& bitmap) const;

    double slogl(const DataFrame& df) const;

    void save(const std::string name) { util::save_object(*this, name); }

    py::tuple __getstate__() const;
    static KDE __setstate__(py::tuple& t);
    static KDE __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    void check_fitted() const {
        if (!fitted()) throw std::invalid_argument("KDE factor not fitted.");
    }
    template <typename ArrowType>
    DataFrame _training_data() const;

    template <typename ArrowType, bool contains_null>
    void _fit(const DataFrame& df);
    template <typename ArrowType>
    VectorXd _logl(const DataFrame& df) const;
    template <typename ArrowType>
    double _slogl(const DataFrame& df) const;

    template <typename ArrowType, typename KDEType>
    void _logl_impl(typename ArrowType::c_type* test_buffer, int m, typename ArrowType::c_type* res) const;

    void copy_bandwidth();

    template <typename ArrowType>
    py::tuple __getstate__() const;

    std::vector<std::string> m_variables;
    bool m_fitted;
    std::shared_ptr<BandwidthSelector> m_bselector;
    MatrixXd m_bandwidth;
    Matrix<double, Dynamic, Dynamic> m_H_cholesky_double;
    Matrix<float, Dynamic, 1> m_H_cholesky_float;
    Matrix<double, Dynamic, 1> m_training_double;
    Matrix<float, Dynamic, 1> m_training_float;
    double m_lognorm_const;
    int N;
    std::shared_ptr<arrow::DataType> m_training_type;
};

template <typename ArrowType>
DataFrame KDE::_training_data() const {
    arrow::NumericBuilder<ArrowType> builder;

    std::vector<Array_ptr> columns;
    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);
    for (size_t i = 0; i < m_variables.size(); ++i) {
        auto status = builder.Resize(N);
        RAISE_STATUS_ERROR(builder.AppendValues(training_raw<ArrowType>() + i * N, N));

        Array_ptr out;
        RAISE_STATUS_ERROR(builder.Finish(&out));

        columns.push_back(out);
        builder.Reset();

        auto f = arrow::field(m_variables[i], out->type());
        RAISE_STATUS_ERROR(b.AddField(f));
    }

    RAISE_RESULT_ERROR(auto schema, b.Finish())

    auto rb = arrow::RecordBatch::Make(schema, N, columns);
    return DataFrame(rb);
}

template <typename ArrowType, bool contains_null>
void KDE::_fit(const DataFrame& df) {
    using CType = typename ArrowType::c_type;

    auto d = m_variables.size();

    m_bandwidth = m_bselector->bandwidth(df, m_variables);

    auto llt_cov = m_bandwidth.llt();
    auto llt_matrix = llt_cov.matrixLLT();

    if constexpr (std::is_same_v<CType, double>) {
        m_H_cholesky_double = llt_matrix;
    } else {
        using MatrixType = Matrix<CType, Dynamic, Dynamic>;
        MatrixType casted_cholesky = llt_matrix.template cast<CType>();

        m_H_cholesky_float = Matrix<float, Dynamic, 1>(d * d);
        memcpy(m_H_cholesky_float.data(), casted_cholesky.data(), d*d*sizeof(float));
    }

    auto training_data = df.to_eigen<false, ArrowType, contains_null>(m_variables);
    N = training_data->rows();
    if constexpr (std::is_same_v<CType, double>) {
        m_training_double = Matrix<double, Dynamic, 1>(N * d);
        memcpy(m_training_double.data(), training_data->data(), N*d*sizeof(double));
    } else {
        m_training_float = Matrix<float, Dynamic, 1>(N * d);
        memcpy(m_training_float.data(), training_data->data(), N*d*sizeof(float));
    }

    m_lognorm_const =
        -llt_matrix.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<double>) - std::log(N);
}

template <typename ArrowType, typename EigenMatrix>
void KDE::fit(EigenMatrix bandwidth,
              typename ArrowType::c_type* training_data,
              std::shared_ptr<arrow::DataType> training_type,
              int training_instances) {
    using CType = typename ArrowType::c_type;

    if ((bandwidth.rows() != bandwidth.cols()) || (static_cast<size_t>(bandwidth.rows()) != m_variables.size())) {
        throw std::invalid_argument("Bandwidth matrix must be a square matrix with dimensionality " +
                                    std::to_string(m_variables.size()));
    }

    m_bandwidth = bandwidth;
    auto d = m_variables.size();
    auto llt_cov = bandwidth.llt();
    auto cholesky = llt_cov.matrixLLT();

    if constexpr (std::is_same_v<CType, double>) {
        m_H_cholesky_double = Matrix<double, Dynamic, 1>(d * d);
        std::memcpy(m_H_cholesky_double.data(), cholesky.data(), d*d*sizeof(double));
    } else {
        using MatrixType = Matrix<CType, Dynamic, Dynamic>;
        MatrixType casted_cholesky = cholesky.template cast<CType>();

        m_H_cholesky_float = Matrix<float, Dynamic, 1>(d * d);
        std::memcpy(m_H_cholesky_float.data(), casted_cholesky.data(), d*d*sizeof(float));
    }

    N = training_instances;
    if constexpr (std::is_same_v<CType, double>) {
        m_training_double = Matrix<double, Dynamic, 1>(N * d);
        std::memcpy(m_training_double.data(), training_data, N*d*sizeof(double));
    } else {
        m_training_float = Matrix<float, Dynamic, 1>(N * d);
        std::memcpy(m_training_float.data(), training_data, N*d*sizeof(float));
    }

    m_training_type = training_type;
    m_lognorm_const = -cholesky.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<double>) - std::log(N);
    m_fitted = true;
}

template <typename ArrowType>
VectorXd KDE::_logl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    if (df.null_count(m_variables) == 0) {
        VectorType read_data(df->num_rows());
        read_data = logl_buffer<ArrowType>(df);
        if constexpr (!std::is_same_v<CType, double>)
            return read_data.template cast<double>();
        else
            return read_data;
    } else {
        auto m = df.valid_rows(m_variables);
        VectorType read_data(m);
        auto bitmap = df.combined_bitmap(m_variables);
        auto bitmap_data = bitmap->data();

        read_data = logl_buffer<ArrowType>(df);

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

template <typename ArrowType>
double KDE::_slogl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;

    auto m = df.valid_rows(m_variables);
    auto logl_mat = logl_buffer<ArrowType>(df);

    CType result = -1;
    Kernel<CType>::instance().sum1d(logl_mat.data(), m, &result);
    return static_cast<double>(result);
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> KDE::logl_buffer(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);

    VectorType res(test_matrix->rows());
    if (m_variables.size() == 1)
        _logl_impl<ArrowType, UnivariateKDE>(test_matrix->data(), test_matrix->rows(), res.data());
    else
        _logl_impl<ArrowType, MultivariateKDE>(test_matrix->data(), test_matrix->rows(), res.data());
    return res;
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> KDE::logl_buffer(const DataFrame& df, Buffer_ptr& bitmap) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto test_matrix = df.to_eigen<false, ArrowType>(bitmap, m_variables);

    VectorType res(test_matrix->rows());
    if (m_variables.size() == 1)
        _logl_impl<ArrowType, UnivariateKDE>(test_matrix->data(), test_matrix->rows(), res.data());
    else
        _logl_impl<ArrowType, MultivariateKDE>(test_matrix->data(), test_matrix->rows(), res.data());
    return res;
}

template <typename ArrowType, typename KDEType>
void KDE::_logl_impl(typename ArrowType::c_type* test_buffer, int m, typename ArrowType::c_type* res) const {
    using CType = typename ArrowType::c_type;
    Kernel<CType> kernels = Kernel<CType>::instance();

    auto d = m_variables.size();

    auto allocated_m = std::min(m, 64);
    auto iterations = static_cast<int>(std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m)));

    CType* out = (CType*)malloc(N * allocated_m * sizeof(CType));

    for (auto i = 0; i < (iterations - 1); ++i) {
        KDEType::template execute_logl_mat<ArrowType>(training_raw<ArrowType>(),
                                                      N,
                                                      test_buffer,
                                                      m,
                                                      i * allocated_m,
                                                      allocated_m,
                                                      d,
                                                      cholesky_raw<ArrowType>(),
                                                      m_lognorm_const,
                                                      out);
        kernels.logsumexp_cols_offset(out, N, allocated_m, res, i * allocated_m);
    }
    free(out);

    auto remaining_m = m - (iterations - 1) * allocated_m;
    out = (CType*)malloc(N * remaining_m * sizeof(CType));

    KDEType::template execute_logl_mat<ArrowType>(training_raw<ArrowType>(),
                                                  N,
                                                  test_buffer,
                                                  m,
                                                  m - remaining_m,
                                                  remaining_m,
                                                  d,
                                                  cholesky_raw<ArrowType>(),
                                                  m_lognorm_const,
                                                  out);
    kernels.logsumexp_cols_offset(out, N, remaining_m, res, (iterations - 1) * allocated_m);
    free(out);
}

template <typename ArrowType>
py::tuple KDE::__getstate__() const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    MatrixXd bw;
    VectorType training_data;
    double lognorm_const = -1;
    int N_export = -1;
    int training_type = -1;

    if (m_fitted) {
        training_data = VectorType(N * m_variables.size());
        std::memcpy(training_data.data(), training_raw<ArrowType>(), N*m_variables.size()*sizeof(CType));
        lognorm_const = m_lognorm_const;
        training_type = static_cast<int>(m_training_type->id());
        N_export = N;
        bw = m_bandwidth;
    }

    return py::make_tuple(
        m_variables, m_fitted, m_bselector, bw, training_data, lognorm_const, N_export, training_type);
}

}  // namespace kde

#endif  // PYBNESIAN_KDE_KDE_HPP
