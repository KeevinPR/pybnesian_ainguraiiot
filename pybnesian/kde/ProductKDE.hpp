#ifndef PYBNESIAN_KDE_PRODUCTKDE_HPP
#define PYBNESIAN_KDE_PRODUCTKDE_HPP

#include <util/pickle.hpp>
#include <kde/BandwidthSelector.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <util/math_constants.hpp>
#include <iostream>
#include <kernels/kernel.hpp>

namespace kde {

class ProductKDE {
public:
    ProductKDE()
        : m_variables(),
          m_fitted(),
          m_bselector(std::make_shared<NormalReferenceRule>()),
          N(0),
          m_training_type(arrow::float64()) {}

    ProductKDE(std::vector<std::string> variables) : ProductKDE(variables, std::make_shared<NormalReferenceRule>()) {}

    ProductKDE(std::vector<std::string> variables, std::shared_ptr<BandwidthSelector> b_selector)
        : m_variables(variables), m_fitted(false), m_bselector(b_selector), N(0), m_training_type(arrow::float64()) {
        if (b_selector == nullptr) throw std::runtime_error("Bandwidth selector procedure must be non-null.");

        if (m_variables.empty()) {
            throw std::invalid_argument("Cannot create a ProductKDE model with 0 variables");
        }
    }

    const std::vector<std::string>& variables() const { return m_variables; }

    void fit(const DataFrame& df);

    const VectorXd& bandwidth() const { return m_bandwidth; }
    void setBandwidth(VectorXd& new_bandwidth) {
        if (static_cast<size_t>(new_bandwidth.rows()) != m_variables.size())
            throw std::invalid_argument(
                "The bandwidth matrix must be a vector with shape "
                "(" +
                std::to_string(m_variables.size()) + ")");

        m_bandwidth = new_bandwidth;
        if (m_bandwidth.rows() > 0) copy_bandwidth();
    }

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

    double slogl(const DataFrame& df) const;

    void save(const std::string name) { util::save_object(*this, name); }

    py::tuple __getstate__() const;
    static ProductKDE __setstate__(py::tuple& t);
    static ProductKDE __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    void check_fitted() const {
        if (!fitted()) throw std::invalid_argument("ProductKDE factor not fitted.");
    }

    template <typename ArrowType>
    DataFrame _training_data() const;

    template <typename ArrowType, bool contains_null>
    void _fit(const DataFrame& df);

    template <typename ArrowType>
    VectorXd _logl(const DataFrame& df) const;
    template <typename ArrowType>
    double _slogl(const DataFrame& df) const;

    template <typename ArrowType>
    void product_logl_mat(typename ArrowType::c_type* test_buffer,
                          const unsigned int test_offset,
                          const unsigned int test_length,
                          typename ArrowType::c_type* output_mat) const;

    template <typename ArrowType>
    void _logl_impl(typename ArrowType::c_type* test_buffer, int m, typename ArrowType::c_type* res) const;

    void copy_bandwidth();

    template <typename ArrowType>
    py::tuple __getstate__() const;

    std::vector<std::string> m_variables;
    bool m_fitted;
    std::shared_ptr<BandwidthSelector> m_bselector;
    VectorXd m_bandwidth;
    std::vector<Matrix<double, Dynamic, 1>> m_bandwidth_double;
    std::vector<Matrix<float, Dynamic, 1>> m_bandwidth_float;
    std::vector<Matrix<double, Dynamic, 1>> m_training_double;
    std::vector<Matrix<float, Dynamic, 1>> m_training_float;
    double m_lognorm_const;
    size_t N;
    std::shared_ptr<arrow::DataType> m_training_type;
};

template <typename ArrowType>
DataFrame ProductKDE::_training_data() const {
    using CType = typename ArrowType::c_type;
    arrow::NumericBuilder<ArrowType> builder;

    const CType* tmp;

    std::vector<Array_ptr> columns;
    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);
    for (size_t i = 0; i < m_variables.size(); ++i) {
        if constexpr (std::is_same_v<CType, double>) {
            tmp = m_training_double[i].data();
        } else {
            tmp = m_training_float[i].data();
        }

        auto status = builder.Resize(N);
        RAISE_STATUS_ERROR(builder.AppendValues(tmp, N));

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
void ProductKDE::_fit(const DataFrame& df) {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    if (static_cast<size_t>(m_bandwidth.rows()) != m_variables.size()) m_bandwidth = VectorXd(m_variables.size());
    m_bandwidth_double.clear();
    m_bandwidth_float.clear();
    m_training_double.clear();
    m_training_float.clear();

    Buffer_ptr combined_bitmap;
    if constexpr (contains_null) combined_bitmap = df.combined_bitmap(m_variables);

    N = df.valid_rows(m_variables);

    m_bandwidth = m_bselector->diag_bandwidth(df, m_variables);

    for (size_t i = 0; i < m_variables.size(); ++i) {
        if constexpr (std::is_same_v<CType, double>) {
            auto sqrt = std::sqrt(m_bandwidth(i));
            VectorType aux(1);
            aux[0] = sqrt;
            m_bandwidth_double.push_back(aux);
        } else {
            auto casted = std::sqrt(static_cast<CType>(m_bandwidth(i)));
            VectorType aux(1);
            aux[0] = casted;
            m_bandwidth_float.push_back(aux);
        }

        if constexpr (contains_null) {
            auto column = df.to_eigen<false, ArrowType>(combined_bitmap, m_variables[i]);
            VectorType aux(N);
            std::memcpy(aux.data(), column->data(), N*sizeof(CType));
            if constexpr (std::is_same_v<CType, double>) {
                m_training_double.push_back(aux);
            } else {
                m_training_float.push_back(aux);
            }
        } else {
            auto column = df.to_eigen<false, ArrowType, false>(m_variables[i]);
            VectorType aux(N);
            std::memcpy(aux.data(), column->data(), N*sizeof(CType));
            if constexpr (std::is_same_v<CType, double>) {
                m_training_double.push_back(aux);
            } else {
                m_training_float.push_back(aux);
            }
        }
    }

    m_lognorm_const = -0.5 * static_cast<double>(m_variables.size()) * std::log(2 * util::pi<double>) -
                      0.5 * m_bandwidth.array().log().sum() - std::log(N);
}

template <typename ArrowType>
VectorXd ProductKDE::_logl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;

    auto read_data = logl_buffer<ArrowType>(df);
    if (df.null_count(m_variables) == 0) {
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
                res(i) = static_cast<double>(read_data[k++]);
            } else {
                res(i) = util::nan<double>;
            }
        }

        return res;
    }
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> ProductKDE::logl_buffer(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
    auto m = test_matrix->rows();

    VectorType res(m);
    _logl_impl<ArrowType>(test_matrix->data(), m, res.data());
    return res;
}

template <typename ArrowType>
void ProductKDE::product_logl_mat(typename ArrowType::c_type* test_buffer,
                                  const unsigned int test_offset,
                                  const unsigned int test_length,
                                  typename ArrowType::c_type* output_mat) const {
    using CType = typename ArrowType::c_type;
    Kernel<CType> kernels = Kernel<CType>::instance();

    const CType* m_training_tmp;
    if constexpr (std::is_same_v<CType, double>)
        m_training_tmp = m_training_double[0].data();
    else
        m_training_tmp = m_training_float[0].data();

    CType m_cl_bandwidth_tmp;
    if constexpr (std::is_same_v<CType, double>)
        m_cl_bandwidth_tmp = m_bandwidth_double[0][0];
    else
        m_cl_bandwidth_tmp = m_bandwidth_float[0][0];

    kernels.prod_logl_values_1d_mat(m_training_tmp, N, test_buffer, test_offset, m_cl_bandwidth_tmp, m_lognorm_const, output_mat, test_length);

    for (size_t i = 1; i < m_variables.size(); ++i) {
        if constexpr (std::is_same_v<CType, double>)
            m_training_tmp = m_training_double[i].data();
        else
            m_training_tmp = m_training_float[i].data();

        if constexpr (std::is_same_v<CType, double>)
            m_cl_bandwidth_tmp = m_bandwidth_double[i][0];
        else
            m_cl_bandwidth_tmp = m_bandwidth_float[i][0];

        kernels.add_logl_values_1d_mat(m_training_tmp, N, test_buffer, (i * test_length) + test_offset, m_cl_bandwidth_tmp, output_mat, test_length);
    }
}

template <typename ArrowType>
void ProductKDE::_logl_impl(typename ArrowType::c_type* test_buffer, int m, typename ArrowType::c_type* res) const {
    using CType = typename ArrowType::c_type;

    Kernel<CType> kernels = Kernel<CType>::instance();

    auto allocated_m = std::min(m, 64);
    auto iterations = static_cast<int>(std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m)));

    CType* tmp = (CType*)malloc(N*allocated_m*sizeof(CType));

    for (auto i = 0; i < (iterations - 1); ++i) {
        product_logl_mat<ArrowType>(test_buffer, i * allocated_m, allocated_m, tmp);
        kernels.logsumexp_cols_offset(tmp, N, allocated_m, res, i * allocated_m);
    }

    auto remaining_m = m - (iterations - 1) * allocated_m;
    product_logl_mat<ArrowType>(test_buffer, m - remaining_m, remaining_m, tmp);
    kernels.logsumexp_cols_offset(tmp, N, remaining_m, res, m - remaining_m);
    free(tmp);
}

template <typename ArrowType>
double ProductKDE::_slogl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;

    auto m = df.valid_rows(m_variables);
    auto logl_buff = logl_buffer<ArrowType>(df);

    CType result = -1;
    Kernel<CType>::instance().sum1d(logl_buff.data(), m, &result);
    return static_cast<double>(result);
}

template <typename ArrowType>
py::tuple ProductKDE::__getstate__() const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    VectorXd bw;
    std::vector<VectorType> training_data;
    double lognorm_const = -1;
    int N_export = -1;
    int training_type = -1;

    if (m_fitted) {
        for (size_t i = 0; i < m_variables.size(); ++i) {
            VectorType column(N);
            if constexpr (std::is_same_v<CType, double>) {
                column = m_training_double[i];
            } else {
                column = m_training_float[i];
            }
        }

        lognorm_const = m_lognorm_const;
        training_type = static_cast<int>(m_training_type->id());
        N_export = N;
        bw = m_bandwidth;
    }

    return py::make_tuple(
        m_variables, m_fitted, m_bselector, bw, training_data, lognorm_const, N_export, training_type);
}

}  // namespace kde

#endif  // PYBNESIAN_KDE_PRODUCTKDE_HPP
