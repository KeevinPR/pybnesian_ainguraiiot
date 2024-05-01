#ifndef PYBNESIAN_KDE_PRODUCTKDE_HPP
#define PYBNESIAN_KDE_PRODUCTKDE_HPP

#include <util/pickle.hpp>
#include <kde/BandwidthSelector.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <opencl/opencl_config.hpp>
#include <util/math_constants.hpp>
#include <iostream>

using opencl::OpenCLConfig, opencl::OpenCL_kernel_traits;

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
        if (m_bandwidth.rows() > 0) copy_bandwidth_opencl();
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

    void copy_bandwidth_opencl();

    template <typename ArrowType>
    py::tuple __getstate__() const;

    std::vector<std::string> m_variables;
    bool m_fitted;
    std::shared_ptr<BandwidthSelector> m_bselector;
    VectorXd m_bandwidth;
    std::vector<Matrix<double, Dynamic, 1>> m_cl_bandwidth_raw_double;
    std::vector<Matrix<float, Dynamic, 1>> m_cl_bandwidth_raw_float;
    std::vector<Matrix<double, Dynamic, 1>> m_training_raw_double;
    std::vector<Matrix<float, Dynamic, 1>> m_training_raw_float;
    double m_lognorm_const;
    size_t N;
    std::shared_ptr<arrow::DataType> m_training_type;
};

template <typename ArrowType>
DataFrame ProductKDE::_training_data() const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    arrow::NumericBuilder<ArrowType> builder;

    VectorType tmp_buffer(N);

    std::vector<Array_ptr> columns;
    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);
    for (size_t i = 0; i < m_variables.size(); ++i) {
        if constexpr (std::is_same_v<CType, double>) {
            tmp_buffer = m_training_raw_double[i];
        } else {
            tmp_buffer = m_training_raw_float[i];
        }


        auto status = builder.Resize(N);
        RAISE_STATUS_ERROR(builder.AppendValues(tmp_buffer.data(), N));

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
    // m_cl_bandwidth.clear();
    m_cl_bandwidth_raw_double.clear();
    m_cl_bandwidth_raw_float.clear();
    // m_training.clear();
    m_training_raw_double.clear();
    m_training_raw_float.clear();

    Buffer_ptr combined_bitmap;
    if constexpr (contains_null) combined_bitmap = df.combined_bitmap(m_variables);

    N = df.valid_rows(m_variables);

    m_bandwidth = m_bselector->diag_bandwidth(df, m_variables);

    for (size_t i = 0; i < m_variables.size(); ++i) {
        if constexpr (std::is_same_v<CType, double>) {
            auto sqrt = std::sqrt(m_bandwidth(i));

            VectorType aux(1);
            aux[0] = sqrt;
            m_cl_bandwidth_raw_double.push_back(aux);
        } else {
            auto casted = std::sqrt(static_cast<CType>(m_bandwidth(i)));

            VectorType aux(1);
            aux[0] = casted;
            m_cl_bandwidth_raw_float.push_back(aux);
        }

        if constexpr (contains_null) {
            auto column = df.to_eigen<false, ArrowType>(combined_bitmap, m_variables[i]);
            VectorType aux(N);
            for(int i = 0; i < N; ++i)
                aux[i] = column->data()[i];
            if constexpr (std::is_same_v<CType, double>) {
                m_training_raw_double.push_back(aux);
            } else {
                m_training_raw_float.push_back(aux);
            }
        } else {
            auto column = df.to_eigen<false, ArrowType, false>(m_variables[i]);

            VectorType aux(N);
            for(int i = 0; i < N; ++i)
                aux[i] = column->data()[i];
            if constexpr (std::is_same_v<CType, double>) {
                m_training_raw_double.push_back(aux);
            } else {
                m_training_raw_float.push_back(aux);
            }
        }
    }

    m_lognorm_const = -0.5 * static_cast<double>(m_variables.size()) * std::log(2 * util::pi<double>) -
                      0.5 * m_bandwidth.array().log().sum() - std::log(N);
}

template <typename ArrowType>
VectorXd ProductKDE::_logl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto read_data = logl_buffer<ArrowType>(df);
    if (df.null_count(m_variables) == 0) {
        if constexpr (!std::is_same_v<CType, double>)
            return read_data.template cast<double>();
        else
            return read_data;
    } else {
        auto m = df.valid_rows(m_variables);
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
    auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
    auto m = test_matrix->rows();

    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    VectorType res(m);
    _logl_impl<ArrowType>(test_matrix->data(), m, res.data());

    return res;
}

template <typename T>
void prod_logl_values_1d_mat(T* train_vector,
                        uint train_rows,
                        T* test_vector,
                        uint test_offset,
                        T *standard_deviation,
                        T lognorm_factor,
                        T* result,
                        uint test_length) {
    #define ROW(idx, rows) (idx) % (rows)
    #define COL(idx, rows) (idx) / (rows)
    for(uint i = 0; i < train_rows * test_length; ++i){

        int train_idx = ROW(i, train_rows);
        int test_idx = COL(i, train_rows);
        T d = (train_vector[train_idx] - test_vector[test_offset + test_idx]) / standard_deviation[0];
        result[i] = (-0.5*d*d) + lognorm_factor;
    }
}

template <typename T>
void add_logl_values_1d_mat(T* train_vector,
                            uint train_rows,
                            T* test_vector,
                            uint test_offset,
                            T *standard_deviation,
                            T* result,
                            uint test_length) {
    #define ROW(idx, rows) (idx) % (rows)
    #define COL(idx, rows) (idx) / (rows)
    for(uint i = 0; i < train_rows * test_length; ++i){
        int train_idx = ROW(i, train_rows);
        int test_idx = COL(i, train_rows);
        T d = (train_vector[train_idx] - test_vector[test_offset + test_idx]) / standard_deviation[0];

        result[i] += -0.5*d*d;
    }
}

template <typename ArrowType>
void ProductKDE::product_logl_mat(typename ArrowType::c_type* test_buffer,
                                  const unsigned int test_offset,
                                  const unsigned int test_length,
                                  typename ArrowType::c_type* output_mat) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    VectorType m_training_tmp(N);
    if constexpr (std::is_same_v<CType, double>)
        m_training_tmp = m_training_raw_double[0];
    else
        m_training_tmp = m_training_raw_float[0];
    auto m_training_raw = m_training_tmp.data();

    VectorType m_cl_bandwidth_tmp(1);
    if constexpr (std::is_same_v<CType, double>)
        m_cl_bandwidth_tmp = m_cl_bandwidth_raw_double[0];
    else
        m_cl_bandwidth_tmp = m_cl_bandwidth_raw_float[0];

    auto m_cl_bandwidth_raw = m_cl_bandwidth_tmp.data();


    prod_logl_values_1d_mat<CType>(m_training_raw, N, test_buffer, test_offset, m_cl_bandwidth_raw, m_lognorm_const, output_mat, test_length);


    for (size_t i = 1; i < m_variables.size(); ++i) {
        if constexpr (std::is_same_v<CType, double>)
            m_training_tmp = m_training_raw_double[i];
        else
            m_training_tmp = m_training_raw_float[i];
        auto m_training_raw = m_training_tmp.data();

        if constexpr (std::is_same_v<CType, double>)
            m_cl_bandwidth_tmp = m_cl_bandwidth_raw_double[i];
        else
            m_cl_bandwidth_tmp = m_cl_bandwidth_raw_float[i];
        auto m_cl_bandwidth_raw = m_cl_bandwidth_tmp.data();

        add_logl_values_1d_mat<CType>(m_training_raw, N, test_buffer, (i * test_length) + test_offset, m_cl_bandwidth_raw, output_mat, test_length);
    }

}

template <typename ArrowType>
void ProductKDE::_logl_impl(typename ArrowType::c_type* test_buffer, int m, typename ArrowType::c_type* res) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    auto& opencl = OpenCLConfig::get();

    auto allocated_m = std::min(m, 64);
    auto iterations = static_cast<int>(std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m)));

    VectorType tmp(N * allocated_m);
    auto tmp_raw = tmp.data();

    for (auto i = 0; i < (iterations - 1); ++i) {
        product_logl_mat<ArrowType>(test_buffer, i * allocated_m, allocated_m, tmp_raw);
        opencl.logsumexp_cols_offset<ArrowType>(tmp_raw, N, allocated_m, res, i * allocated_m, m);
    }

    auto remaining_m = m - (iterations - 1) * allocated_m;
    product_logl_mat<ArrowType>(test_buffer, m - remaining_m, remaining_m, tmp_raw);
    opencl.logsumexp_cols_offset<ArrowType>(tmp_raw, N, remaining_m, res, m - remaining_m, m);
}

template <typename ArrowType>
double ProductKDE::_slogl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto& opencl = OpenCLConfig::get();

    auto m = df.valid_rows(m_variables);
    auto logl_buff = logl_buffer<ArrowType>(df);

    VectorType result(1);
    opencl.sum1d<ArrowType>(logl_buff.data(), m, result.data());
    return static_cast<double>(result[0]);
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
                column = m_training_raw_double[i];
            } else {
                column = m_training_raw_float[i];
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