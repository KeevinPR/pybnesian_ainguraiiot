#include <kde/KDE.hpp>
#include <arrow/python/helpers.h>

namespace kde {

void KDE::copy_bandwidth_opencl() {
    auto d = m_variables.size();
    auto llt_cov = m_bandwidth.llt();
    auto llt_matrix = llt_cov.matrixLLT();

    m_lognorm_const = -llt_matrix.diagonal().array().log().sum() -
                      0.5 * m_variables.size() * std::log(2 * util::pi<double>) - std::log(N);

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            Matrix<double, Dynamic, 1> aux_mat(d * d);
            double* aux = aux_mat.data();
            for(int i = 0; i < d * d; ++i){
                aux[i] = llt_matrix.data()[i];
            }
            m_H_cholesky_raw_double = aux_mat;
            break;
        }
        case Type::FLOAT: {
            MatrixXf casted_cholesky = llt_matrix.template cast<float>();
            Matrix<float, Dynamic, 1> aux_mat(d * d);
            float* aux = aux_mat.data();
            for(int i = 0; i < d * d; ++i){
                aux[i] = casted_cholesky.data()[i];
            }
            m_H_cholesky_raw_float = aux_mat;
            break;
        }
        default:
            throw std::invalid_argument("Unreachable code.");
    }
}

DataFrame KDE::training_data() const {
    check_fitted();
    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return _training_data<arrow::DoubleType>();
        case Type::FLOAT:
            return _training_data<arrow::FloatType>();
        default:
            throw std::invalid_argument("Unreachable code.");
    }
}

void KDE::fit(const DataFrame& df) {
    m_training_type = df.same_type(m_variables);

    bool contains_null = df.null_count(m_variables) > 0;

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (contains_null)
                _fit<arrow::DoubleType, true>(df);
            else
                _fit<arrow::DoubleType, false>(df);
            break;
        }
        case Type::FLOAT: {
            if (contains_null)
                _fit<arrow::FloatType, true>(df);
            else
                _fit<arrow::FloatType, false>(df);
            break;
        }
        default:
            throw std::invalid_argument("Wrong data type to fit KDE. [double] or [float] data is expected.");
    }

    m_fitted = true;
}

VectorXd KDE::logl(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _logl<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _logl<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

double KDE::slogl(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _slogl<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _slogl<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

py::tuple KDE::__getstate__() const {
    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return __getstate__<arrow::DoubleType>();
        case Type::FLOAT:
            return __getstate__<arrow::FloatType>();
        default:
            // Not fitted model.
            return __getstate__<arrow::DoubleType>();
    }
}

KDE KDE::__setstate__(py::tuple& t) {
    if (t.size() != 8) throw std::runtime_error("Not valid KDE.");

    KDE kde(t[0].cast<std::vector<std::string>>());

    kde.m_fitted = t[1].cast<bool>();
    kde.m_bselector = t[2].cast<std::shared_ptr<BandwidthSelector>>();
    BandwidthSelector::keep_python_alive(kde.m_bselector);

    if (kde.m_fitted) {
        kde.m_bandwidth = t[3].cast<MatrixXd>();
        kde.m_lognorm_const = t[5].cast<double>();
        kde.N = static_cast<size_t>(t[6].cast<int>());
        kde.m_training_type = pyarrow::GetPrimitiveType(static_cast<arrow::Type::type>(t[7].cast<int>()));

        auto llt_cov = kde.m_bandwidth.llt();
        auto llt_matrix = llt_cov.matrixLLT();


        switch (kde.m_training_type->id()) {
            case Type::DOUBLE: {
                
                Matrix<double, Dynamic, 1> aux_mat(kde.m_variables.size() * kde.m_variables.size());
                double* aux = aux_mat.data();
                for(int i = 0; i < kde.m_variables.size() * kde.m_variables.size(); ++i){
                    aux[i] = llt_matrix.data()[i];
                }
                kde.m_H_cholesky_raw_double = aux_mat;

                auto training_data = t[4].cast<VectorXd>();

                Matrix<double, Dynamic, 1> aux_mat2(kde.N * kde.m_variables.size());
                double* aux2 = aux_mat2.data();
                for(int i = 0; i < kde.N * kde.m_variables.size(); ++i){
                    aux2[i] = training_data.data()[i];
                }
                kde.m_training_raw_double = aux_mat2;

                break;
            }
            case Type::FLOAT: {
                MatrixXf casted_cholesky = llt_matrix.template cast<float>();

                Matrix<float, Dynamic, 1> aux_mat(kde.m_variables.size() * kde.m_variables.size());
                float* aux = aux_mat.data();
                for(int i = 0; i < kde.m_variables.size() * kde.m_variables.size(); ++i){
                    aux[i] = casted_cholesky.data()[i];
                }
                kde.m_H_cholesky_raw_float = aux_mat;

                auto training_data = t[4].cast<VectorXf>();

                Matrix<float, Dynamic, 1> aux_mat2(kde.N * kde.m_variables.size());
                float* aux2 = aux_mat2.data();
                for(int i = 0; i < kde.N * kde.m_variables.size(); ++i){
                    aux2[i] = training_data.data()[i];
                }
                kde.m_training_raw_float = aux_mat2;
                break;
            }
            default:
                throw std::runtime_error("Not valid data type in KDE.");
        }
    }

    return kde;
}

}  // namespace kde
