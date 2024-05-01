#include <kde/KDE.hpp>
#include <arrow/python/helpers.h>

namespace kde {

void KDE::copy_bandwidth() {
    auto d = m_variables.size();
    auto llt_cov = m_bandwidth.llt();
    auto llt_matrix = llt_cov.matrixLLT();

    m_lognorm_const = -llt_matrix.diagonal().array().log().sum() -
                      0.5 * m_variables.size() * std::log(2 * util::pi<double>) - std::log(N);

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            m_H_cholesky_double = Matrix<double, Dynamic, 1>(d * d);
            std::memcpy(m_H_cholesky_double.data(), llt_matrix.data(), d*d*sizeof(double));
            break;
        }
        case Type::FLOAT: {
            MatrixXf casted_cholesky = llt_matrix.template cast<float>();
            m_H_cholesky_float = Matrix<float, Dynamic, 1>(d * d);
            std::memcpy(m_H_cholesky_float.data(), casted_cholesky.data(), d*d*sizeof(float));
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
        int nvar = kde.m_variables.size();

        switch (kde.m_training_type->id()) {
            case Type::DOUBLE: {
                kde.m_H_cholesky_double = Matrix<double, Dynamic, 1>(nvar * nvar);
                std::memcpy(kde.m_H_cholesky_double.data(), llt_matrix.data(), nvar * nvar*sizeof(double));

                auto training_data = t[4].cast<VectorXd>();
                kde.m_training_double = Matrix<double, Dynamic, 1>(kde.N * nvar);
                std::memcpy(kde.m_training_double.data(), training_data.data(), kde.N * nvar*sizeof(double));
                break;
            }
            case Type::FLOAT: {
                MatrixXf casted_cholesky = llt_matrix.template cast<float>();
                kde.m_H_cholesky_float = Matrix<float, Dynamic, 1>(nvar * nvar);
                std::memcpy(kde.m_H_cholesky_float.data(), casted_cholesky.data(), nvar * nvar*sizeof(float));

                auto training_data = t[4].cast<VectorXf>();
                kde.m_training_float = Matrix<float, Dynamic, 1>(kde.N * nvar);
                std::memcpy(kde.m_training_float.data(), training_data.data(), kde.N*nvar*sizeof(float));
                break;
            }
            default:
                throw std::runtime_error("Not valid data type in KDE.");
        }
    }

    return kde;
}
}  // namespace kde
