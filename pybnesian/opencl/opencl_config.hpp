#ifndef PYBNESIAN_OPENCL_OPENCL_CONFIG_HPP
#define PYBNESIAN_OPENCL_OPENCL_CONFIG_HPP

#include <cmath>
#include <arrow/api.h>
#ifdef OPENCL
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION  120
#include <CL/cl2.hpp>
#endif // OPENCL
#include <util/bit_util.hpp>
#include <iostream>
#include <pybind11/eigen.h>
#include <math.h>

#include <dataset/dataset.hpp>

// #define CL_HPP_ENABLE_EXCEPTIONS
// #ifdef CL_HPP_MINIMUM_OPENCL_VERSION
// #undef CL_HPP_MINIMUM_OPENCL_VERSION
// #endif
// #ifdef CL_HPP_TARGET_OPENCL_VERSION
// #undef CL_HPP_TARGET_OPENCL_VERSION
// #endif

#include <kernels/kernel.hpp>

#ifdef OPENCL

#define RAISE_ENQUEUEKERNEL_ERROR(enqueue)                                                                             \
    {                                                                                                                  \
        cl_int err_code = CL_SUCCESS;                                                                                  \
        err_code = enqueue;                                                                                            \
        if (err_code != CL_SUCCESS) {                                                                                  \
            throw std::runtime_error(std::string("Error enqueuing OpenCL kernel. ") + opencl::opencl_error(err_code) + \
                                     " (" + std::to_string(err_code) + ")." + " in file -> " + __FILE__ + "; code line -> " + std::to_string(__LINE__));\
        }                                                                                                              \
    }

#define OCL_CHECK(call) {                                                \
    cl_int err = call;                                                       \
    if( CL_SUCCESS != err) {                                                 \
        throw std::runtime_error(std::string("OpenCL error in file ") +  __FILE__ + " in line " + std::to_string(__LINE__) + " : Code " + std::to_string(err) + ".\n");                                   \
    } }

#endif // OPENCL

template <typename T>
void max1d_func( T* input,
            uint input_length,
            T* localMaxs,
            T* output,
            uint output_offset,
            uint global_size,
            uint local_size) {
  Kernel<T>::instance().max1d(input, input_length, localMaxs, output, output_offset, global_size, local_size);
}

template <typename T>
void sum1d_func( T* input,
            uint input_length,
            T* localMaxs,
            T* output,
            uint output_offset,
            uint global_size,
            uint local_size) {
  Kernel<T>::instance().sum1d(input, input_length, localMaxs, output, output_offset, global_size, local_size);
}

template<typename T>
void max_mat_cols_func( const T* mat,
                        uint mat_rows,
                        T* localMaxs,
                        T* output,
                        uint output_offset,
                        uint size_dim1,
                        uint size_dim2,
                        uint local_size) {
  Kernel<T>::instance().max_mat_cols(mat, mat_rows, localMaxs, output, output_offset, size_dim1, size_dim2, local_size);
}

template <typename T>
void sum_mat_cols_func( const T* mat,
                        uint mat_rows,
                        T *localMaxs,
                        T* output,
                        uint output_offset,
                        uint size_dim1,
                        uint size_dim2,
                        uint local_size) {
  Kernel<T>::instance().sum_mat_cols(mat, mat_rows, localMaxs, output, output_offset, size_dim1, size_dim2, local_size);
}

namespace opencl {

#ifdef OPENCL

const char* opencl_error(cl_int error);

#endif // OPENCL

template <typename ArrowType>
struct OpenCL_kernel_traits;

template <>
struct OpenCL_kernel_traits<arrow::DoubleType> {
    inline constexpr static const char* max1d = "max1d_double";
    inline constexpr static const char* max_mat_cols = "max_mat_cols_double";
    inline constexpr static const char* sum1d = "sum1d_double";
    inline constexpr static const char* sum_mat_cols = "sum_mat_cols_double";
    inline constexpr static const char* logsumexp_coeffs = "logsumexp_coeffs_double";
    inline constexpr static const char* solve = "solve_double";
    inline constexpr static const char* square = "square_double";
    inline constexpr static const char* logl_values_1d_mat = "logl_values_1d_mat_double";
    inline constexpr static const char* add_logl_values_1d_mat = "add_logl_values_1d_mat_double";
    inline constexpr static const char* substract = "substract_double";
    inline constexpr static const char* logl_values_mat_column = "logl_values_mat_column_double";
    inline constexpr static const char* logl_values_mat_row = "logl_values_mat_row_double";
    inline constexpr static const char* finish_lse_offset = "finish_lse_offset_double";
    inline constexpr static const char* substract_vectors = "substract_vectors_double";
    inline constexpr static const char* exp_elementwise = "exp_elementwise_double";
    inline constexpr static const char* accum_sum_mat_cols = "accum_sum_mat_cols_double";
    inline constexpr static const char* add_accum_sum_mat_cols = "add_accum_sum_mat_cols_double";
    inline constexpr static const char* normalize_accum_sum_mat_cols = "normalize_accum_sum_mat_cols_double";
    inline constexpr static const char* find_random_indices = "find_random_indices_double";
    inline constexpr static const char* conditional_means_1d = "conditional_means_1d_double";
    inline constexpr static const char* conditional_means_column = "conditional_means_column_double";
    inline constexpr static const char* conditional_means_row = "conditional_means_row_double";
    inline constexpr static const char* univariate_normal_cdf = "univariate_normal_cdf_double";
    inline constexpr static const char* normal_cdf = "normal_cdf_double";
    inline constexpr static const char* product_elementwise = "product_elementwise_double";
    inline constexpr static const char* division_elementwise = "division_elementwise_double";
    inline constexpr static const char* sum_ucv_1d = "sum_ucv_1d_double";
    inline constexpr static const char* triangular_substract_mat = "triangular_substract_mat_double";
    inline constexpr static const char* sum_ucv_mat = "sum_ucv_mat_double";
    inline constexpr static const char* ucv_diag = "ucv_diag_double";
    inline constexpr static const char* sum_ucv_diag = "sum_ucv_diag_double";
    inline constexpr static const char* copy_ucv_diag = "copy_ucv_diag_double";
};

template <>
struct OpenCL_kernel_traits<arrow::FloatType> {
    inline constexpr static const char* max1d = "max1d_float";
    inline constexpr static const char* max_mat_cols = "max_mat_cols_float";
    inline constexpr static const char* sum1d = "sum1d_float";
    inline constexpr static const char* sum_mat_cols = "sum_mat_cols_float";
    inline constexpr static const char* logsumexp_coeffs = "logsumexp_coeffs_float";
    inline constexpr static const char* solve = "solve_float";
    inline constexpr static const char* square = "square_float";
    inline constexpr static const char* logl_values_1d_mat = "logl_values_1d_mat_float";
    inline constexpr static const char* add_logl_values_1d_mat = "add_logl_values_1d_mat_float";
    inline constexpr static const char* substract = "substract_float";
    inline constexpr static const char* logl_values_mat_column = "logl_values_mat_column_float";
    inline constexpr static const char* logl_values_mat_row = "logl_values_mat_row_float";
    inline constexpr static const char* finish_lse_offset = "finish_lse_offset_float";
    inline constexpr static const char* substract_vectors = "substract_vectors_float";
    inline constexpr static const char* exp_elementwise = "exp_elementwise_float";
    inline constexpr static const char* accum_sum_mat_cols = "accum_sum_mat_cols_float";
    inline constexpr static const char* add_accum_sum_mat_cols = "add_accum_sum_mat_cols_float";
    inline constexpr static const char* normalize_accum_sum_mat_cols = "normalize_accum_sum_mat_cols_float";
    inline constexpr static const char* find_random_indices = "find_random_indices_float";
    inline constexpr static const char* conditional_means_1d = "conditional_means_1d_float";
    inline constexpr static const char* conditional_means_column = "conditional_means_column_float";
    inline constexpr static const char* conditional_means_row = "conditional_means_row_float";
    inline constexpr static const char* univariate_normal_cdf = "univariate_normal_cdf_float";
    inline constexpr static const char* normal_cdf = "normal_cdf_float";
    inline constexpr static const char* product_elementwise = "product_elementwise_float";
    inline constexpr static const char* division_elementwise = "division_elementwise_float";
    inline constexpr static const char* sum_ucv_1d = "sum_ucv_1d_float";
    inline constexpr static const char* triangular_substract_mat = "triangular_substract_mat_float";
    inline constexpr static const char* sum_ucv_mat = "sum_ucv_mat_float";
    inline constexpr static const char* ucv_diag = "ucv_diag_float";
    inline constexpr static const char* sum_ucv_diag = "sum_ucv_diag_float";
    inline constexpr static const char* copy_ucv_diag = "copy_ucv_diag_float";
};

template <typename ArrowType>
struct MaxReduction {
    using CType = typename ArrowType::c_type;

    inline constexpr static const char* reduction1d = OpenCL_kernel_traits<ArrowType>::max1d;
    inline constexpr static const char* reduction_mat = OpenCL_kernel_traits<ArrowType>::max_mat_cols;
    inline constexpr static void (* reduction1d_fun)(CType*, uint, CType*, CType*, uint, uint, uint) = max1d_func<CType>;
    inline constexpr static void (* reduction_mat_fun)(const CType*, uint, CType*, CType*, uint, uint, uint, uint) = max_mat_cols_func<CType>;
};

template <typename ArrowType>
struct SumReduction {
    using CType = typename ArrowType::c_type;

    inline constexpr static const char* reduction1d = OpenCL_kernel_traits<ArrowType>::sum1d;
    inline constexpr static const char* reduction_mat = OpenCL_kernel_traits<ArrowType>::sum_mat_cols;
    inline constexpr static void (* reduction1d_fun)(CType*, uint, CType*, CType*, uint, uint, uint) = sum1d_func<CType>;
    inline constexpr static void (* reduction_mat_fun)(const CType*, uint, CType*, CType*, uint, uint, uint, uint) = sum_mat_cols_func<CType>;
};

inline constexpr int default_platform_idx = 0;
inline constexpr int default_device_idx = 0;

class OpenCLConfig {
public:
    static OpenCLConfig& get();

#ifdef OPENCL
    cl::Context& context() { return m_context; }
    cl::Program& program() { return m_program; }
    cl::Device& device() { return m_device; }


    template <typename T>
    cl::Buffer copy_to_buffer(const T* d, int size);

    template <typename T>
    void read_from_buffer(T* dest, const cl::Buffer& from, int size);

    template <typename T>
    cl::Buffer new_buffer(int size, cl_mem_flags flags = CL_MEM_READ_WRITE);

    template <typename T>
    cl::Buffer copy_buffer(const cl::Buffer& input,
                           unsigned int offset,
                           unsigned int length,
                           cl_mem_flags flags = CL_MEM_READ_WRITE);

    template <typename T>
    void fill_buffer(cl::Buffer& b, const T value, unsigned int length);

    template <typename ArrowType>
    std::pair<cl::Buffer, uint64_t> allocate_temp_mat(size_t rows, size_t cols, size_t max_cols = 64) {
        using CType = typename ArrowType::c_type;
        auto allocated_m = std::min(cols, max_cols);
        return std::make_pair(new_buffer<CType>(rows * allocated_m), allocated_m);
    }

    cl::Kernel& kernel(const char* name);
    cl::CommandQueue& queue() { return m_queue; }

    template <typename ArrowType>
    std::vector<cl::Buffer> create_reduction1d_buffers(int length, const char* kernel_name);

    template <typename ArrowType>
    std::vector<cl::Buffer> create_reduction_mat_buffers(int length, int cols_mat, const char* kernel_name);
#endif // OPENCL

    template <typename ArrowType, typename Reduction>
    void reduction1d(typename ArrowType::c_type* input_vec, int input_length, typename ArrowType::c_type* output_buffer, int ouput_offset);

    template <typename ArrowType>
    void sum1d(typename ArrowType::c_type* input_vec, int input_length, typename ArrowType::c_type* output) {
        reduction1d<ArrowType, SumReduction<ArrowType>>(input_vec, input_length, output, 0);
    }

    template <typename ArrowType, typename Reduction>
    void reduction_cols(const typename ArrowType::c_type* input_mat, int input_rows, int input_cols, typename ArrowType::c_type* res);

    template <typename ArrowType, typename Reduction>
    void reduction_cols_offset(
        const typename ArrowType::c_type* input_mat, int input_rows, int input_cols, typename ArrowType::c_type* output_vec, int output_offset, int output_size);

    template <typename ArrowType>
    typename ArrowType::c_type* amax_cols(const typename ArrowType::c_type* input_mat, int input_rows, int input_cols, typename ArrowType::c_type* res) {
        using CType = typename ArrowType::c_type;
        CType* max = (CType*)malloc(input_cols * sizeof(CType));
        reduction_cols<ArrowType, MaxReduction<ArrowType>>(input_mat, input_rows, input_cols, max);
        return max;
    }

    template <typename ArrowType>
    void sum_cols_offset(
        const typename ArrowType::c_type* input_mat, int input_rows, int input_cols, typename ArrowType::c_type* output_vec, int output_offset, int output_size) {
        reduction_cols_offset<ArrowType, SumReduction<ArrowType>>(
            input_mat, input_rows, input_cols, output_vec, output_offset, output_size);
    }

    template <typename ArrowType>
    void logsumexp_cols_offset(
        typename ArrowType::c_type* input_mat, int input_rows, int input_cols, typename ArrowType::c_type* output_vec, int output_offset, int m);

    template <typename ArrowType>
    typename ArrowType::c_type*accum_sum_cols(typename ArrowType::c_type* mat, int input_rows, int input_cols);

#ifdef OPENCL

    size_t kernel_local_size(const char* kernel_name);

    cl_ulong kernel_local_memory(const char* kernel_name);


    size_t max_local_size() { return m_max_local_size; }

    cl_ulong max_local_memory() { return m_max_local_memory_bytes; }

#endif // OPENCL

    OpenCLConfig(const OpenCLConfig&) = delete;
    void operator=(const OpenCLConfig&) = delete;

private:
    OpenCLConfig();

#ifdef OPENCL
    cl::Context m_context;
    cl::CommandQueue m_queue;
    cl::Program m_program;
    cl::Device m_device;
    std::unordered_map<const char*, cl::Kernel> m_kernels;
    std::unordered_map<const char*, size_t> m_kernels_local_size;
    std::unordered_map<const char*, cl_ulong> m_kernels_local_memory;
    size_t m_max_local_size;
    cl_ulong m_max_local_memory_bytes;
#endif // OPENCL
};

#ifdef OPENCL

template <typename T>
cl::Buffer OpenCLConfig::copy_to_buffer(const T* d, int size) {
    cl::Buffer b = new_buffer<T>(size);

    cl_int err_code = CL_SUCCESS;
    err_code = m_queue.enqueueWriteBuffer(b, CL_TRUE, 0, sizeof(T) * size, d);

    if (err_code != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error copying OpenCL buffer. ") + opencl::opencl_error(err_code) + " (" +
                                 std::to_string(err_code) + ").");
    }

    return b;
}

template <typename T>
void OpenCLConfig::read_from_buffer(T* dest, const cl::Buffer& from, int size) {
    cl_int err_code = CL_SUCCESS;
    err_code = m_queue.enqueueReadBuffer(from, CL_TRUE, 0, sizeof(T) * size, dest);

    if (err_code != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error reading buffer. ") + opencl::opencl_error(err_code) + " (" +
                                 std::to_string(err_code) + ").");
    }
}

template <typename T>
cl::Buffer OpenCLConfig::new_buffer(int size, cl_mem_flags flags) {
    cl_int err_code = CL_SUCCESS;
    cl::Buffer b(m_context, flags, sizeof(T) * size, NULL, &err_code);

    if (err_code != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error creating OpenCL buffer of size ") + std::to_string(size) +
                                 opencl::opencl_error(err_code) + " (" + std::to_string(err_code) + ").");
    }

    return b;
}

template <typename T>
cl::Buffer OpenCLConfig::copy_buffer(const cl::Buffer& input,
                                     unsigned int offset,
                                     unsigned int length,
                                     cl_mem_flags flags) {
    cl::Buffer b = new_buffer<T>(length, flags);

    cl_int err_code = CL_SUCCESS;
    err_code = m_queue.enqueueCopyBuffer(input, b, sizeof(T) * offset, 0, sizeof(T) * length);

    if (err_code != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error copying OpenCL buffer. ") + opencl::opencl_error(err_code) + " (" +
                                 std::to_string(err_code) + ").");
    }

    return b;
}

template <typename T>
void OpenCLConfig::fill_buffer(cl::Buffer& buffer, const T value, unsigned int length) {
    cl_int err_code = CL_SUCCESS;
    err_code = m_queue.enqueueFillBuffer<T>(buffer, value, 0, length * sizeof(T));

    if (err_code != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error filling OpenCL buffer. ") + opencl::opencl_error(err_code) + " (" +
                                 std::to_string(err_code) + ").");
    }
}

template <typename ArrowType>
std::vector<cl::Buffer> OpenCLConfig::create_reduction1d_buffers(int length, const char* kernel_name) {
    using CType = typename ArrowType::c_type;
    std::vector<cl::Buffer> res;

    auto k_local_size = kernel_local_size(kernel_name);
    auto k_local_memory = kernel_local_memory(kernel_name);
    auto free_local_memory = m_max_local_memory_bytes - k_local_memory;

    auto device_max_local_size = std::min(static_cast<int>(free_local_memory / static_cast<double>(sizeof(CType))),
                                          static_cast<int>(k_local_size));

    auto current_length = length;
    while (current_length > device_max_local_size) {
        auto num_groups = static_cast<int>(
            std::ceil(static_cast<double>(current_length) / static_cast<double>(device_max_local_size)));
        auto reduc_buffer = new_buffer<CType>(num_groups);
        res.push_back(std::move(reduc_buffer));
        current_length = num_groups;
    }

    return res;
}

template <typename ArrowType>
std::vector<cl::Buffer> OpenCLConfig::create_reduction_mat_buffers(int length, int cols_mat, const char* kernel_name) {
    using CType = typename ArrowType::c_type;
    std::vector<cl::Buffer> res;

    auto k_local_size = kernel_local_size(kernel_name);
    auto k_local_memory = kernel_local_memory(kernel_name);
    auto free_local_memory = m_max_local_memory_bytes - k_local_memory;

    auto device_max_local_size = std::min(static_cast<int>(free_local_memory / static_cast<double>(sizeof(CType))),
                                          static_cast<int>(k_local_size));

    auto current_length = length;
    while (current_length > device_max_local_size) {
        auto num_groups = static_cast<int>(
            std::ceil(static_cast<double>(current_length) / static_cast<double>(device_max_local_size)));
        auto reduc_buffer = new_buffer<CType>(num_groups * cols_mat);
        res.push_back(std::move(reduc_buffer));
        current_length = num_groups;
    }

    return res;
}

#endif // OPENCL

void update_reduction_status(int& length, int& num_groups, int& local_size, int& global_size, int max_local_size);

template <typename ArrowType, typename Reduction>
void OpenCLConfig::reduction1d(typename ArrowType::c_type* input_vec, int input_length, typename ArrowType::c_type* output_buffer, int output_offset) {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto length = input_length;
    auto local_size = length;
    auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(local_size)));
    auto global_size = local_size * num_groups;

    VectorType tmp_buffer_tmp( local_size);
    auto tmp_buffer_raw = tmp_buffer_tmp.data();

    if (num_groups == 1) {
        Reduction::reduction1d_fun(input_vec, length, tmp_buffer_raw, output_buffer, output_offset, global_size, local_size);
        return;
    }
}

template <typename ArrowType, typename Reduction>
void OpenCLConfig::reduction_cols(const typename ArrowType::c_type* input_mat, int input_rows, int input_cols, typename ArrowType::c_type* res) {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto length = input_rows;
    int local_size = length;
    auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(local_size)));
    auto global_size = local_size * num_groups;

    Reduction::reduction_mat_fun(input_mat, length, NULL, res, 0, global_size, input_cols, local_size);
}

template <typename ArrowType, typename Reduction>
void OpenCLConfig::reduction_cols_offset(
    const typename ArrowType::c_type* input_mat, int input_rows, int input_cols, typename ArrowType::c_type* output_vec, int output_offset, int output_size) {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto length = input_rows;
    auto local_size = input_rows;
    auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(local_size)));
    auto global_size = local_size * num_groups;

    CType* tmp_buffer_raw = (CType*)malloc(local_size * sizeof(CType));

    Reduction::reduction_mat_fun(input_mat, length, tmp_buffer_raw, output_vec, output_offset, global_size, input_cols, local_size);
    
    free(tmp_buffer_raw);
}

template <typename ArrowType>
void OpenCLConfig::logsumexp_cols_offset(
    typename ArrowType::c_type* input_mat, int input_rows, int input_cols, typename ArrowType::c_type* output_vec, int output_offset, int m) {

    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    Kernel<CType> kernels = Kernel<CType>::instance();

    CType* max_buffer_raw = amax_cols<ArrowType>(input_mat, input_rows, input_cols, NULL);
    kernels.logsumexp_coeffs(input_mat, input_rows, max_buffer_raw, input_rows * input_cols);
    sum_cols_offset<ArrowType>(input_mat, input_rows, input_cols, output_vec, static_cast<unsigned int>(output_offset), m);
    kernels.finish_lse_offset(output_vec, output_offset, max_buffer_raw, input_cols);
    free(max_buffer_raw);
}

template <typename T>
void add_accum_sum_mat_cols(T* mat,
                            uint mat_rows,
                            uint mat_offset,
                            uint size_per_group,
                            uint num_groups,
                            T* sums,
                            uint size_dim1,
                            uint size_dim2) {
    #define IDX(i, j, rows) (i) + ((j)*(rows))
    for(uint row_id = 0; row_id < size_dim1; ++row_id)
        for(uint col_id = 0; col_id < size_dim2; ++col_id)
            mat[IDX(row_id + mat_offset, col_id, mat_rows)] += sums[IDX((row_id / size_per_group)+1, col_id, num_groups)];
}

template <typename T>
void accum_sum_mat_cols(T* mat,
                        uint mat_rows,
                        T* local_block,
                        T* sums,
                        uint size_dim1,
                        uint size_dim2,
                        uint local_size) {
    uint group_size = local_size;
    uint num_groups = size_dim1/local_size;
    uint offset;

    for(uint col_id = 0; col_id < size_dim2; ++col_id){
        for(uint group_id = 0; group_id < num_groups; ++group_id){
            for(uint local_id = 0; local_id < group_size; ++local_id){
                uint row_id = 0 + local_id + group_id * group_size;
                if (2*row_id+1 < mat_rows) {
                    local_block[2*local_id] = mat[IDX(2*row_id, col_id, mat_rows)];
                    local_block[2*local_id+1] = mat[IDX(2*row_id+1, col_id, mat_rows)];
                } else {
                    local_block[2*local_id] = 0;
                    local_block[2*local_id+1] = 0;
                }
            }

            if (group_id == num_groups-1) {
                local_block[mat_rows - 2*group_id*group_size - 1] = mat[IDX(mat_rows-1, col_id, mat_rows)];
            }

            offset = 1;
            /* build the sum in place up the tree */
            for (uint d = group_size; d > 0; d /= 2){
                for(uint local_id = 0; local_id < group_size; ++local_id){
                    uint row_id = 0 + local_id + group_id * group_size;

                    if (local_id < d) {
                        uint ai = offset * (2 * local_id + 1) - 1;
                        uint bi = offset * (2 * local_id + 2) - 1;

                        local_block[bi] += local_block[ai];
                    }
                }
                offset *= 2;
            }

            for(uint local_id = 0; local_id < group_size; ++local_id){
                uint row_id = 0 + local_id + group_id * group_size;
                /* store the value in sum buffer before making it to 0 */
                sums[IDX(group_id, col_id, num_groups)] = local_block[2*group_size - 1];
            }

            offset = 1;
            for (uint d = group_size; d > 0; d /= 2)
                offset *= 2;

            // /* scan back down the tree */

            // /* clear the last element */
            local_block[2*group_size - 1] = 0;

            /* traverse down the tree building the scan in the place */
            for (uint d = 1; d <= group_size; d *= 2) {
                offset /= 2;
                for (uint local_id = 0; local_id < group_size; ++local_id) {
                    uint row_id = 0 + local_id + group_id * group_size;

                    if (local_id < d) {
                        uint ai = offset * (2 * local_id + 1) - 1;
                        uint bi = offset * (2 * local_id + 2) - 1;

                        float t = local_block[ai];
                        local_block[ai] = local_block[bi];
                        local_block[bi] += t;
                    }
                }
            }

            for (uint local_id = 0; local_id < group_size; ++local_id) {
                uint row_id = 0 + local_id + group_id * group_size;

                // write the results back to global memory

                if ((2*row_id+1) < mat_rows) {
                    mat[IDX(2*row_id, col_id, mat_rows)] = local_block[2*local_id];
                    mat[IDX(2*row_id+1, col_id, mat_rows)] = local_block[2*local_id+1];
                } else if (2*row_id < mat_rows) {
                    mat[IDX(2*row_id, col_id, mat_rows)] = local_block[2*local_id];
                }
            }
        }
    }
}

template <typename ArrowType>
typename ArrowType::c_type* OpenCLConfig::accum_sum_cols(typename ArrowType::c_type* mat, int input_rows, int input_cols) {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto local_wg = input_rows/2;
    auto num_groups = static_cast<int>(std::ceil(static_cast<double>(input_rows) / static_cast<double>(2 * local_wg)));
    auto global_wg = static_cast<int>(std::ceil(static_cast<double>(num_groups * local_wg)));

    CType* group_sums_raw = (CType*)malloc(num_groups * input_cols * sizeof(CType));

    CType* local_block_raw = (CType*)malloc(2 * local_wg * sizeof(CType));

    accum_sum_mat_cols<CType>(mat, input_rows, local_block_raw, group_sums_raw, global_wg, input_cols, local_wg);
    free(local_block_raw);

    return group_sums_raw;
}

}  // namespace opencl

#endif  // PYBNESIAN_OPENCL_OPENCL_CONFIG_HPP
