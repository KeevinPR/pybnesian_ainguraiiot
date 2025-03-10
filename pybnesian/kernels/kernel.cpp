#ifndef KERNEL_CPP
#define KERNEL_CPP

#include <kernels/kernel.hpp>
#include <iostream>
#include <omp.h>

template <class T>
Kernel<T>::Kernel(){
}

template <class T>
void Kernel<T>::max1d(const T* input,
                      uint input_length,
                      T* localMaxs,
                      T* output,
                      uint output_offset,
                      uint global_size,
                      uint local_size) {

    uint group_size = local_size;
    uint num_groups = global_size/local_size;
    uint offset;

    for(uint group_id = 0; group_id < num_groups; ++group_id){
        if (group_id == num_groups-1) group_size = input_length - group_id*group_size;
        for(uint local_id = 0; local_id < group_size; ++local_id){
            uint global_id = 0 + local_id + group_id * group_size;
            if (group_id == num_groups-1) {
                if (global_id < input_length) {
                    localMaxs[local_id] = input[global_id];
                }
            } else {
                localMaxs[local_id] = input[global_id];
            }
        }

        while (group_size > 1) {
            int stride = group_size / 2;
            for(uint local_id = 0; local_id < group_size; ++local_id){
                if (group_size % 2 == 0) {
                    if (local_id < stride)
                        localMaxs[local_id] = std::max(localMaxs[local_id], localMaxs[local_id + stride]);
                } else {
                    if (local_id < stride)
                        localMaxs[local_id] = std::max(localMaxs[local_id + 1], localMaxs[local_id + 1 + stride]);
                }
            }

            if (group_size % 2 == 0) group_size = group_size / 2;
            else group_size = (group_size / 2) + 1;
        }

        output[output_offset + group_id] = localMaxs[0];
    }   
}

template <class T>
void Kernel<T>::sum1d(const T* input,
                      uint input_length,
                      T* localMaxs,
                      T* output,
                      uint output_offset,
                      uint global_size,
                      uint local_size) {
    uint group_size = local_size;
    uint num_groups = global_size/local_size;

    for (uint group_id = 0; group_id < num_groups; ++group_id) {
        if (group_id == num_groups-1) group_size = input_length - group_id*group_size;
        for (uint local_id = 0; local_id < group_size; ++local_id) {
            uint global_id = 0 + local_id + group_id * group_size;
            if (group_id == num_groups-1) {
                if (global_id < input_length) {
                    localMaxs[local_id] = input[global_id];
                }
            } else {
                localMaxs[local_id] = input[global_id];
            }
        }

        while (group_size > 1) {
            uint stride = group_size / 2;
            for (uint local_id = 0; local_id < group_size; ++local_id) {
                if (group_size % 2 == 0) {
                    if (local_id < stride) {
                        localMaxs[local_id] += localMaxs[local_id + stride];
                    }
                } else {
                    if (local_id < stride) {
                        localMaxs[local_id + 1] += localMaxs[local_id + 1 + stride];
                    }
                }
            }
            if (group_size % 2 == 0) group_size = group_size / 2;
            else group_size = (group_size / 2) + 1;
        }

        output[output_offset + group_id] = localMaxs[0];
    }   
}

template <class T>
void Kernel<T>::max_mat_cols(const T* mat,
                             uint mat_rows,
                            //  T* localMaxs,
                             T* output,
                             uint output_offset,
                            //  uint size_dim1,
                             uint size_dim2) {
#pragma omp parallel
#pragma omp single
{
    #pragma omp taskloop num_tasks(omp_get_num_threads())
    for (uint j = 0; j < size_dim2; ++j) {
        T max = -std::numeric_limits<T>::max();
        for (uint i = 0; i < mat_rows; ++i) {
            max = std::max(mat[IDX(i, j, mat_rows)], max);
        }
        output[output_offset + j] = max;
    }
}
}

template <class T>
void Kernel<T>::sum_mat_cols(const T* mat,
                             uint mat_rows,
                            //  T *localMaxs,
                             T* output,
                             uint output_offset,
                            //  uint size_dim1,
                             uint size_dim2) {
#pragma omp parallel
#pragma omp single
{
    #pragma omp taskloop num_tasks(omp_get_num_threads())
    for (uint j = 0; j < size_dim2; ++j) {
        T sum = 0;
        for (uint i = 0; i < mat_rows; ++i)
            sum+=mat[IDX(i, j, mat_rows)];
        output[output_offset + j] = sum;
    }
}
}

template <class T>
void Kernel<T>::logsumexp_coeffs(T* input,
                                 uint input_rows,
                                 T* max,
                                 uint size) {
#pragma omp parallel
#pragma omp single
{
    #pragma omp taskloop num_tasks(omp_get_num_threads())
    for (uint idx = 0; idx < size; ++idx) {
        uint col = COL(idx, input_rows);
        input[idx] = exp(input[idx] - max[col]);
    }
}
}

template <class T>
void Kernel<T>::finish_lse_offset(T* res,
                                  uint res_offset,
                                  T* max_vec,
                                  uint size) {
    for (uint idx = 0; idx < size; ++idx)
        res[idx + res_offset] = log(res[idx + res_offset]) + max_vec[idx];
}


template <class T>
void Kernel<T>::conditional_means_column(const T* training_data,
                                         uint training_data_physical_rows,
                                         const T* substract_evidence,
                                         uint substract_evidence_physical_rows,
                                         const T* transform_vector,
                                         uint evidence_columns,
                                         T* res,
                                         uint res_col_idx,
                                         uint res_physical_rows) {
    for (uint i = 0; i < substract_evidence_physical_rows; ++i) {
        T mean = training_data[IDX(i, 0, training_data_physical_rows)];

        for (uint j = 0; j < evidence_columns; ++j)
            mean += transform_vector[j]*substract_evidence[IDX(i, j, substract_evidence_physical_rows)];

        res[IDX(i, res_col_idx, res_physical_rows)] = mean;
    }
}

template <class T>
void Kernel<T>::conditional_means_row(const T* training_data,
                                      uint training_data_physical_rows,
                                      const T* substract_evidence,
                                      uint substract_evidence_physical_rows,
                                      const T* transform_vector,
                                      uint evidence_columns,
                                      T* res,
                                      uint res_row_idx,
                                      uint res_physical_rows) {
    for (uint i = 0; i < substract_evidence_physical_rows; ++i) {
        T mean = training_data[IDX(res_row_idx, 0, training_data_physical_rows)];

        for (uint j = 0; j < evidence_columns; ++j)
            mean -= transform_vector[j]*substract_evidence[IDX(i, j, substract_evidence_physical_rows)];

        res[IDX(res_row_idx, i, res_physical_rows)] = mean;
    }
}

template <class T>
void Kernel<T>::logl_values_mat_row_new(T* square_data,
                                        uint square_cols,
                                        T* sol_mat,
                                        uint sol_rows,
                                        uint sol_row_idx,
                                        T lognorm_factor,
                                        uint square_rows) {
    for (uint test_idx = 0; test_idx < square_rows; test_idx++) {
        uint sol_idx = IDX(sol_row_idx, test_idx, sol_rows);

        auto summation = square_data[IDX(test_idx, 0, square_rows)];
        for (uint i = 1; i < square_cols; i++)
            summation += square_data[IDX(test_idx, i, square_rows)];
        sol_mat[sol_idx] = (-0.5 * summation) + lognorm_factor;
    }
}

template <class T>
void Kernel<T>::square_inplace_new(T* X, int n) {
    for(int i = 0; i < n; i++) X[i] *= X[i];
}

template <class T>
void Kernel<T>::logl_values_mat_column_new(const T* square_data,
                                           uint square_cols,
                                           T* sol_mat,
                                           uint sol_rows,
                                           uint sol_col_idx,
                                           T lognorm_factor,
                                           uint square_rows) {
    for (uint test_idx = 0; test_idx < square_rows; test_idx++) {
        uint sol_idx = IDX(test_idx, sol_col_idx, sol_rows);

        auto summation = square_data[IDX(test_idx, 0, square_rows)];
        for (uint i = 1; i < square_cols; i++)
            summation += square_data[IDX(test_idx, i, square_rows)];
        sol_mat[sol_idx] = (-0.5 * summation) + lognorm_factor;
    }

}

template <class T>
void Kernel<T>::substract_domain_specific_new(const T* A,
                                              int A_physical_rows,
                                              int A_offset,
                                              int A_rows,
                                              const T* B,
                                              int B_physical_rows,
                                              int B_offset,
                                              int B_idx,
                                              int num_cols,
                                              T* C) {
    for (int i = 0; i < A_rows*num_cols; i++) {
        auto r = (i%A_rows)+A_offset;
        auto c = i/A_rows;
        auto B_index = c*B_physical_rows+(B_offset+B_idx);
        auto A_index = c*A_physical_rows+(r);
        C[i] = B[B_index] - A[A_index];
    }
}

template <class T>
void Kernel<T>::solve_specific_new(T* diff_matrix, 
                                   int diff_matrix_rows,
                                   int matrices_cols, 
                                   const T* cholesky_matrix) {
    for (int i = 0; i < diff_matrix_rows; i++) {
        for (int c = 0; c < matrices_cols; c++) {
            for (int j = 0; j < c; j++)
                diff_matrix[c*diff_matrix_rows+i] -= cholesky_matrix[j*matrices_cols+c] * diff_matrix[j*diff_matrix_rows+i];
        diff_matrix[c*diff_matrix_rows+i] /= cholesky_matrix[c*matrices_cols+c];
        }
    }
}

template <class T>
void Kernel<T>::logl_values_1d_mat(const T* train_vector,
                                   uint train_rows,
                                   const T* test_vector,
                                   uint test_offset,
                                   const T *standard_deviation,
                                   T lognorm_factor,
                                   T* result,
                                   uint test_length) {
    int size = train_rows * test_length;
#pragma omp parallel
#pragma omp single
{
    int n_tasks = omp_get_num_threads();
    int gs = std::max(size / n_tasks, 1);
    #pragma omp taskloop grainsize(gs)
    for (int i = 0; i < size; ++i) {
        int train_idx = ROW(i, train_rows);
        int test_idx = COL(i, train_rows);
        T d = (train_vector[train_idx] - test_vector[test_offset + test_idx]) / standard_deviation[0];
        result[i] = (-0.5*d*d) + lognorm_factor;
    }
}
}

template <class T>
void Kernel<T>::conditional_means_1d(const T* train_mat,
                                     uint train_physical_rows,
                                     const T* test_vector,
                                     uint test_physical_rows,
                                     uint test_offset,
                                     const T* transform_mean,
                                     T* result,
                                     uint test_length) {
    for (uint i = 0; i < train_physical_rows * test_length; ++i) {
        int train_idx = ROW(i, train_physical_rows);
        int test_idx = COL(i, train_physical_rows);

        result[i] = train_mat[IDX(train_idx, 0, train_physical_rows)] +
                    transform_mean[0]*(test_vector[IDX(test_offset + test_idx, 0, test_physical_rows)] -
                                       train_mat[IDX(train_idx, 1, train_physical_rows)]);
    }
}

template <class T>
void Kernel<T>::prod_logl_values_1d_mat(const T* train_vector,
                                        uint train_rows,
                                        T* test_vector,
                                        uint test_offset,
                                        T standard_deviation,
                                        T lognorm_factor,
                                        T* result,
                                        uint test_length) {
    for (uint i = 0; i < train_rows * test_length; ++i) {
        int train_idx = ROW(i, train_rows);
        int test_idx = COL(i, train_rows);
        T d = (train_vector[train_idx] - test_vector[test_offset + test_idx]) / standard_deviation;
        result[i] = (-0.5*d*d) + lognorm_factor;
    }
}

template <class T>
void Kernel<T>::add_logl_values_1d_mat(const T* train_vector,
                                       uint train_rows,
                                       T* test_vector,
                                       uint test_offset,
                                       T standard_deviation,
                                       T* result,
                                       uint test_length) {
    for (uint i = 0; i < train_rows * test_length; ++i) {
        int train_idx = ROW(i, train_rows);
        int test_idx = COL(i, train_rows);
        T d = (train_vector[train_idx] - test_vector[test_offset + test_idx]) / standard_deviation;
        result[i] += -0.5*d*d;
    }
}

template <class T>
void Kernel<T>::substract_vectors(T* v1, T* v2, uint m) {
    for (uint idx = 0; idx < m; ++idx)
        v1[idx] -= v2[idx];
}

template <class T>
void Kernel<T>::exp_elementwise(T* mat, uint size) {
    for (uint idx = 0; idx < size; ++idx)
        mat[idx] = exp(mat[idx]);
}

template <class T>
void Kernel<T>::normalize_accum_sum_mat_cols(T* mat,
                                             uint mat_rows,
                                             T* sums,
                                             uint size_dim1,
                                             uint size_dim2) {
    for (uint row_id = 0; row_id < size_dim1; ++row_id)
        for (uint col_id = 0; col_id < size_dim2; ++col_id)
            mat[IDX(row_id + 1, col_id, mat_rows)] /= sums[col_id];
}

template <class T>
void Kernel<T>::find_random_indices(T* mat,
                                    uint mat_rows,
                                    uint mat_offset,
                                    T* random_numbers,
                                    int* indices,
                                    uint size_dim1,
                                    uint size_dim2) {
    for (uint row_id = 0; row_id < size_dim1; ++row_id) {
        for (uint col_id = 0; col_id < size_dim2; ++col_id) {
            T rn = random_numbers[mat_offset + col_id];
            if (mat[IDX(row_id, col_id, mat_rows)] <= rn && rn < mat[IDX(row_id+1, col_id, mat_rows)])
                indices[mat_offset + col_id] = row_id;
        }
    }
}

template <class T>
void Kernel<T>::univariate_normal_cdf(const T* means,
                                      uint means_physical_rows,
                                      const T* x,
                                      uint x_offset,
                                      T inv_std,
                                      T inv_N,
                                      T* cdf_mat,
                                      uint m) {
    for (uint i = 0; i < means_physical_rows * m; ++i) {
        int means_idx = ROW(i, means_physical_rows);
        int x_idx = COL(i, means_physical_rows);
        cdf_mat[i] = inv_N*(0.5*erfc(M_SQRT1_2l * inv_std * -(x[x_offset + x_idx] - means[means_idx])));
    }
}

template <class T>
void Kernel<T>::normal_cdf(T* means,
                           uint means_physical_rows,
                           T* x,
                           uint x_offset,
                           T inv_std,
                           uint size) {
    for (uint i = 0; i < size; ++i) {
        int col_idx = COL(i, means_physical_rows);
        means[i] = 0.5*erfc(M_SQRT1_2 * inv_std * (means[i] - x[x_offset + col_idx]));
    }
}

template <class T>
void Kernel<T>::product_elementwise(T* mat1, T* mat2, uint size) {
    for (uint i = 0; i < size; ++i)
        mat1[i] *= mat2[i];
}

template <class T>
void Kernel<T>::division_elementwise(T* mat1,
                                     uint mat1_offset,
                                     T* mat2,
                                     uint size) {
    for (uint i = 0; i < size; ++i)
        mat1[mat1_offset + i] /= mat2[i];
}

template <class T>
void Kernel<T>::accum_sum_mat_cols(T* mat,
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
                    if (local_id < d) {
                        uint ai = offset * (2 * local_id + 1) - 1;
                        uint bi = offset * (2 * local_id + 2) - 1;

                        local_block[bi] += local_block[ai];
                    }
                }
                offset *= 2;
            }

            for(uint local_id = 0; local_id < group_size; ++local_id){
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

// AUXILIAR KERNELS

template <class T>
void Kernel<T>::logsumexp_cols_offset(T* input_mat, 
                                      int input_rows, 
                                      int input_cols, 
                                      T* output_vec, 
                                      int output_offset) {
    T* max_buffer_raw = (T*)malloc(input_cols * sizeof(T));
    amax_cols(input_mat, input_rows, input_cols, max_buffer_raw);
    logsumexp_coeffs(input_mat, input_rows, max_buffer_raw, input_rows * input_cols);
    sum_cols_offset(input_mat, input_rows, input_cols, output_vec, static_cast<unsigned int>(output_offset));
    finish_lse_offset(output_vec, output_offset, max_buffer_raw, input_cols);
    free(max_buffer_raw);
}

template <class T>
void Kernel<T>::sum_cols_offset(const T* input_mat, 
                                int input_rows, 
                                int input_cols, 
                                T* output_vec, 
                                int output_offset) {
    reduction_cols_offset<SumReduction>(input_mat, input_rows, input_cols, output_vec, output_offset);
}

template <class T> template<typename Reduction>
void Kernel<T>::reduction_cols_offset(const T* input_mat, 
                                      int input_rows, 
                                      int input_cols, 
                                      T* output_vec, 
                                      int output_offset) {
    (this->*Reduction::reduction_mat_fun)(input_mat, input_rows, output_vec, output_offset, input_cols);
}

template <class T>
void Kernel<T>::amax_cols(const T* input_mat, 
                          int input_rows, 
                          int input_cols, 
                          T* res) {
    reduction_cols<MaxReduction>(input_mat, input_rows, input_cols, res);
}

template <class T> template<typename Reduction>
void Kernel<T>::reduction_cols(const T* input_mat, 
                               int input_rows, 
                               int input_cols, 
                               T* res) {
    (this->*Reduction::reduction_mat_fun)(input_mat, input_rows, res, 0, input_cols);
}

template <class T>
void Kernel<T>::accum_sum_cols(T* mat, int input_rows, int input_cols, T* res) {
    T* local_block_raw = (T*)malloc(input_cols * sizeof(T));
    accum_sum_mat_cols(mat, input_rows, local_block_raw, res, input_cols/2, input_cols, input_cols/2);
    free(local_block_raw);
}

template <class T>
void Kernel<T>::sum1d(const T* input_vec, int input_length, T* output) {
    reduction1d<SumReduction>(input_vec, input_length, output, 0);
}

template <typename T> template <typename Reduction>
void Kernel<T>::reduction1d(const T* input_vec, int input_length, T* output_buffer, int output_offset) {
    auto length = input_length;
    auto local_size = length;
    auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(local_size)));
    auto global_size = local_size * num_groups;

    T* tmp = (T*)malloc(local_size*sizeof(T));

    if (num_groups == 1) {
        (this->*Reduction::reduction1d_fun)(input_vec, length, tmp, output_buffer, output_offset, global_size, local_size);
        free(tmp);
        return;
    }
    free(tmp);
}

#endif
