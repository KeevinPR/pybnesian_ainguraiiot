#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <cstddef>
#include <stdint.h>
#include <math.h>
#include <cmath>

#define ROW(idx, rows) (idx) % (rows)
#define COL(idx, rows) (idx) / (rows)
#define IDX(i, j, rows) (i) + ((j)*(rows))

template <class T>
class Kernel {

    public:
        static Kernel &instance(){
            static Kernel K;
            return K;
        };

        // KERNELS
        void max1d(const T* input, uint input_length, T* localMaxs, T* output, uint output_offset, uint global_size, uint local_size);
        void sum1d(const T* input, uint input_length, T* localMaxs, T* output, uint output_offset, uint global_size, uint local_size);
        void max_mat_cols(const T* mat, uint mat_rows, T* output, uint output_offset, uint size_dim2);
        void sum_mat_cols(const T* mat, uint mat_rows, T* output, uint output_offset, uint size_dim2);
        void logsumexp_coeffs(T* input, uint input_rows, T* max, uint size);
        void finish_lse_offset(T* res, uint res_offset, T* max_vec, uint size);
        void conditional_means_column(const T* training_data, uint training_data_physical_rows, const T* substract_evidence, uint substract_evidence_physical_rows, const T* transform_vector, uint evidence_columns, T* res, uint res_col_idx, uint res_physical_rows);
        void conditional_means_row( const T* training_data, uint training_data_physical_rows, const T* substract_evidence, uint substract_evidence_physical_rows, const T* transform_vector, uint evidence_columns, T* res, uint res_row_idx, uint res_physical_rows);
        void logl_values_mat_row_new(T* square_data, uint square_cols, T* sol_mat, uint sol_rows, uint sol_row_idx, T lognorm_factor, uint square_rows);
        void logl_values_mat_column_new(const T* square_data, uint square_cols, T* sol_mat, uint sol_rows, uint sol_col_idx, T lognorm_factor, uint square_rows);
        void square_inplace_new(T* X, int n);
        void solve_specific_new(T* diff_matrix, int diff_matrix_rows, int matrices_cols, const T* cholesky_matrix);
        void substract_domain_specific_new(const T* A, int A_physical_rows, int A_offset, int A_rows, const T* B, int B_physical_rows, int B_offset, int B_idx, int num_cols, T* C);
        void conditional_means_1d(const T* train_mat, uint train_physical_rows, const T* test_vector, uint test_physical_rows, uint test_offset, const T* transform_mean, T* result, uint test_length);
        void logl_values_1d_mat(const T* train_vector, uint train_rows, const T* test_vector, uint test_offset, const T *standard_deviation, T lognorm_factor, T* result, uint test_length);
        void prod_logl_values_1d_mat(const T* train_vector, uint train_rows, T* test_vector, uint test_offset, T standard_deviation, T lognorm_factor, T* result, uint test_length);
        void add_logl_values_1d_mat(const T* train_vector, uint train_rows, T* test_vector, uint test_offset, T standard_deviation, T* result, uint test_length);
        void substract_vectors(T* v1, T* v2, uint m);
        void exp_elementwise(T* mat, uint size);
        void find_random_indices(T* mat, uint mat_rows, uint mat_offset, T* random_numbers, int* indices, uint size_dim1, uint size_dim2);
        void normalize_accum_sum_mat_cols(T* mat, uint mat_rows, T* sums, uint size_dim1, uint size_dim2);
        void univariate_normal_cdf(const T* means, uint means_physical_rows, const T* x, uint x_offset, T inv_std, T inv_N, T* cdf_mat, uint m);
        void normal_cdf(T* means, uint means_physical_rows, T* x, uint x_offset, T inv_std, uint size);
        void product_elementwise(T* mat1, T* mat2, uint size);
        void division_elementwise(T* mat1, uint mat1_offset, T* mat2, uint size);
        void accum_sum_mat_cols(T* mat, uint mat_rows, T* local_block, T* sums, uint size_dim1, uint size_dim2, uint local_size);
        // AUXILIAR KERNELS
        void logsumexp_cols_offset(T* input_mat, int input_rows, int input_cols, T* output_vec, int output_offset);
        void sum_cols_offset(const T* input_mat, int input_rows, int input_cols, T* output_vec, int output_offset);
        template <typename Reduction>
        void reduction_cols_offset(const T* input_mat, int input_rows, int input_cols, T* output_vec, int output_offset);
        void amax_cols(const T* input_mat, int input_rows, int input_cols, T* res);
        template <typename Reduction>
        void reduction_cols(const T* input_mat, int input_rows, int input_cols, T* res);
        void accum_sum_cols(T* mat, int input_rows, int input_cols, T* res);
        void sum1d(const T* input_vec, int input_length, T* output);
        template <typename Reduction>
        void reduction1d(const T* input_vec, int input_length, T* output_buffer, int output_offset);

        // REDUCTIONS

        struct MaxReduction {
            inline constexpr static void (Kernel<T>::* reduction1d_fun)(const T*, uint, T*, T*, uint, uint, uint) = &Kernel<T>::max1d;
            inline constexpr static void (Kernel<T>::* reduction_mat_fun)(const T*, uint, T*, uint, uint) = &Kernel<T>::max_mat_cols;
        };

        struct SumReduction {
            inline constexpr static void (Kernel<T>::* reduction1d_fun)(const T*, uint, T*, T*, uint, uint, uint) = &Kernel<T>::sum1d;
            inline constexpr static void (Kernel<T>::* reduction_mat_fun)(const T*, uint, T*, uint, uint) = &Kernel<T>::sum_mat_cols;
        };

    private:
    
        Kernel();
};



#include <kernels/kernel.cpp>

#endif
