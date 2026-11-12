#include "algorithms.h"
#include "cuda_backend.h"
#include <omp.h>
#include <algorithm>
#include <stdexcept>

namespace
{
    void matmul_omp_tiled_rows(const Matrix &A, const Matrix &B, Matrix &C,
                               int row_begin, int row_end, int num_threads,
                               int tile_size)
    {
        const int n = A.size();
        const double *a = A.data();
        const double *b = B.data();
        double *c = C.data();
        const int block = std::max(1, tile_size);

#pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = row_begin; i < row_end; ++i)
        {
            for (int k0 = 0; k0 < n; k0 += block)
            {
                const int k_end = std::min(k0 + block, n);
                for (int j0 = 0; j0 < n; j0 += block)
                {
                    const int j_end = std::min(j0 + block, n);
                    for (int k = k0; k < k_end; ++k)
                    {
                        const double a_value = a[i * n + k];
#pragma omp simd
                        for (int j = j0; j < j_end; ++j)
                        {
                            c[i * n + j] += a_value * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
} // namespace

// 1. Thuật toán tuần tự nguyên bản (Sequential Baseline)
void matmul_sequential(const Matrix &A, const Matrix &B, Matrix &C)
{
    int N = A.size();
    C.zero();
    const double *a = A.data();
    const double *b = B.data();
    double *c = C.data();

    for (int i = 0; i < N; ++i)
    {
        for (int k = 0; k < N; ++k)
        {
            double a_ik = a[i * N + k];
#pragma omp simd
            for (int j = 0; j < N; ++j)
            {
                c[i * N + j] += a_ik * b[k * N + j];
            }
        }
    }
}

// 2. Thuật toán song song OpenMP cơ bản (Basic OpenMP Parallel)
void matmul_omp_basic(const Matrix &A, const Matrix &B, Matrix &C, int num_threads)
{
    int N = A.size();
    C.zero();
    const double *a = A.data();
    const double *b = B.data();
    double *c = C.data();

#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < N; ++i)
    {
        for (int k = 0; k < N; ++k)
        {
            double a_ik = a[i * N + k];
#pragma omp simd
            for (int j = 0; j < N; ++j)
            {
                c[i * N + j] += a_ik * b[k * N + j];
            }
        }
    }
}

// 3. Thuật toán song song OpenMP phân cấp L2 + L3 (Tiled L2/L3 Parallel)
// Thiết kế khớp Mục 3.6.2 tài liệu nghiên cứu:
// Siêu khối Tile3 (L3 Cache), khối con Tile2 (L2 Cache), vi khối Tile1 (L1 & SIMD)
void matmul_omp_tiled_l2_l3(const Matrix &A, const Matrix &B, Matrix &C,
                            int num_threads,
                            int bs_l3, int bs_l2, int bs_l1)
{
    int N = A.size();
    C.zero();
    const double *a = A.data();
    const double *b = B.data();
    double *c = C.data();

    // Điều chỉnh kích thước khối cho hợp lý với kích thước N
    if (bs_l3 > N)
        bs_l3 = N;
    if (bs_l2 > bs_l3)
        bs_l2 = bs_l3;
    if (bs_l1 > bs_l2)
        bs_l1 = bs_l2;

// Song song hóa trên các dải siêu khối hàng I
#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int I = 0; I < N; I += bs_l3)
    {
        int I_end = std::min(I + bs_l3, N);

        // Duyệt theo K (Tile3 của A và B nằm gọn trong L3 Cache dùng chung)
        for (int K = 0; K < N; K += bs_l3)
        {
            int K_end = std::min(K + bs_l3, N);

            // Duyệt theo J (Tile3 của B và C)
            for (int J = 0; J < N; J += bs_l3)
            {
                int J_end = std::min(J + bs_l3, N);

                // Khối Tile2 - L2 Cache của từng nhân CPU
                for (int ii = I; ii < I_end; ii += bs_l2)
                {
                    int ii_end = std::min(ii + bs_l2, I_end);

                    for (int kk = K; kk < K_end; kk += bs_l2)
                    {
                        int kk_end = std::min(kk + bs_l2, K_end);

                        for (int jj = J; jj < J_end; jj += bs_l2)
                        {
                            int jj_end = std::min(jj + bs_l2, J_end);

                            // Khối Tile1 - L1 Data Cache & Register micro-kernel
                            for (int iii = ii; iii < ii_end; iii += bs_l1)
                            {
                                int iii_end = std::min(iii + bs_l1, ii_end);

                                for (int kkk = kk; kkk < kk_end; kkk += bs_l1)
                                {
                                    int kkk_end = std::min(kkk + bs_l1, kk_end);

                                    for (int jjj = jj; jjj < jj_end; jjj += bs_l1)
                                    {
                                        int jjj_end = std::min(jjj + bs_l1, jj_end);

                                        // Vòng lặp tính toán vi mô tối ưu thanh ghi CPU & vector hóa AVX2
                                        for (int i = iii; i < iii_end; ++i)
                                        {
                                            for (int k = kkk; k < kkk_end; ++k)
                                            {
                                                double a_val = a[i * N + k];
#pragma omp simd
                                                for (int j = jjj; j < jjj_end; ++j)
                                                {
                                                    c[i * N + j] += a_val * b[k * N + j];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// 4. Thuật toán song song OpenMP L3 + L1 "Bỏ qua L2" (Bypass L2 Tiled Parallel)
// Khớp theo Mục 3.6.3 trong tài liệu:
// Dữ liệu từ L3 được bơm trực tiếp xuống L1 Data Cache của từng nhân,
// loại bỏ chi phí đánh chỉ mục và nạp trung gian qua L2.
void matmul_omp_tiled_l3_l1_bypass(const Matrix &A, const Matrix &B, Matrix &C,
                                   int num_threads,
                                   int bs_l3, int bs_l1)
{
    int N = A.size();
    C.zero();
    const double *a = A.data();
    const double *b = B.data();
    double *c = C.data();

    if (bs_l3 > N)
        bs_l3 = N;
    if (bs_l1 > bs_l3)
        bs_l1 = bs_l3;

#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int I = 0; I < N; I += bs_l3)
    {
        int I_end = std::min(I + bs_l3, N);

        for (int K = 0; K < N; K += bs_l3)
        {
            int K_end = std::min(K + bs_l3, N);

            for (int J = 0; J < N; J += bs_l3)
            {
                int J_end = std::min(J + bs_l3, N);

                // Đi thẳng từ khối L3 xuống các khối L1 nhỏ của từng nhân
                for (int ii = I; ii < I_end; ii += bs_l1)
                {
                    int ii_end = std::min(ii + bs_l1, I_end);

                    for (int kk = K; kk < K_end; kk += bs_l1)
                    {
                        int kk_end = std::min(kk + bs_l1, K_end);

                        for (int jj = J; jj < J_end; jj += bs_l1)
                        {
                            int jj_end = std::min(jj + bs_l1, J_end);

                            for (int i = ii; i < ii_end; ++i)
                            {
                                for (int k = kk; k < kk_end; ++k)
                                {
                                    double a_val = a[i * N + k];
#pragma omp simd
                                    for (int j = jj; j < jj_end; ++j)
                                    {
                                        c[i * N + j] += a_val * b[k * N + j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void matmul_heterogeneous_omp_cuda(const Matrix &A, const Matrix &B, Matrix &C,
                                   int num_threads, double gpu_fraction,
                                   int cuda_tile)
{
    if (!cuda_backend_available())
    {
        throw std::runtime_error(
            "CUDA backend is unavailable. Build with 'make CUDA_ENABLED=1' "
            "and use a CUDA-capable NVIDIA device.");
    }
    if (A.size() != B.size() || A.size() != C.size())
    {
        throw std::invalid_argument("Heterogeneous matmul requires equal matrix sizes");
    }
    if (num_threads <= 0)
    {
        throw std::invalid_argument("CPU thread count must be positive");
    }
    if (A.size() < 2)
    {
        throw std::invalid_argument("Heterogeneous matmul requires N >= 2");
    }
    if (!(gpu_fraction > 0.0 && gpu_fraction < 1.0))
    {
        throw std::invalid_argument("gpu_fraction must be between 0 and 1");
    }

    const int n = A.size();
    const int gpu_rows = std::clamp(static_cast<int>(n * gpu_fraction), 1, n - 1);
    C.zero();

    const int previous_nested = omp_get_nested();
    omp_set_nested(1);
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            matmul_omp_tiled_rows(A, B, C, gpu_rows, n, num_threads, 32);
        }
#pragma omp section
        {
            if (!matmul_cuda_rows(A.data(), B.data(), C.data(), n, gpu_rows,
                                  cuda_tile))
            {
                throw std::runtime_error(
                    "CUDA row multiplication did not complete");
            }
        }
    }
    omp_set_nested(previous_nested);
}

namespace
{
    void validate_cuda_matrices(const Matrix &A, const Matrix &B, const Matrix &C)
    {
        if (A.size() <= 0 || A.size() != B.size() || A.size() != C.size())
        {
            throw std::invalid_argument(
                "CUDA matrix multiplication requires equal positive square matrices");
        }
        if (!cuda_backend_available())
        {
            throw std::runtime_error(
                "CUDA backend is unavailable. Build with 'make CUDA_ENABLED=1' "
                "and use a CUDA-capable NVIDIA device.");
        }
    }
} // namespace

void matmul_cuda_global_memory(const Matrix &A, const Matrix &B, Matrix &C,
                               int block_size)
{
    validate_cuda_matrices(A, B, C);
    if (block_size != 16 && block_size != 32)
    {
        throw std::invalid_argument("CUDA block size must be 16 or 32");
    }
    if (!matmul_cuda_global(A.data(), B.data(), C.data(), A.size(),
                            block_size))
    {
        throw std::runtime_error("CUDA global-memory multiplication did not complete");
    }
}

void matmul_cuda_shared_memory_tiling(const Matrix &A, const Matrix &B,
                                      Matrix &C, int tile_size)
{
    validate_cuda_matrices(A, B, C);
    if (tile_size != 16 && tile_size != 32)
    {
        throw std::invalid_argument("CUDA tile size must be 16 or 32");
    }
    if (!matmul_cuda_shared(A.data(), B.data(), C.data(), A.size(), tile_size))
    {
        throw std::runtime_error("CUDA shared-memory multiplication did not complete");
    }
}

void run_algorithm(AlgorithmType type, const Matrix &A, const Matrix &B, Matrix &C,
                   int num_threads, int bs_l3, int bs_l2, int bs_l1,
                   int cuda_tile)
{
    switch (type)
    {
    case AlgorithmType::SEQUENTIAL_BASELINE:
        matmul_sequential(A, B, C);
        break;
    case AlgorithmType::OMP_BASIC_PARALLEL:
        matmul_omp_basic(A, B, C, num_threads);
        break;
    case AlgorithmType::OMP_TILED_L2_L3:
        matmul_omp_tiled_l2_l3(A, B, C, num_threads, bs_l3, bs_l2, bs_l1);
        break;
    case AlgorithmType::OMP_TILED_L3_L1_BYPASS:
        matmul_omp_tiled_l3_l1_bypass(A, B, C, num_threads, bs_l3, bs_l1);
        break;
    case AlgorithmType::HETEROGENEOUS_OMP_CUDA:
        matmul_heterogeneous_omp_cuda(A, B, C, num_threads, 0.8,
                                      cuda_tile);
        break;
    case AlgorithmType::CUDA_GLOBAL_MEMORY:
        matmul_cuda_global_memory(A, B, C, cuda_tile);
        break;
    case AlgorithmType::CUDA_SHARED_MEMORY_TILING:
        matmul_cuda_shared_memory_tiling(A, B, C, cuda_tile);
        break;
    default:
        throw std::runtime_error("Unknown algorithm type");
    }
}
