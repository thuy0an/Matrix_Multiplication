#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "matrix.h"
#include "common.h"

// Backend CUDA được biên dịch khi dùng: make CUDA_ENABLED=1.
bool cuda_backend_available();

// 1. Thuật toán tuần tự nguyên bản (Sequential Baseline)
// Độ phức tạp O(N^3), dùng làm mốc so sánh hệ số tăng tốc (Speedup)
void matmul_sequential(const Matrix& A, const Matrix& B, Matrix& C);

// 2. Thuật toán song song OpenMP cơ bản (Basic OpenMP Parallel)
// Phân chia theo hàng giữa các luồng nhưng chưa tối ưu Cache Tiling
void matmul_omp_basic(const Matrix& A, const Matrix& B, Matrix& C, int num_threads);

// 3. Thuật toán song song OpenMP phân cấp L2 + L3 (Tiled L2/L3 Parallel)
// - Siêu khối Tile3 kích thước BS_L3 (tối ưu nạp và tái sử dụng trên L3 Shared Cache)
// - Khối Tile2 kích thước BS_L2 (tối ưu L2 Cache của từng nhân)
// - Khối Tile1 kích thước BS_L1 (tối ưu L1 Data Cache và Register SIMD AVX2)
void matmul_omp_tiled_l2_l3(const Matrix& A, const Matrix& B, Matrix& C, 
                           int num_threads, 
                           int bs_l3 = 256, int bs_l2 = 64, int bs_l1 = 32);

// 4. Thuật toán song song OpenMP L3 + L1 "Bỏ qua L2" (Bypass L2 Tiled Parallel)
// - Nạp siêu khối Tile3 trực tiếp vào L3
// - OpenMP phân chia công việc (work-sharing) nạp trực tiếp dữ liệu từ L3 vào L1 Data Cache
void matmul_omp_tiled_l3_l1_bypass(const Matrix& A, const Matrix& B, Matrix& C, 
                                  int num_threads, 
                                  int bs_l3 = 256, int bs_l1 = 64);

// Phân chia theo dải hàng: GPU xử lý gpu_fraction, CPU xử lý phần còn lại.
// Hai phần chạy đồng thời; kết quả không ghi đè vì mỗi thiết bị sở hữu một
// vùng hàng độc lập của C.
void matmul_heterogeneous_omp_cuda(const Matrix& A, const Matrix& B, Matrix& C,
                                   int num_threads, double gpu_fraction = 0.8,
                                   int cuda_tile = 16);

// Hàm điều phối chạy thuật toán theo cấu hình Record
void run_algorithm(AlgorithmType type, const Matrix& A, const Matrix& B, Matrix& C,
                   int num_threads, int bs_l3, int bs_l2, int bs_l1);

#endif // ALGORITHMS_H
