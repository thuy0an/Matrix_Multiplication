#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>
#include <cstdint>

// Thông tin phần cứng CPU và bộ nhớ đệm (Cache)
struct HardwareInfo {
    std::string cpu_name = "AMD Ryzen 7 6800H with Radeon Graphics";
    int physical_cores = 8;
    int logical_processors = 16;
    int l1d_cache_kb = 32;       // 32 KB L1 Data per core
    int l2_cache_kb = 512;       // 512 KB L2 per core
    int l3_cache_kb = 16384;     // 16 MB L3 shared
    std::string compiler_info = "GCC 13.2.0 (OpenMP 4.5/5.0, AVX2, FMA)";
};

// Phân loại thuật toán thực hiện
enum class AlgorithmType {
    SEQUENTIAL_BASELINE,    // 1. Tuần tự nguyên bản (O(N^3) Baseline)
    OMP_BASIC_PARALLEL,     // 2. Song song OpenMP cơ bản (Chia hàng không Tiling)
    OMP_TILED_L2_L3,        // 3. Song song OpenMP phân cấp L2 + L3 (Mục 3.6.2)
    OMP_TILED_L3_L1_BYPASS, // 4. Song song OpenMP L3 + L1 bỏ qua L2 (Mục 3.6.3)
    HETEROGENEOUS_OMP_CUDA  // 5. Kết hợp CPU OpenMP và GPU CUDA theo dải hàng
};

// Cấu trúc lưu trữ một dòng kết quả đo đạc thực nghiệm (Benchmark Parameter Record)
struct BenchmarkRecord {
    int id = 0;                             // Mã số lần đo
    int N = 0;                              // Kích thước ma trận vuông (N x N)
    int threads = 1;                        // Số lượng luồng OpenMP (CPU Cores)
    std::string algo_name;                  // Tên thuật toán
    AlgorithmType algo_type;                // Loại thuật toán
    int l3_tile = 0;                        // Kích thước khối L3 (Tile3)
    int l2_tile = 0;                        // Kích thước khối L2 (Tile2)
    int l1_tile = 0;                        // Kích thước khối L1 (Tile1)
    double time_sec = 0.0;                  // Thời gian thực thi (giây)
    double time_ms = 0.0;                   // Thời gian thực thi (mili-giây)
    uint64_t flops = 0;                     // Tổng số phép tính dấu chấm động (2 * N^3)
    double gflops = 0.0;                    // Tốc độ tính toán (GFLOPS = 10^9 FLOP/s)
    double speedup = 1.0;                   // Hệ số tăng tốc (Speedup = T_seq / T_par)
    double efficiency = 100.0;              // Hiệu suất song song (Efficiency = Speedup / Threads * 100%)
    double amdahl_speedup = 1.0;            // Tăng tốc lý thuyết theo Định luật Amdahl
    double estimated_cache_miss_reduction = 1.0; // Ước tính giảm thiểu trượt cache
    double max_error = 0.0;                 // Sai số tuyệt đối lớn nhất so với chuẩn
    bool verified = true;                   // Kết quả kiểm định đúng/sai
    std::string timestamp;                  // Thời điểm ghi nhận
};

#endif // COMMON_H
