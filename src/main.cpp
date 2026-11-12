#include "common.h"
#include "matrix.h"
#include "algorithms.h"
#include "cuda_backend.h"
#include "excel_export.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <ctime>
#include <omp.h>
#include <algorithm>

// Lấy thời gian hiện tại theo chuỗi
std::string get_current_timestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now_time));
    return std::string(buf);
}

// In tiêu đề trang trí
void print_banner(const HardwareInfo &hw)
{
    std::cout << "\n================================================================================\n";
    std::cout << "                   CHUONG TRINH NHAN MA TRAN CPU + CUDA GPU                      \n";
    std::cout << "                                                                                \n";
    std::cout << "================================================================================\n";
    std::cout << " CPU Hien tai         : " << hw.cpu_name << "\n";
    std::cout << " CUDA GPU             : " << hw.gpu_name;
    if (hw.gpu_compute_major > 0)
    {
        std::cout << " (compute " << hw.gpu_compute_major << "."
                  << hw.gpu_compute_minor << ")";
    }
    std::cout << "\n";
    std::cout << " So nhan vat ly       : " << hw.physical_cores << " Cores\n";
    std::cout << " So luong logic       : " << hw.logical_processors << " Threads (OpenMP Max: " << omp_get_max_threads() << ")\n";
    std::cout << " Bo nho dem L1 Data   : " << hw.l1d_cache_kb << " KB / core\n";
    std::cout << " Bo nho dem L2        : " << hw.l2_cache_kb << " KB / core (" << (hw.l2_cache_kb * hw.physical_cores / 1024) << " MB tong cong)\n";
    std::cout << " Bo nho dem L3        : " << (hw.l3_cache_kb / 1024) << " MB (Chia se dung chung / Shared Cache)\n";
    std::cout << "================================================================================\n\n";
}

// In bang ket qua console
void print_records_table(const std::vector<BenchmarkRecord> &records)
{
    if (records.empty())
    {
        std::cout << "[Thong bao] Chua co du lieu thuc nghiem nao.\n";
        return;
    }
    std::cout << "\n"
              << std::string(115, '-') << "\n";
    std::cout << std::left
              << std::setw(5) << "STT"
              << std::setw(8) << "N"
              << std::setw(8) << "Luong"
              << std::setw(28) << "Thuat toan"
              << std::setw(14) << "Thoi gian(s)"
              << std::setw(12) << "GFLOPS"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Hieu suat"
              << std::setw(12) << "Kiem dinh"
              << "\n";
    std::cout << std::string(115, '-') << "\n";

    for (const auto &r : records)
    {
        std::cout << std::left
                  << std::setw(5) << r.id
                  << std::setw(8) << r.N
                  << std::setw(8) << r.threads
                  << std::setw(28) << (r.algo_name.length() > 27 ? r.algo_name.substr(0, 24) + "..." : r.algo_name)
                  << std::fixed << std::setprecision(5) << std::setw(14) << r.time_sec
                  << std::fixed << std::setprecision(2) << std::setw(12) << r.gflops
                  << std::fixed << std::setprecision(2) << std::setw(12) << (std::to_string(r.speedup).substr(0, 4) + "x")
                  << std::fixed << std::setprecision(1) << std::setw(12) << (std::to_string(r.efficiency).substr(0, 4) + "%")
                  << std::setw(12) << (r.verified ? "DAT (OK)" : "LOI")
                  << "\n";
    }
    std::cout << std::string(115, '-') << "\n\n";
}

// Chạy một thử nghiệm đơn lẻ và trả về BenchmarkRecord (hỗ trợ num_runs lần lặp lấy trung bình)
BenchmarkRecord run_single_test(int test_id, int N, int threads,
                                AlgorithmType type, const std::string &name,
                                int bs_l3, int bs_l2, int bs_l1,
                                const Matrix &A, const Matrix &B, Matrix &C,
                                const Matrix *ref_C = nullptr, double t_seq = -1.0,
                                int num_runs = 1, int cuda_tile = 16)
{
    BenchmarkRecord rec;
    rec.id = test_id;
    rec.N = N;
    rec.threads = threads;
    rec.algo_name = name;
    rec.algo_type = type;
    rec.l3_tile = (type == AlgorithmType::OMP_TILED_L2_L3 || type == AlgorithmType::OMP_TILED_L3_L1_BYPASS) ? bs_l3 : 0;
    rec.l2_tile = (type == AlgorithmType::OMP_TILED_L2_L3) ? bs_l2 : 0;
    rec.l1_tile = (type == AlgorithmType::OMP_TILED_L2_L3 || type == AlgorithmType::OMP_TILED_L3_L1_BYPASS) ? bs_l1 : 0;
    if (type == AlgorithmType::HETEROGENEOUS_OMP_CUDA ||
        type == AlgorithmType::CUDA_GLOBAL_MEMORY ||
        type == AlgorithmType::CUDA_SHARED_MEMORY_TILING)
    {
        rec.cuda_tile = cuda_tile;
    }
    rec.flops = 2ULL * N * N * N;
    rec.timestamp = get_current_timestamp();

    // Đo thời gian chính xác cao qua num_runs lần lặp lấy trung bình (Loại trừ nhiễu OS)
    int actual_runs = std::max(1, num_runs);
    if (type == AlgorithmType::HETEROGENEOUS_OMP_CUDA ||
        type == AlgorithmType::CUDA_GLOBAL_MEMORY ||
        type == AlgorithmType::CUDA_SHARED_MEMORY_TILING)
    {
        run_algorithm(type, A, B, C, threads, bs_l3, bs_l2, bs_l1, cuda_tile);
    }
    double total_time = 0.0;
    for (int r = 0; r < actual_runs; ++r)
    {
        double t_start = omp_get_wtime();
        run_algorithm(type, A, B, C, threads, bs_l3, bs_l2, bs_l1, cuda_tile);
        double t_end = omp_get_wtime();
        total_time += (t_end - t_start);
    }

    rec.time_sec = total_time / actual_runs;
    rec.time_ms = rec.time_sec * 1000.0;
    rec.gflops = (rec.time_sec > 0) ? (static_cast<double>(rec.flops) / (rec.time_sec * 1e9)) : 0.0;

    // Kiểm định kết quả bằng Sai số tương đối theo chuẩn Frobenius: ||C - ref||_F / ||ref||_F < 1e-5
    if (ref_C != nullptr && ref_C->data() != nullptr)
    {
        rec.max_error = C.relative_error(*ref_C);
        rec.verified = (rec.max_error <= 1e-5);
    }
    else
    {
        rec.max_error = 0.0;
        rec.verified = true;
    }

    // Tốc độ tăng tốc & Hiệu suất song song
    if (t_seq > 0)
    {
        rec.speedup = t_seq / rec.time_sec;
    }
    else
    {
        rec.speedup = 1.0;
    }
    rec.efficiency = (rec.speedup / threads) * 100.0;

    // Amdahl lý thuyết với p = 0.98 (98% mã song song)
    double p = 0.98;
    rec.amdahl_speedup = 1.0 / ((1.0 - p) + (p / threads));

    // Hệ số ước tính giảm tải truy xuất RAM/Cache Miss
    if (rec.l3_tile > 0)
    {
        rec.estimated_cache_miss_reduction = static_cast<double>(N) / rec.l3_tile;
    }
    else
    {
        rec.estimated_cache_miss_reduction = 1.0;
    }

    return rec;
}

void run_cuda_benchmarks(std::vector<BenchmarkRecord> &records, int &test_id,
                         int N, int cuda_tile, const Matrix &A, const Matrix &B,
                         Matrix &C_test, const Matrix &C_ref, double t_seq,
                         int num_runs)
{
    if (!cuda_backend_available())
    {
        return;
    }

    const AlgorithmType cuda_algorithms[] = {
        AlgorithmType::CUDA_GLOBAL_MEMORY,
        AlgorithmType::CUDA_SHARED_MEMORY_TILING};
    const char *names[] = {
        "6. CUDA Global Memory",
        "7. CUDA Shared Memory Tiling"};

    for (int i = 0; i < 2; ++i)
    {
        std::cout << "       + " << names[i] << ": " << std::flush;
        BenchmarkRecord record = run_single_test(
            test_id++, N, 1, cuda_algorithms[i], names[i], 0, 0, 0, A, B,
            C_test, &C_ref, t_seq, num_runs, cuda_tile);
        records.push_back(record);
        std::cout << std::fixed << std::setprecision(4) << record.time_sec
                  << "s | " << std::fixed << std::setprecision(1)
                  << record.gflops << " GFLOPS | Speedup: " << std::fixed
                  << std::setprecision(2) << record.speedup << "x (Err: "
                  << std::scientific << std::setprecision(1)
                  << record.max_error << ")\n";
    }
}

// Chạy bộ thực nghiệm chuẩn (hỗ trợ tùy biến N, Threads và số lần lặp num_runs)
void run_standard_benchmark_suite(std::vector<BenchmarkRecord> &records, const HardwareInfo &hw,
                                  int num_runs = 1,
                                  std::vector<int> N_list = {512, 1024, 2048},
                                  std::vector<int> thread_list = {1, 2, 4, 8, 16},
                                  int cuda_tile = 16)
{
    std::cout << "\n>>> BAT DAU CHAY BO THUC NGHIEM (So lan lap: " << num_runs << " lan lay trung binh)...\n";
    std::cout << "Kich thuoc ma tran N: ";
    for (size_t i = 0; i < N_list.size(); ++i)
        std::cout << N_list[i] << (i + 1 < N_list.size() ? ", " : "");
    std::cout << "\nCac muc luong: ";
    for (size_t i = 0; i < thread_list.size(); ++i)
        std::cout << thread_list[i] << (i + 1 < thread_list.size() ? ", " : "");
    std::cout << " (Toi da theo phan cung: " << hw.logical_processors << ")\n";
    std::cout << "CUDA GPU             : "
              << (cuda_backend_available() ? "San sang" : "Khong san sang")
              << "\n\n";

    // Giữ số luồng không vượt quá khả năng phần cứng hoặc tối đa 16
    int max_t = hw.logical_processors;
    if (std::find(thread_list.begin(), thread_list.end(), max_t) == thread_list.end() && thread_list.back() < max_t)
    {
        thread_list.push_back(max_t);
        std::sort(thread_list.begin(), thread_list.end());
    }

    int test_id = static_cast<int>(records.size()) + 1;

    for (int N : N_list)
    {
        std::cout << "====================================================================\n";
        std::cout << " >> DANG THUC HIEN MA TRAN KICH THUOC N = " << N << " x " << N << " (Bo nho: "
                  << std::fixed << std::setprecision(1) << (3.0 * N * N * sizeof(double) / (1024 * 1024)) << " MB)\n";
        std::cout << "====================================================================\n";

        Matrix A(N), B(N), C_ref(N), C_test(N);
        A.fill_random(0.5, 3.0, 101);
        B.fill_random(0.5, 3.0, 202);

        // 1. Chạy đo mốc tuần tự Baseline
        std::cout << " [1/7] Chay Tuan tu Baseline (1 Core)... " << std::flush;
        BenchmarkRecord seq_rec = run_single_test(test_id++, N, 1,
                                                  AlgorithmType::SEQUENTIAL_BASELINE,
                                                  "1. Tuan tu (Baseline)",
                                                  0, 0, 0, A, B, C_ref, nullptr, -1.0, num_runs);
        double t_seq = seq_rec.time_sec;
        records.push_back(seq_rec);
        std::cout << "Xong (" << std::fixed << std::setprecision(3) << seq_rec.time_sec
                  << "s, " << std::setprecision(1) << seq_rec.gflops << " GFLOPS)\n";

        // Khối Tile tối ưu theo bộ nhớ đệm
        int bs_l3 = (N >= 1024) ? 256 : 128;
        int bs_l2 = 64;
        int bs_l1 = 32;

        for (int q : thread_list)
        {
            std::cout << "   --- So luong: " << q << " Cores/Threads ---\n";

            // 2. Song song OpenMP cơ bản
            std::cout << "       + OpenMP Co ban       : " << std::flush;
            BenchmarkRecord r_basic = run_single_test(test_id++, N, q,
                                                      AlgorithmType::OMP_BASIC_PARALLEL,
                                                      "2. Song song OpenMP Co ban",
                                                      0, 0, 0, A, B, C_test, &C_ref, t_seq, num_runs);
            records.push_back(r_basic);
            std::cout << std::fixed << std::setprecision(4) << r_basic.time_sec << "s | "
                      << std::fixed << std::setprecision(1) << r_basic.gflops << " GFLOPS | Speedup: "
                      << std::fixed << std::setprecision(2) << r_basic.speedup << "x (Err: "
                      << std::scientific << std::setprecision(1) << r_basic.max_error << ")\n";

            // 3. Song song OpenMP Tiled L2 + L3 (Mục 3.6.2)
            std::cout << "       + OpenMP Tiled L2+L3  : " << std::flush;
            BenchmarkRecord r_l2_l3 = run_single_test(test_id++, N, q,
                                                      AlgorithmType::OMP_TILED_L2_L3,
                                                      "3. Song song Tiled L2+L3",
                                                      bs_l3, bs_l2, bs_l1, A, B, C_test, &C_ref, t_seq, num_runs);
            records.push_back(r_l2_l3);
            std::cout << std::fixed << std::setprecision(4) << r_l2_l3.time_sec << "s | "
                      << std::fixed << std::setprecision(1) << r_l2_l3.gflops << " GFLOPS | Speedup: "
                      << std::fixed << std::setprecision(2) << r_l2_l3.speedup << "x (Err: "
                      << std::scientific << std::setprecision(1) << r_l2_l3.max_error << ")\n";

            // 4. Song song OpenMP Tiled L3 + L1 Bypass L2 (Mục 3.6.3)
            std::cout << "       + OpenMP Bypass L2    : " << std::flush;
            BenchmarkRecord r_bypass = run_single_test(test_id++, N, q,
                                                       AlgorithmType::OMP_TILED_L3_L1_BYPASS,
                                                       "4. Song song Bypass L2",
                                                       bs_l3, 0, 64, A, B, C_test, &C_ref, t_seq, num_runs);
            records.push_back(r_bypass);
            std::cout << std::fixed << std::setprecision(4) << r_bypass.time_sec << "s | "
                      << std::fixed << std::setprecision(1) << r_bypass.gflops << " GFLOPS | Speedup: "
                      << std::fixed << std::setprecision(2) << r_bypass.speedup << "x (Err: "
                      << std::scientific << std::setprecision(1) << r_bypass.max_error << ")\n";

            if (cuda_backend_available())
            {
                std::cout << "       + CPU + CUDA GPU      : " << std::flush;
                BenchmarkRecord r_heterogeneous = run_single_test(
                    test_id++, N, q, AlgorithmType::HETEROGENEOUS_OMP_CUDA,
                    "5. Ket hop CPU + CUDA GPU", 0, 0, 16, A, B, C_test,
                    &C_ref, t_seq, num_runs, cuda_tile);
                records.push_back(r_heterogeneous);
                std::cout << std::fixed << std::setprecision(4)
                          << r_heterogeneous.time_sec << "s | "
                          << std::fixed << std::setprecision(1)
                          << r_heterogeneous.gflops << " GFLOPS | Speedup: "
                          << std::fixed << std::setprecision(2)
                          << r_heterogeneous.speedup << "x (Err: "
                          << std::scientific << std::setprecision(1)
                          << r_heterogeneous.max_error << ")\n";
            }
        }
        run_cuda_benchmarks(records, test_id, N, cuda_tile, A, B, C_test,
                            C_ref, t_seq, num_runs);
        std::cout << "\n";
    }

    std::cout << ">>> HOAN THANH BO THUC NGHIEM!\n";
}

// Chạy tùy chỉnh theo người dùng nhập
void run_custom_interactive(std::vector<BenchmarkRecord> &records,
                            const HardwareInfo &hw, int cuda_tile)
{
    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "             CHE DO THUC NGHIEM TUY CHINH                  \n";
    std::cout << "------------------------------------------------------------\n";

    int N = 1024;
    std::cout << "Nhap kich thuoc ma tran N (vi du: 512, 1024, 2048, hoac so bat ky): ";
    if (!(std::cin >> N) || N <= 0)
    {
        std::cin.clear();
        std::string dummy;
        std::getline(std::cin, dummy);
        std::cout << "[Canh bao] Gia tri N khong hop le, dat mac dinh N = 1024.\n";
        N = 1024;
    }

    int threads = hw.physical_cores;
    std::cout << "Nhap so luong nhan/luong CPU (1 den " << hw.logical_processors << ", mac dinh: " << hw.physical_cores << "): ";
    if (!(std::cin >> threads) || threads <= 0)
    {
        std::cin.clear();
        std::string dummy;
        std::getline(std::cin, dummy);
        std::cout << "[Canh bao] So luong khong hop le, dat mac dinh: " << hw.physical_cores << ".\n";
        threads = hw.physical_cores;
    }

    std::cout << "\nChon phuong thuc chay:\n";
    std::cout << "  1. So sanh cac thuat toan CPU va GPU CUDA\n";
    std::cout << "  2. Chi chay Song song Tiled L2 + L3\n";
    std::cout << "  3. Chi chay Song song Tiled L3 + L1 Bypass L2\n";
    std::cout << "  4. Chi chay ket hop CPU OpenMP + CUDA GPU\n";
    std::cout << "  5. So sanh hai kernel CUDA (Global Memory / Shared Tiling)\n";
    std::cout << "Lua chon cua ban [1-5]: ";
    int choice = 1;
    std::cin >> choice;

    int test_id = static_cast<int>(records.size()) + 1;
    Matrix A(N), B(N), C_ref(N), C_test(N);
    A.fill_random(1.0, 5.0, 1234);
    B.fill_random(1.0, 5.0, 5678);

    int bs_l3 = (N >= 1024) ? 256 : 128;
    int bs_l2 = 64;
    int bs_l1 = 32;

    std::cout << "\nDang khoi chay thuc nghiem voi N = " << N << ", Luong = " << threads << "...\n";

    if (choice == 1)
    {
        std::cout << "  [1/7] Chay Tuan tu Baseline... " << std::flush;
        BenchmarkRecord r1 = run_single_test(test_id++, N, 1, AlgorithmType::SEQUENTIAL_BASELINE,
                                             "1. Tuan tu (Baseline)", 0, 0, 0, A, B, C_ref, nullptr, -1.0);
        double t_seq = r1.time_sec;
        records.push_back(r1);
        std::cout << r1.time_sec << "s (" << r1.gflops << " GFLOPS)\n";

        std::cout << "  [2/7] Chay Song song OpenMP Co ban... " << std::flush;
        BenchmarkRecord r2 = run_single_test(test_id++, N, threads, AlgorithmType::OMP_BASIC_PARALLEL,
                                             "2. Song song OpenMP Co ban", 0, 0, 0, A, B, C_test, &C_ref, t_seq);
        records.push_back(r2);
        std::cout << r2.time_sec << "s (" << r2.gflops << " GFLOPS, Speedup: " << r2.speedup << "x)\n";

        std::cout << "  [3/7] Chay Song song Tiled L2+L3 ... " << std::flush;
        BenchmarkRecord r3 = run_single_test(test_id++, N, threads, AlgorithmType::OMP_TILED_L2_L3,
                                             "3. Song song Tiled L2+L3", bs_l3, bs_l2, bs_l1, A, B, C_test, &C_ref, t_seq);
        records.push_back(r3);
        std::cout << r3.time_sec << "s (" << r3.gflops << " GFLOPS, Speedup: " << r3.speedup << "x)\n";

        std::cout << "  [4/7] Chay Song song Tiled L3+L1 Bypass L2 ... " << std::flush;
        BenchmarkRecord r4 = run_single_test(test_id++, N, threads, AlgorithmType::OMP_TILED_L3_L1_BYPASS,
                                             "4. Song song Bypass L2", bs_l3, 0, 64, A, B, C_test, &C_ref, t_seq);
        records.push_back(r4);
        std::cout << r4.time_sec << "s (" << r4.gflops << " GFLOPS, Speedup: " << r4.speedup << "x)\n";

        if (cuda_backend_available())
        {
            std::cout << "  [5/7] Chay ket hop CPU OpenMP + CUDA GPU ... " << std::flush;
            BenchmarkRecord r5 = run_single_test(
                test_id++, N, threads, AlgorithmType::HETEROGENEOUS_OMP_CUDA,
                "5. Ket hop CPU + CUDA GPU", 0, 0, 16, A, B, C_test, &C_ref,
                t_seq, 1, cuda_tile);
            records.push_back(r5);
            std::cout << r5.time_sec << "s (" << r5.gflops
                      << " GFLOPS, Speedup: " << r5.speedup << "x)\n";
        }
        else
        {
            std::cout << "  [Bo qua] Khong co CUDA GPU hoac chuong trinh chua "
                         "duoc build voi CUDA_ENABLED=1.\n";
        }
        run_cuda_benchmarks(records, test_id, N, cuda_tile, A, B, C_test,
                            C_ref, t_seq, 1);
    }
    else if (choice == 2)
    {
        std::cout << "  Chay Song song Tiled L2+L3 ... " << std::flush;
        BenchmarkRecord r = run_single_test(test_id++, N, threads, AlgorithmType::OMP_TILED_L2_L3,
                                            "3. Song song Tiled L2+L3", bs_l3, bs_l2, bs_l1, A, B, C_test, nullptr, -1.0);
        records.push_back(r);
        std::cout << r.time_sec << "s (" << r.gflops << " GFLOPS)\n";
    }
    else if (choice == 3)
    {
        std::cout << "  Chay Song song Tiled L3+L1 Bypass L2 ... " << std::flush;
        BenchmarkRecord r = run_single_test(test_id++, N, threads, AlgorithmType::OMP_TILED_L3_L1_BYPASS,
                                            "4. Song song Bypass L2", bs_l3, 0, 64, A, B, C_test, nullptr, -1.0);
        records.push_back(r);
        std::cout << r.time_sec << "s (" << r.gflops << " GFLOPS)\n";
    }
    else if (choice == 4)
    {
        if (!cuda_backend_available())
        {
            std::cout << "  [Loi] CUDA GPU khong san sang. Hay build bang "
                         "'make CUDA_ENABLED=1' tren may co GPU NVIDIA.\n";
        }
        else
        {
            std::cout << "  Chay ket hop CPU OpenMP + CUDA GPU ... " << std::flush;
            BenchmarkRecord r = run_single_test(
                test_id++, N, threads, AlgorithmType::HETEROGENEOUS_OMP_CUDA,
                "5. Ket hop CPU + CUDA GPU", 0, 0, 16, A, B, C_test, nullptr,
                -1.0, 1, cuda_tile);
            records.push_back(r);
            std::cout << r.time_sec << "s (" << r.gflops << " GFLOPS)\n";
        }
    }
    else if (choice == 5)
    {
        if (!cuda_backend_available())
        {
            std::cout << "  [Loi] CUDA GPU khong san sang. Hay build bang "
                         "'make CUDA_ENABLED=1' tren may co GPU NVIDIA.\n";
        }
        else
        {
            const double reference_start = omp_get_wtime();
            matmul_sequential(A, B, C_ref);
            const double t_seq = omp_get_wtime() - reference_start;
            run_cuda_benchmarks(records, test_id, N, cuda_tile, A, B, C_test,
                                C_ref, t_seq, 1);
        }
    }
    else
    {
        std::cout << "  [Loi] Lua chon khong hop le.\n";
    }

    std::cout << ">>> Thuc nghiem tuy chinh hoan tat!\n";
}

// Khảo sát độ nhạy kích thước khối Cache (Cache Block Tuning)
void run_cache_tuning_experiment(std::vector<BenchmarkRecord> &records, const HardwareInfo &hw)
{
    int N = 1024;
    int threads = hw.physical_cores;
    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "    THUC NGHIEM KHAO SAT ANH HUONG KICH THUOC KHOI CACHE     \n";
    std::cout << "    Ma tran N = 1024, So luong = " << threads << " Cores                   \n";
    std::cout << "------------------------------------------------------------\n";

    Matrix A(N), B(N), C_ref(N), C_test(N);
    A.fill_random(1.0, 3.0, 999);
    B.fill_random(1.0, 3.0, 888);

    std::cout << "Chay chuan Reference... " << std::flush;
    matmul_sequential(A, B, C_ref);
    std::cout << "Xong.\n\n";

    struct Config
    {
        int l3;
        int l2;
        int l1;
        std::string desc;
    };
    std::vector<Config> configs = {
        {128, 64, 32, "Khoi nho (Tile3=128, Tile2=64, Tile1=32)"},
        {256, 64, 32, "Khoi chuan L2/L3 (Tile3=256, Tile2=64, Tile1=32)"},
        {256, 128, 64, "Khoi L2 lon (Tile3=256, Tile2=128, Tile1=64)"},
        {512, 128, 64, "Khoi sieu lon (Tile3=512, Tile2=128, Tile1=64)"}};

    int test_id = static_cast<int>(records.size()) + 1;
    for (const auto &cfg : configs)
    {
        std::cout << "  Khao sat: " << cfg.desc << "... " << std::flush;
        BenchmarkRecord r = run_single_test(test_id++, N, threads, AlgorithmType::OMP_TILED_L2_L3,
                                            "Tuning: " + cfg.desc, cfg.l3, cfg.l2, cfg.l1,
                                            A, B, C_test, &C_ref, -1.0);
        records.push_back(r);
        std::cout << r.time_sec << "s (" << r.gflops << " GFLOPS)\n";
    }
    std::cout << ">>> Hoan thanh khao sat do nhay khoi Cache!\n";
}

// Lưu file Excel và CSV
void export_all_files(const std::vector<BenchmarkRecord> &records, const HardwareInfo &hw,
                      const std::string &xls_name = "benchmark_results.xls",
                      const std::string &csv_name = "benchmark_results.csv")
{
    if (records.empty())
    {
        std::cout << "[Luu y] Chua co du lieu nao de xuat ra file Excel.\n";
        return;
    }
    std::cout << "\nDang xuat ket qua thuc nghiem ra file...\n";
    bool ok_xls = ExcelExporter::exportToExcel(xls_name, records, hw);
    bool ok_csv = ExcelExporter::exportToCSV(csv_name, records, hw);

    if (ok_xls)
    {
        std::cout << "  [+] Da xuat file Microsoft Excel thanh cong : " << xls_name << "\n";
        std::cout << "      (File .xls dinh dang XML Spreadsheet 2003 co mau sac, bang bieuthong ke, cong thuc)\n";
    }
    if (ok_csv)
    {
        std::cout << "  [+] Da xuat file CSV thuc nghiem thanh cong: " << csv_name << "\n";
        std::cout << "      (Phuc vu phan tich do thi Python, R, SPSS, MATLAB)\n";
    }
    std::cout << "\n";
}

int main(int argc, char *argv[])
{
    HardwareInfo hw;
    if (cuda_backend_available())
    {
        cuda_backend_device_info(hw.gpu_name, hw.gpu_compute_major,
                                 hw.gpu_compute_minor);
    }
    std::vector<BenchmarkRecord> records;

    // Phân tích tham số dòng lệnh chuyên nghiệp (CLI Flags)
    bool auto_mode = false;
    int runs = 1;
    int cuda_tile = 16;
    std::vector<int> custom_N_list;
    std::vector<int> custom_thread_list;
    std::string custom_csv = "benchmark_results.csv";
    std::string custom_xls = "benchmark_results.xls";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--auto" || arg == "-auto" || arg == "--benchmark" || arg == "-b")
        {
            auto_mode = true;
        }
        else if ((arg == "--runs" || arg == "-r") && i + 1 < argc)
        {
            runs = std::max(1, std::stoi(argv[++i]));
        }
        else if ((arg == "--size" || arg == "-N") && i + 1 < argc)
        {
            std::string s = argv[++i];
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ','))
            {
                if (!item.empty())
                    custom_N_list.push_back(std::stoi(item));
            }
        }
        else if ((arg == "--threads" || arg == "-t") && i + 1 < argc)
        {
            std::string s = argv[++i];
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ','))
            {
                if (!item.empty())
                    custom_thread_list.push_back(std::stoi(item));
            }
        }
        else if (arg == "--cuda-tile" && i + 1 < argc)
        {
            cuda_tile = std::stoi(argv[++i]);
        }
        else if ((arg == "--csv" || arg == "-o") && i + 1 < argc)
        {
            custom_csv = argv[++i];
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: matmul_benchmark [OPTIONS]\n"
                      << "  --auto, -b               Chay bo benchmark tu dong\n"
                      << "  --runs <K>, -r <K>       So lan chay lap lai de tinh trung binh (VD: -r 5)\n"
                      << "  --size <N>, -N <N>       Kich thuoc ma tran (VD: -N 512,1024,2048 hoac -N 4096)\n"
                      << "  --threads <T>, -t <T>    Danh sach so luong (VD: -t 1,2,4,8,16 hoac -t 8)\n"
                      << "  --cuda-tile <16|32>      Block CUDA / kich thuoc tile shared memory (mac dinh: 16)\n"
                      << "  --csv <file>, -o <file>  Ten file CSV xuat ket qua (VD: -o results.csv)\n"
                      << "  --help, -h               Hien thi huong dan nay\n";
            return 0;
        }
    }

    if (cuda_tile != 16 && cuda_tile != 32)
    {
        std::cerr << "[Loi] --cuda-tile chi nhan gia tri 16 hoac 32.\n";
        return 2;
    }

    if (auto_mode || !custom_N_list.empty())
    {
        print_banner(hw);
        std::vector<int> N_list = custom_N_list.empty() ? std::vector<int>{512, 1024, 2048} : custom_N_list;
        std::vector<int> thread_list = custom_thread_list.empty() ? std::vector<int>{1, 2, 4, 8, 16} : custom_thread_list;
        std::cout << "[Che do tu dong] Chay day du bo thuc nghiem (Lap " << runs << " lan lay trung binh)...\n";
        run_standard_benchmark_suite(records, hw, runs, N_list, thread_list,
                                     cuda_tile);
        print_records_table(records);
        export_all_files(records, hw, custom_xls, custom_csv);
        return 0;
    }

    // Menu tuong tac nguoi dung
    int choice = -1;
    while (choice != 0)
    {
        print_banner(hw);
        std::cout << "              MENU THUC NGHIEM CPU + CUDA GPU                  \n";
        std::cout << " [1] Chay bo benchmark CPU va CUDA (Global / Shared Memory)\n";
        std::cout << "     -> Do day du tham so & Tu dong xuat file Excel (.xls) + CSV\n";
        std::cout << " [2] Thuc nghiem tuy chinh (Nhap N bat ky, tuy chinh so Luong CPU theo y muon)\n";
        std::cout << " [3] Khao sat do nhay kich thuoc khoi Cache (Cache Block Size Tuning)\n";
        std::cout << " [4] Xuat ket qua hien tai ra file Excel (.xls) va CSV (.csv)\n";
        std::cout << " [5] Xem lai bang tong hop ket qua do dac da chay\n";
        std::cout << " [0] Thoat chuong trinh\n";
        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << "Nhap lua chon cua ban [0-5]: ";

        if (!(std::cin >> choice))
        {
            if (std::cin.eof())
            {
                std::cout << "\n[Phat hien moi truong khong tuong tac / Profiler] Tu dong chay Benchmark chuan...\n";
                run_standard_benchmark_suite(records, hw, 1, {512, 1024, 2048},
                                             {1, 2, 4, 8, 16}, cuda_tile);
                print_records_table(records);
                export_all_files(records, hw);
                return 0;
            }
            std::cin.clear();
            std::string dummy;
            std::getline(std::cin, dummy);
            continue;
        }

        switch (choice)
        {
        case 1:
            run_standard_benchmark_suite(records, hw, 1, {512, 1024, 2048},
                                         {1, 2, 4, 8, 16}, cuda_tile);
            print_records_table(records);
            export_all_files(records, hw);
            break;
        case 2:
            run_custom_interactive(records, hw, cuda_tile);
            print_records_table(records);
            export_all_files(records, hw);
            break;
        case 3:
            run_cache_tuning_experiment(records, hw);
            print_records_table(records);
            export_all_files(records, hw);
            break;
        case 4:
            export_all_files(records, hw);
            break;
        case 5:
            print_records_table(records);
            break;
        case 0:
            std::cout << "\nTam biet! Cam on ban da su dung chuong trinh.\n";
            break;
        default:
            std::cout << "\nLua chon khong hop le, vui long chon lai!\n";
            break;
        }

        if (choice != 0)
        {
            std::cout << "\nNhan Enter de tiep tuc...";
            std::cin.ignore(10000, '\n');
            std::cin.get();
        }
    }

    return 0;
}
