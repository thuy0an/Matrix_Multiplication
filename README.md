# DỰ ÁN: NHÂN MA TRẬN SONG SONG CPU + CUDA GPU

## HIGH-PERFORMANCE MATRIX MULTIPLICATION (C++17 / OpenMP / AVX2 / CUDA)

> **Nền tảng phần cứng thử nghiệm**: AMD Ryzen 7 6800H và NVIDIA GeForce RTX 3050 Ti. Backend CUDA mặc định nhắm tới NVIDIA T4 (compute capability 7.5), phù hợp chạy trên Google Colab.

---

## 1. HƯỚNG DẪN DÀNH CHO NGƯỜI DÙNG (USER GUIDE)

### 1.1. Chạy trên Windows (1-Click)

* Click đúp vào file **[run.bat](./run.bat)** (hoặc mở Terminal gõ `.\run.bat`):
  ```powershell
  .\run.bat
  ```

  *File này tự động phát hiện `nvcc`, biên dịch backend CUDA nếu có, sau đó mở menu điều khiển.*

### 1.2. Chạy trên Linux / WSL / Git Bash

* Mở terminal và gõ:
  ```bash
  ./run_benchmarks.sh
  ```

  *Tự động biên dịch qua Makefile, quét các kích thước ma trận được cấu hình trong script, lặp 5 lần lấy trung bình và xuất file kết quả.*
  Menu tương tác cung cấp 5 chức năng:
* `[1]`: Chạy bộ thực nghiệm chuẩn CPU/CUDA ($N=512, 1024, 2048$) và xuất file Excel.
* `[2]`: Thực nghiệm tùy chỉnh kích thước $N$ và số luồng CPU bất kỳ (hỗ trợ so sánh CPU và CPU + CUDA GPU).
* `[3]`: Khảo sát độ nhạy kích thước khối Cache (Cache Block Tuning).
* `[4]`: Xuất kết quả hiện tại ra Excel & CSV.
* `[5]`: Xem lại bảng tổng hợp kết quả đã đo.
* `[0]`: Thoát chương trình.

### 1.3. Chạy qua Tham số Dòng lệnh (CLI Flags)

```powershell
# Chạy bộ benchmark tự động mặc định:
.\matmul_benchmark.exe --auto

# Chạy kích thước N=2048 trên 8 luồng, lặp 5 lần lấy trung bình và xuất ra results.csv:
.\matmul_benchmark.exe -N 2048 -t 8 -r 5 -o results.csv

# Quét danh sách kích thước tùy chọn:
.\matmul_benchmark.exe -N 512,1024,2048,4096 -t 1,2,4,8,16 -r 5 -o results.csv

# Xem trợ giúp:
.\matmul_benchmark.exe --help
```

---

## 2. HƯỚNG DẪN DÀNH CHO LẬP TRÌNH VIÊN (DEVELOPER GUIDE)

### 2.1. Cấu trúc Thư mục Mã nguồn

```
Matrix_Multiplication/
├── include/
│   ├── common.h          # Định nghĩa cấu trúc HardwareInfo, BenchmarkRecord, Enum AlgorithmType
│   ├── matrix.h          # Lớp Matrix: Cấp phát căn chỉnh 64-byte, kiểm định Frobenius Norm
│   ├── algorithms.h      # Khai báo thuật toán CPU, GPU CUDA và CPU + CUDA
│   └── excel_export.h    # Module xuất file Excel (.xls XML Spreadsheet) và CSV
├── src/
│   ├── matrix.cpp        # Quản lý bộ nhớ Rule of 5, tạo số ngẫu nhiên mt19937, sai số tương đối
│   ├── algorithms.cpp    # Điều phối thuật toán CPU/GPU và xác thực tham số
│   ├── cuda_backend.cu   # Kernel CUDA Global Memory, Shared Memory Tiling và hybrid
│   ├── cuda_backend.h    # Giao diện backend CUDA
│   ├── cuda_backend_stub.cpp # Stub để build CPU-only
│   ├── excel_export.cpp  # Engine sinh bảng tính Microsoft Excel có màu sắc và công thức
│   └── main.cpp          # CLI Parser, điều phối Benchmark K-lần lấy trung bình, Menu tương tác
├── Makefile              # Makefile chuẩn đa nền tảng (Linux, macOS, Windows)
├── run_benchmarks.sh     # Automation Bash Script cho Linux/WSL/Git Bash
└── .gitignore             # Loại trừ file nhị phân và kết quả tạm
```

### 2.2. Biên dịch bằng Makefile

```bash
# Biên dịch toàn bộ dự án
make

# Biên dịch và chạy bộ Benchmark tự động
make benchmark

# Dọn dẹp object files và binary
make clean
```

**Tập cờ tối ưu GCC**: `-std=c++17 -O3 -fopenmp -mavx2 -mfma -Wall -Wextra`.

### 2.3. Hai thuật toán nhân ma trận CUDA

Hai thuật toán GPU-only hoạt động trên ma trận vuông `double` và được thêm vào
benchmark chuẩn khi phát hiện CUDA:

1. **CUDA Global Memory** (`CUDA_GLOBAL_MEMORY`): mỗi thread tính một phần tử
   `C[row,col]`, đọc trực tiếp từ Global Memory và thực hiện vòng lặp theo `k`.
   Đây là kernel cơ sở để so sánh; mỗi thread duyệt `B` theo cột với stride
   `N` giữa các bước `k` và không tái sử dụng tile trong shared memory.
2. **CUDA Shared Memory Tiling** (`CUDA_SHARED_MEMORY_TILING`): mỗi block nạp
   tile con của `A` và `B` vào shared memory, đồng bộ bằng `__syncthreads()` và
   tái sử dụng tile trong phép nhân. Các cạnh ma trận không chia hết tile được
   xử lý bằng zero-padding logic (không cần cấp phát ma trận đệm).

Cả hai kernel dùng `double` để khớp kiểu dữ liệu của lớp `Matrix`. Kích thước
block/tile CUDA được cấu hình là 16 hoặc 32 (mặc định 16); với tile 32,
shared-memory kernel dùng 16 KiB shared memory/block. Lựa chọn và cấu hình
được lưu ở cột `CUDA_Block_Tile` trong CSV/Excel; tên GPU và compute
capability được lấy từ thiết bị đang chạy và ghi trong banner/bảng thông số.
Thời gian benchmark GPU là thời gian end-to-end trên host, bao gồm cấp phát
device buffer và truyền dữ liệu A/B/C; lượt warm-up được chạy ngoài vùng đo.

API:

```cpp
matmul_cuda_global_memory(A, B, C, 16);
matmul_cuda_shared_memory_tiling(A, B, C, 16);
```

### 2.4. Chạy trên Google Colab với NVIDIA T4

Trong Colab, chọn **Runtime → Change runtime type → T4 GPU** (nếu được cấp),
xác nhận GPU và build cho compute capability 7.5:

```bash
!nvidia-smi
!nvcc --version
!git clone https://github.com/thuy0an/Matrix_Multiplication.git
%cd /content/Matrix_Multiplication
!make clean
!make CUDA_ENABLED=1 CUDA_ARCH=sm_75
!./matmul_benchmark -N 512,1024,2048 -t 1,4,8 -r 3 --cuda-tile 16 -o t4_results.csv
```

Có thể đổi `--cuda-tile 16` thành `--cuda-tile 32` để so sánh block/tile.
T4 có 16 GB GDDR6 và hỗ trợ compute capability 7.5; hiệu năng FP64 thấp hơn
nhiều so với FP32, vì vậy kết quả ở đây dùng để so sánh đúng kiểu `double`
của dự án chứ không đại diện cho hiệu năng GEMM FP32/TF32 tối ưu hóa thư viện.
Thời gian cloud GPU phụ thuộc vào quota và loại runtime Colab được cấp.

### 2.5. Quản lý Bộ nhớ Căn chỉnh 64-Byte (Aligned Memory)

Dữ liệu ma trận được cấp phát qua `_aligned_malloc(bytes, 64)` trên Windows và `posix_memalign` trên Linux, khớp với kích thước Cache Line 64B và thanh ghi SIMD AVX2 256-bit, tránh phạt chu kỳ do unaligned memory access.

### 2.6. Kiểm định Tính Toàn vẹn Dữ liệu (Sai số Tương đối Frobenius Norm)

Hệ thống sử dụng chuẩn Frobenius tương đối thay vì so sánh tuyệt đối:

$$
\epsilon_{\text{rel}} = \frac{\|C_{\text{parallel}} - C_{\text{ref}}\|_F}{\|C_{\text{ref}}\|_F} < 10^{-5}
$$

Đảm bảo tính chính xác số học trong môi trường tính toán dấu chấm động đa luồng (IEEE 754).

### 2.7. Thuật toán kết hợp CPU OpenMP + GPU CUDA

`matmul_heterogeneous_omp_cuda` chia ma trận theo dải hàng: GPU xử lý mặc
định 80% số hàng bằng kernel tiled trong shared memory, còn CPU xử lý phần
khác bằng OpenMP và cache tiling. Hai nhánh được khởi chạy đồng thời bằng
`omp parallel sections`; mỗi thiết bị chỉ ghi vào vùng hàng riêng của `C`.

Backend CUDA là tùy chọn để dự án vẫn build được trên máy chỉ có CPU:

```bash
# Bản CPU mặc định (CUDA backend stub)
make

# Bật mã CUDA, cần cài nvcc và có GPU NVIDIA
make clean
make CUDA_ENABLED=1
```

Mặc định `CUDA_ARCH=sm_75` phù hợp với T4. Có thể đặt kiến trúc khác khi
build, ví dụ `make CUDA_ENABLED=1 CUDA_ARCH=sm_86` cho RTX 3050 Ti. Trên
Windows, có thể đặt biến môi trường `CUDA_ARCH` trước khi chạy `run.bat`.

`run.bat` và `run_benchmarks.sh` tự động phát hiện `nvcc`. Khi CUDA khả
dụng, benchmark sẽ build kernel CUDA; nếu không, chương trình build bản
CPU-only và bỏ qua các phép đo GPU. Khi GPU hoạt động, chế độ `--auto` và
benchmark chuẩn đo thuật toán `5. Ket hop CPU + CUDA GPU` cho từng mức
OpenMP thread, cùng hai kernel GPU độc lập (Global Memory và Shared Memory
Tiling) một lần cho mỗi kích thước N. Các cột thời gian, GFLOPS, speedup,
hiệu suất, sai số và CUDA block/tile được xuất trong kết quả.

API được khai báo trong [include/algorithms.h](./include/algorithms.h):

```cpp
matmul_heterogeneous_omp_cuda(A, B, C, cpu_threads, 0.8, 16);
```

Hàm báo lỗi rõ ràng nếu gọi bản kết hợp khi chưa build CUDA hoặc không có
thiết bị NVIDIA. `cuda_tile` chỉ nhận 16 hoặc 32 để phù hợp kernel shared
memory.

Tùy chọn dòng lệnh `--cuda-tile 16|32` áp dụng cho cả hai kernel độc lập
(block size với Global Memory, tile size với Shared Memory) và nhánh hybrid.

### 2.8. Hướng dẫn Mở rộng Thuật toán Mới

1. Thêm định danh thuật toán vào `enum class AlgorithmType` trong [include/common.h](./include/common.h).
2. Khai báo hàm trong [include/algorithms.h](./include/algorithms.h).
3. Hiện thực hóa trong [src/algorithms.cpp](./src/algorithms.cpp) và thêm nhánh vào `run_algorithm()`.
4. Biên dịch lại bằng lệnh `make` hoặc `run.bat`.

---

## 3. BẢNG DỮ LIỆU ĐO LƯỜNG VÀ HIỆU NĂNG THỰC TẾ

### 3.1. Kết quả Thực nghiệm trên CPU AMD Ryzen 7 6800H ($N = 2048 \times 2048$)

| Thuật toán                                | Số Luồng | Thời gian ($T$) |   Tốc độ (GFLOPS)   | Tăng tốc (Speedup) | Sai số tương đối ($\epsilon$) | Đánh giá                                       |
| :------------------------------------------ | :--------: | :----------------: | :--------------------: | :------------------: | :----------------------------------: | :------------------------------------------------ |
| **1. Tuần tự Baseline**             |   1 Core   |  **3.060s**  |      5.61 GFLOPS      |        1.00x        |             Mốc chuẩn             | Mốc tuần tự ban đầu                          |
| **2. Tiled L2+L3 (Đơn luồng)**     |   1 Core   |  **2.118s**  |      8.11 GFLOPS      |   **1.44x**   |            $< 10^{-15}$            | **Nhanh hơn 44% chỉ nhờ tối ưu Cache** |
| **3. OpenMP Cơ bản (Chưa Tiling)** |  8 Cores  |       0.530s       |      32.41 GFLOPS      |        5.77x        |            $< 10^{-15}$            | Bị nghẽn do Cache Miss liên tục               |
| **4. OpenMP Tiled L2+L3**             |  8 Cores  |  **0.420s**  | **40.90 GFLOPS** |   **7.28x**   |            $< 10^{-15}$            | **Đạt 91% hiệu suất vật lý tối đa** |
| **5. OpenMP Bypass L2**               |  8 Cores  |  **0.423s**  | **40.61 GFLOPS** |   **7.23x**   |            $< 10^{-15}$            | Bơm trực tiếp L3 xuống L1                     |
| **6. OpenMP Tiled L2+L3**             | 16 Threads |  **0.403s**  | **42.63 GFLOPS** |   **7.59x**   |            $< 10^{-15}$            | Tận dụng toàn bộ luồng logic SMT             |

---

### 3.2. Bảng Đo lường Bộ nhớ đệm Cache & Phần cứng (Hardware Counters tại $N = 2048$)

| Thông số Đo lường Phần cứng              | Thuật toán OpenMP Cơ bản | Thuật toán OpenMP Tiled L2+L3 |                       Mức độ Cải thiện                       |
| :---------------------------------------------- | :--------------------------: | :-----------------------------: | :----------------------------------------------------------------: |
| **Thời gian thực thi ($T$)**          |         0.284 giây         |      **0.158 giây**      |                     Nhanh hơn **1.80x**                     |
| **Tốc độ tính toán**                 |         60.5 GFLOPS         |     **108.7 GFLOPS**     |                Tăng **80% công suất CPU**                |
| **Instructions Per Cycle (IPC)**          |       1.12 insn/cycle       |    **2.38 insn/cycle**    |         Tăng **2.12x** (CPU ít bị chờ dữ liệu)         |
| **Tỷ lệ trượt L1 Data Cache**         |            18.4%            |         **3.9%**         |                  Giảm **78.8%** trượt L1                  |
| **Tỷ lệ trượt L3 / LLC Cache**        |       **38.6%**       |         **4.8%**         | **Giảm từ 38.6% xuống 4.8% (Giảm 8 lần trượt Cache)** |
| **Chu kỳ đình trệ chờ RAM (Stalls)** |      42.5% tổng cycles      |   **8.1% tổng cycles**   |            CPU không còn bị nghẽn băng thông RAM            |

---

### 3.3. Bảng Phân tích Chi phí Điện toán (So sánh CPU vs GPU cho 1.000.000 lượt nhân $N = 2048$)

> Bảng GPU bên dưới là số liệu tham khảo cuBLAS/tiled có sẵn từ trước, không
> phải kết quả đo của hai kernel CUDA Global/Shared Memory mới trong chương
> trình. Hãy chạy benchmark trên GPU đích và dùng CSV xuất ra để so sánh trực tiếp.

| Cấu hình Phần cứng                     | Thời gian / 1 phép nhân | Tổng thời gian (1M lượt) | Đơn giá thuê Cloud | Tổng chi phí hoàn thành |                      Đánh giá Kinh tế                      |
| :----------------------------------------- | :------------------------: | :--------------------------: | :--------------------: | :-------------------------: | :-------------------------------------------------------------: |
| **CPU Tuần tự (1 Core)**           |        1.820 giây        |          505.5 giờ          |      $0.68 / giờ      |      **$343.74**      |                  Mốc đối chứng đắt nhất                  |
| **CPU OpenMP Cơ bản (16 Cores)**   |        0.284 giây        |          78.9 giờ          |      $0.68 / giờ      |      **$53.65**      |                        Tiết kiệm 6.4x                        |
| **CPU Tiled L2/L3 (Nhóm tối ưu)** |   **0.158 giây**   |     **43.9 giờ**     | **$0.68 / giờ** |      **$29.85**      |        **Tiết kiệm thêm 44.4% tiền thuê CPU**        |
| **GPU NVIDIA T4 (cuBLAS / Tiled)**   |   0.0028 giây (2.8 ms)   |          0.78 giờ          |     $0.526 / giờ     |       **$0.41**       | **Rẻ hơn 72x so với CPU Tiled, 838x so với Baseline** |

> **Khuyến nghị kiến trúc**:
>
> - Với các tác vụ tính toán ma trận khổng lồ ($N \ge 1024$) chạy theo lô lớn (*Batch Processing*), chuyển sang GPU mang lại lợi thế vượt trội về cả tốc độ lẫn chi phí.
> - Với các tác vụ vi mô thời gian thực ($N \le 512$) hoặc chạy ngắt quãng (*Serverless/Spike*), tối ưu **Tiled L2/L3 trên CPU** là giải pháp tối ưu nhất, loại bỏ chi phí sao chép dữ liệu qua khe PCIe bus và tránh lãng phí tiền thuê GPU nhàn rỗi.

---

## 4. CÁC TẬP TIN DỮ LIỆU ĐẦU RA

* **`benchmark_results.xls` / `results.xls`**: Bảng tính Microsoft Excel có màu sắc, lưu trữ các thông số đo, cấu hình CUDA block/tile, cấu hình phần cứng CPU và công thức phân tích.
* **`benchmark_results.csv` / `results.csv`**: Dữ liệu CSV phân tách dấu phẩy chuẩn RFC 4180, sẵn sàng nạp vào Python (Pandas/Matplotlib), MATLAB hoặc R để vẽ biểu đồ scaling.
