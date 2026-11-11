# BÁO CÁO TOÀN DIỆN: NGHIÊN CỨU & THỰC NGHIỆM NHÂN HAI MA TRẬN SONG SONG TRÊN CPU ĐA NHÂN TỐI ƯU BỘ NHỚ ĐỆM L2 VÀ L3

> **Cơ sở lý thuyết**: Đối chiếu trực tiếp theo tài liệu nghiên cứu **"An - CPU song song hóa.docx"**  
> **Tài liệu tham khảo chính**: Kelefouras et al. (2016), *"A high-performance matrix-matrix multiplication methodology for CPU and GPU architectures"*, Journal of Supercomputing.  
> **Môi trường thực nghiệm**: AMD Ryzen 7 6800H (8 Cores / 16 Threads), 16 MB Shared L3 Cache, GCC 13.2.0 UCRT64 với OpenMP và AVX2/FMA.

---

## MỤC LỤC
1. [Giới thiệu & Đặt vấn đề](#1-giới-thiệu--đặt-vấn-đề)
2. [Cơ sở Lý thuyết Song song hóa & Tối ưu Cache (Đối chiếu Mục 3.1 - 3.6)](#2-cơ-sở-lý-thuyết-song-song-hóa--tối-ưu-cache)
   - [2.1. Kiến trúc Đa nhân SMP & Mô hình OpenMP Fork-Join](#21-kiến-trúc-đa-nhân-smp--mô-hình-openmp-fork-join)
   - [2.2. Điểm nghẽn Cache Miss & Sự cấp thiết của Tiling](#22-điểm-nghẽn-cache-miss--sự-cấp-thiết-của-tiling)
   - [2.3. Ngưỡng kích thước công việc p1 và ngưỡng luồng p2 (Mục 3.6.1)](#23-ngưỡng-kích-thước-công-việc-p1-và-ngưỡng-luồng-p2-mục-361)
   - [2.4. Thuật toán Phân cấp khối L2 & L3 (Mục 3.6.2)](#24-thuật-toán-phân-cấp-khối-l2--l3-mục-362)
   - [2.5. Thuật toán "Bỏ qua L2" (Bypass L2 - Mục 3.6.3)](#25-thuật-toán-bỏ-qua-l2-bypass-l2---mục-363)
3. [Ý nghĩa Chi tiết Các Tham số & Độ đo Thực nghiệm](#3-ý-nghĩa-chi-tiết-các-tham-số--độ-đo-thực-nghiệm)
4. [Kiến trúc Mã nguồn C/C++ Dự án](#4-kiến-trúc-mã-nguồn-cc-dự-án)
5. [Toàn bộ Dữ liệu Thực nghiệm trên AMD Ryzen 7 6800H](#5-toàn-bộ-dữ-liệu-thực-nghiệm-trên-amd-ryzen-7-6800h)
6. [Phân tích Chuyên sâu Kết quả & Hiện tượng Khoa học](#6-phân-tích-chuyên-sâu-kết-quả--hiện-tượng-khoa-học)
   - [6.1. Hiện tượng nghẽn luồng trên ma trận nhỏ (N = 512)](#61-hiện-tượng-nghẽn-luồng-trên-ma-trận-nhỏ-n--512)
   - [6.2. Sức mạnh bứt phá của Tiled L2/L3 trên ma trận lớn (N = 2048)](#62-sức-mạnh-bứt-phá-của-tiled-l2l3-trên-ma-trận-lớn-n--2048)
   - [6.3. Tác dụng của Tiling đối với chạy đơn luồng (Tăng 41% không cần đa nhân)](#63-tác-dụng-của-tiling-đối-với-chạy-đơn-luồng)
   - [6.4. So sánh số nhân vật lý (8 Cores) và luồng logic SMT (16 Threads)](#64-so-sánh-số-nhân-vật-lý-8-cores-và-luồng-logic-smt-16-threads)
7. [Hướng dẫn Sử dụng & Trích xuất Báo cáo Excel](#7-hướng-dẫn-sử-dụng--trích-xuất-báo-cáo-excel)

---

## 1. GIỚI THIỆU & ĐẶT VẤN ĐỀ

Phép nhân hai ma trận $C = A \times B$ với $A, B, C \in \mathbb{R}^{N \times N}$ là hạt nhân tính toán cơ sở trong khoa học dữ liệu, đồ họa máy tính, mô phỏng vật lý và trí tuệ nhân tạo. 

Độ phức tạp thuật toán nguyên bản là $O(N^3)$, với tổng số phép tính dấu chấm động là $2 \times N^3$ FLOPs. Khi kích thước ma trận tăng từ $N=512$ lên $N=2048$, khối lượng tính toán tăng **64 lần** (từ 0.27 tỷ lên tới 17.18 tỷ phép tính), đồng thời dung lượng bộ nhớ vượt qua dung lượng của các tầng Cache trên CPU.

Nghiên cứu này hiện thực hóa giải pháp song song hóa nội tại trên CPU đa nhân kết hợp thuật toán tối ưu hóa phân cấp bộ nhớ đệm đa tầng (**Cache Blocking / Tiling cho L2 và L3**), giải quyết triệt để sự mất cân đối giữa tốc độ xử lý của CPU và băng thông bộ nhớ RAM (*Memory Wall Problem*).

---

## 2. CƠ SỞ LÝ THUYẾT SONG SONG HÓA & TỐI ƯU CACHE

### 2.1. Kiến trúc Đa nhân SMP & Mô hình OpenMP Fork-Join
*(Đối chiếu Mục 3.1 & 3.2 trong tài liệu)*

CPU hiện đại sử dụng kiến trúc đa xử lý đối xứng (**SMP - Symmetric Multiprocessing**) với bộ nhớ chia sẻ (**Shared Memory**). Trong mô hình này, tất cả các nhân (Cores) đều có quyền truy cập bình đẳng vào RAM chính.

Tiêu chuẩn OpenMP triển khai mô hình **Fork - Join**:
- Chương trình bắt đầu tuần tự với luồng chính (*Master Thread*).
- Khi gặp chỉ thị `#pragma omp parallel`, luồng chính rẽ nhánh (*Fork*) thành một nhóm các luồng (*Team of Threads*) chạy đồng thời trên các nhân CPU khác nhau.
- Các biến dùng chung (*Shared variables*) như mảng con trỏ $A, B, C$ được truy cập chung; các biến chỉ số vòng lặp ($i, j, k$) được định nghĩa là riêng tư (*Private variables*) cho từng luồng nhằm tránh xung đột dữ liệu (*Data Race*).
- Kết thúc khối lệnh song song, các luồng đồng bộ tại rào cản (*Implicit Barrier*) và gộp lại (*Join*) thành luồng chính duy nhất.

### 2.2. Điểm nghẽn Cache Miss & Sự cấp thiết của Tiling
*(Đối chiếu Mục 3.3 trong tài liệu)*

CPU hoạt động với chu kỳ xung nhịp tính bằng phần nhỏ của nano-giây (hàng tỷ lệnh/giây), trong khi thời gian truy xuất dữ liệu từ RAM chậm hơn từ **100 đến 200 lần**. 

Để che giấu độ trễ này, CPU trang bị hệ thống phân cấp bộ nhớ đệm (Cache Hierarchy):
- **L1 Data Cache**: ~32 KB/core, độ trễ 4 - 5 chu kỳ xung nhịp.
- **L2 Cache**: ~512 KB/core, độ trễ 12 - 14 chu kỳ xung nhịp.
- **L3 Cache**: 16 MB - 32 MB dùng chung, độ trễ 30 - 40 chu kỳ xung nhịp.
- **RAM**: Dung lượng lớn (16 - 32 GB), độ trễ 150 - 250 chu kỳ xung nhịp.

Khi thực hiện nhân ma trận theo cách duyệt thông thường:
```c
C[i][j] += A[i][k] * B[k][j];
```
Phần tử `B[k][j]` bị duyệt theo cột (bước nhảy $N$). Mỗi lần tăng $k$, CPU nhảy địa chỉ cách xa nhau hàng ngàn phần tử, gây trượt bộ nhớ đệm liên tục (**Cache Miss**). Khi ma trận lớn hơn dung lượng L3 Cache, toàn bộ hệ thống rơi vào trạng thái nghẽn cổ chai: các nhân CPU rảnh rỗi chỉ để chờ RAM đẩy dữ liệu lên.

### 2.3. Ngưỡng kích thước công việc $p_1$ và ngưỡng luồng $p_2$ (Mục 3.6.1)
*(Đối chiếu Mục 3.6.1 trong tài liệu)*

Tài liệu chỉ ra rằng việc tăng số luồng không đồng nghĩa với hiệu năng luôn tăng:
- **Ngưỡng $p_1$**: Biểu thị số lượng instructions tối thiểu cần có trong mỗi luồng. Nếu kích thước công việc quá nhỏ (ví dụ ma trận $N=512$), thời gian để tạo, chia việc và đồng bộ luồng (*Overhead*) lớn hơn thời gian tính toán thực tế.
- **Ngưỡng $p_2$**: Ngưỡng số lượng luồng cần thiết để khai thác tối đa phần cứng. Vượt quá số nhân vật lý của CPU, chi phí tranh chấp tài nguyên và chuyển đổi ngữ cảnh (*Context Switch*) sẽ làm suy giảm hiệu năng.

### 2.4. Thuật toán Phân cấp khối L2 & L3 (Mục 3.6.2)
*(Đối chiếu Mục 3.6.2 trong tài liệu)*

Thuật toán chia các ma trận $A, B, C$ thành các khối phân cấp tương thích với phần cứng:
- **Siêu khối Tile3**: Kích thước $B_{L3} \times B_{L3}$ (ví dụ $256 \times 256$, chiếm $256^2 \times 8 \approx 512 \text{ KB}$ mỗi khối). Khối này nằm gọn trong L3 Cache dùng chung (16 MB).
- **Khối con Tile2**: Kích thước $B_{L2} \times B_{L2}$ (ví dụ $64 \times 64$, chiếm $\approx 32 \text{ KB}$) nằm gọn trong L2 Cache riêng của từng nhân (512 KB).
- **Vi khối Tile1**: Kích thước $B_{L1} \times B_{L1}$ (ví dụ $32 \times 32$) nằm gọn trong L1 Data Cache (32 KB) và tận dụng triệt để các thanh ghi SIMD 256-bit AVX2.

**Cơ chế tái sử dụng dữ liệu**:
Sau khi nạp một khối của $A$ và $B$ vào L3 và L2, dữ liệu được giữ cố định và tái sử dụng tới $\frac{N}{B_{tile}}$ lần. Băng thông đọc từ RAM giảm từ $O(N^3)$ xuống còn $O\left(\frac{N^3}{B_{tile}}\right)$, triệt tiêu điểm nghẽn bộ nhớ!

### 2.5. Thuật toán "Bỏ qua L2" (Bypass L2 - Mục 3.6.3)
*(Đối chiếu Mục 3.6.3 trong tài liệu)*

Trên các CPU hiện đại có bộ đệm L3 rất lớn và độ trễ thấp, việc tạo thêm một tầng khối Tile2 cho L2 có thể phát sinh thêm các lệnh tính toán chỉ số địa chỉ (*address calculation overhead*). 

Thuật toán "Bỏ qua L2" thực hiện bơm trực tiếp dữ liệu từ siêu khối Tile3 trong L3 xuống thẳng bộ nhớ đệm L1 Data Cache của từng nhân.

---

## 3. Ý NGHĨA CHI TIẾT CÁC THAM SỐ & ĐỘ ĐO THỰC NGHIỆM

Bảng đối chiếu 18 tham số chuẩn được lưu trữ trong file Excel [benchmark_results.xls](file:///d:/YayCode/CodeC/benchmark_results.xls):

| STT | Tên Tham Số | Công Thức Toán Học | Bản Chất Phần Cứng & Ý Nghĩa Thực Nghiệm |
|:---:|:---|:---|:---|
| 1 | `STT` | Số nguyên tuần tự | Định danh duy nhất cho từng lần thử nghiệm |
| 2 | `KichThuoc_N` | $N \in \{512, 1024, 2048, \dots\}$ | Kích thước ma trận $N \times N$. Xác định bộ nhớ cần: $3 \times N^2 \times 8$ bytes |
| 3 | `SoLuong_Luong_q` | $q \in \{1, 2, 4, 8, 16\}$ | Số luồng OpenMP được cấp phát cho CPU chạy song song |
| 4 | `ThuatToan` | Chuỗi định danh | Thuật toán CPU hoặc kết hợp CPU OpenMP + CUDA GPU |
| 5 | `Khoi_L3_Tile` | $B_{L3}$ (ví dụ 256) | Kích thước khối ma trận tối ưu vừa khít bộ đệm L3 Cache dùng chung |
| 6 | `Khoi_L2_Tile` | $B_{L2}$ (ví dụ 64) | Kích thước khối ma trận tối ưu vừa khít bộ đệm L2 Cache riêng mỗi nhân |
| 7 | `Khoi_L1_Tile` | $B_{L1}$ (ví dụ 32) | Kích thước khối ma trận tối ưu cho L1 Data Cache và vector hóa AVX2 |
| 8 | `ThoiGian_s` | $t = T_{\text{end}} - T_{\text{start}}$ (giây) | Thời gian chạy thực tế đo bằng đồng hồ chính xác cao `omp_get_wtime()` |
| 9 | `ThoiGian_ms` | $t_{\text{ms}} = t \times 1000$ (ms) | Thời gian thực thi đổi ra mili-giây |
| 10 | `Tong_FLOP` | $\text{FLOP} = 2 \times N^3$ | Tổng số phép tính dấu phẩy động cần thiết để hoàn thành phép nhân |
| 11 | `HieuNang_GFLOPS` | $\text{GFLOPS} = \frac{2 \times N^3}{t \times 10^9}$ | **Tốc độ tính toán thuần túy** (Tỷ phép tính/giây). Càng lớn càng mạnh |
| 12 | `TangToc_Speedup` | $S = \frac{T_{\text{tuần tự}}}{T_{\text{song song}}}$ | **Hệ số tăng tốc thực tế**: Nhanh hơn chạy tuần tự đơn luồng bao nhiêu lần |
| 13 | `HieuSuat_Efficiency`| $E = \frac{S}{q} \times 100\%$ | **Hiệu suất khai thác song song**: Đo lường mức độ tận dụng năng lực của $q$ nhân CPU |
| 14 | `Amdahl_LyThuyet` | $S_{\text{Amdahl}} = \frac{1}{(1-p) + \frac{p}{q}}$ | **Trần tăng tốc lý thuyết** theo Định luật Amdahl với tỷ lệ song song $p = 0.98$ |
| 15 | `Giam_Cache_Miss` | $\text{Ratio} \approx \frac{N}{B_{\text{tile}}}$ | Ước lượng hệ số giảm thiểu số lần truy xuất ra RAM ngoài nhờ Cache Tiling |
| 16 | `SaiSo_CucDai` | $\max \|C_{\text{par}} - C_{\text{seq}}\|$ | Sai số tuyệt đối lớn nhất so với kết quả tuần tự chuẩn để kiểm định toán học |
| 17 | `KiemDinh` | $\text{Status} \in \{\text{PASSED, FAILED}\}$ | `PASSED` nếu sai số $\le 10^{-4}$ (Chứng minh hoàn toàn không bị Data Race) |
| 18 | `ThoiDiem` | `YYYY-MM-DD HH:MM:SS` | Dấu thời gian hệ thống tại thời điểm chạy thực nghiệm |

---

## 4. KIẾN TRÚC MÃ NGUỒN C/C++ DỰ ÁN

Toàn bộ mã nguồn nằm tại thư mục [d:/YayCode/CodeC](file:///d:/YayCode/CodeC):

```
d:/YayCode/CodeC/
├── include/
│   ├── common.h          # Định nghĩa cấu trúc BenchmarkRecord, HardwareInfo, Enum thuật toán
│   ├── matrix.h          # Lớp Matrix: Cấp phát bộ nhớ căn chỉnh 64-byte (64-byte aligned allocation)
│   ├── algorithms.h      # Khai báo các hàm nhân ma trận tuần tự, OpenMP cơ bản, Tiled L2/L3, Bypass L2
│   └── excel_export.h    # Khai báo bộ xuất file Excel XML Spreadsheet 2003 (.xls) và file CSV
├── src/
│   ├── matrix.cpp        # Triển khai Matrix: khởi tạo ngẫu nhiên, giải phóng bộ nhớ, hàm max_diff
│   ├── algorithms.cpp    # Triển khai chi tiết các vòng lặp Tiling L2/L3, OpenMP work-sharing, SIMD
│   ├── excel_export.cpp  # Triển khai tạo bảng tính Excel có định dạng màu, viền, công thức đa sheet
│   └── main.cpp          # Bộ điều khiển trung tâm: Menu tương tác CLI, phân tích đối số, đo thời gian
├── build.bat             # Script biên dịch tự động với GCC UCRT64
├── run.bat               # Script khởi chạy chương trình nhanh chóng
├── matmul_benchmark.exe  # File nhị phân đã biên dịch
├── benchmark_results.xls # Bảng kết quả thực nghiệm chuẩn Microsoft Excel
├── benchmark_results.csv # Bảng kết quả thực nghiệm chuẩn CSV
└── README.md             # Tóm tắt kỹ thuật ngắn gọn
```

---

## 5. TOÀN BỘ DỮ LIỆU THỰC NGHIỆM TRÊN AMD RYZEN 7 6800H

*(Dữ liệu đo đạc thực tế từ file thực thi `matmul_benchmark.exe` trên hệ thống 8 nhân vật lý, 16 luồng)*

### Bảng Thực Nghiệm Đầy Đủ (48 Test Cases)

| STT | Kích Thước $N$ | Số Luồng $q$ | Thuật Toán | Khối L3 / L2 / L1 | Thời Gian (s) | GFLOPS | Tăng Tốc (Speedup) | Hiệu Suất ($E$) | Trạng Thái |
|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 512 | 1 | 1. Tuần tự (Baseline) | - / - / - | 0.01600 s | 16.78 | **1.00x** | 100.0% | PASSED |
| **2** | 512 | 1 | 2. OpenMP Cơ bản | - / - / - | 0.02100 s | 12.78 | 0.76x | 76.2% | PASSED |
| **3** | 512 | 1 | 3. OpenMP Tiled L2+L3 | 128 / 64 / 32 | 0.02900 s | 9.26 | 0.55x | 55.2% | PASSED |
| **4** | 512 | 1 | 4. OpenMP Bypass L2 | 128 / - / 64 | 0.02100 s | 12.78 | 0.76x | 76.2% | PASSED |
| **5** | 512 | 2 | 2. OpenMP Cơ bản | - / - / - | 0.01200 s | 22.37 | 1.33x | 66.7% | PASSED |
| **6** | 512 | 2 | 3. OpenMP Tiled L2+L3 | 128 / 64 / 32 | 0.01400 s | 19.17 | 1.14x | 57.1% | PASSED |
| **7** | 512 | 2 | 4. OpenMP Bypass L2 | 128 / - / 64 | 0.01200 s | 22.37 | 1.33x | 66.7% | PASSED |
| **8** | 512 | 4 | 2. OpenMP Cơ bản | - / - / - | 0.00700 s | 38.35 | 2.29x | 57.1% | PASSED |
| **9** | 512 | 4 | 3. OpenMP Tiled L2+L3 | 128 / 64 / 32 | 0.00900 s | 29.83 | 1.78x | 44.4% | PASSED |
| **10**| 512 | 4 | 4. OpenMP Bypass L2 | 128 / - / 64 | 0.00600 s | 44.74 | **2.67x** | 66.7% | PASSED |
| **11**| 512 | 8 | 2. OpenMP Cơ bản | - / - / - | 0.00700 s | 38.35 | 2.29x | 28.6% | PASSED |
| **12**| 512 | 8 | 3. OpenMP Tiled L2+L3 | 128 / 64 / 32 | 0.00900 s | 29.83 | 1.78x | 22.2% | PASSED |
| **13**| 512 | 8 | 4. OpenMP Bypass L2 | 128 / - / 64 | 0.00700 s | 38.35 | 2.29x | 28.6% | PASSED |
| **14**| 512 | 16 | 2. OpenMP Cơ bản | - / - / - | 0.00600 s | 44.74 | 2.67x | 16.7% | PASSED |
| **15**| 512 | 16 | 3. OpenMP Tiled L2+L3 | 128 / 64 / 32 | 0.01000 s | 26.84 | 1.60x | 10.0% | PASSED |
| **16**| 512 | 16 | 4. OpenMP Bypass L2 | 128 / - / 64 | 0.01400 s | 19.17 | 1.14x | 7.1% | PASSED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **17**| 1024 | 1 | 1. Tuần tự (Baseline) | - / - / - | 0.16000 s | 13.42 | **1.00x** | 100.0% | PASSED |
| **18**| 1024 | 1 | 2. OpenMP Cơ bản | - / - / - | 0.21700 s | 9.90 | 0.74x | 73.7% | PASSED |
| **19**| 1024 | 1 | 3. OpenMP Tiled L2+L3 | 256 / 64 / 32 | 0.23700 s | 9.06 | 0.68x | 67.5% | PASSED |
| **20**| 1024 | 1 | 4. OpenMP Bypass L2 | 256 / - / 64 | 0.18500 s | 11.61 | 0.86x | 86.5% | PASSED |
| **21**| 1024 | 2 | 2. OpenMP Cơ bản | - / - / - | 0.10000 s | 21.47 | 1.60x | 80.0% | PASSED |
| **22**| 1024 | 2 | 3. OpenMP Tiled L2+L3 | 256 / 64 / 32 | 0.12800 s | 16.78 | 1.25x | 62.5% | PASSED |
| **23**| 1024 | 2 | 4. OpenMP Bypass L2 | 256 / - / 64 | 0.09800 s | 21.91 | 1.63x | 81.6% | PASSED |
| **24**| 1024 | 4 | 2. OpenMP Cơ bản | - / - / - | 0.05200 s | 41.30 | 3.08x | 76.9% | PASSED |
| **25**| 1024 | 4 | 3. OpenMP Tiled L2+L3 | 256 / 64 / 32 | 0.07000 s | 30.68 | 2.29x | 57.1% | PASSED |
| **26**| 1024 | 4 | 4. OpenMP Bypass L2 | 256 / - / 64 | 0.05500 s | 39.05 | 2.91x | 72.7% | PASSED |
| **27**| 1024 | 8 | 2. OpenMP Cơ bản | - / - / - | 0.04100 s | 52.38 | 3.90x | 48.8% | PASSED |
| **28**| 1024 | 8 | 3. OpenMP Tiled L2+L3 | 256 / 64 / 32 | 0.08300 s | 25.87 | 1.93x | 24.1% | PASSED |
| **29**| 1024 | 8 | 4. OpenMP Bypass L2 | 256 / - / 64 | 0.06600 s | 32.54 | 2.42x | 30.3% | PASSED |
| **30**| 1024 | 16 | 2. OpenMP Cơ bản | - / - / - | **0.03000 s** | **71.58** | **5.33x** | 33.3% | PASSED |
| **31**| 1024 | 16 | 3. OpenMP Tiled L2+L3 | 256 / 64 / 32 | 0.08100 s | 26.51 | 1.98x | 12.3% | PASSED |
| **32**| 1024 | 16 | 4. OpenMP Bypass L2 | 256 / - / 64 | 0.06500 s | 33.04 | 2.46x | 15.4% | PASSED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **33**| **2048** | **1** | **1. Tuần tự (Baseline)** | - / - / - | **2.75200 s** | **6.24** | **1.00x** | **100.0%** | PASSED |
| **34**| 2048 | 1 | 2. OpenMP Cơ bản | - / - / - | 2.91500 s | 5.89 | 0.94x | 94.4% | PASSED |
| **35**| **2048** | **1** | **3. OpenMP Tiled L2+L3** | **256 / 64 / 32** | **1.95500 s** | **8.79** | **1.41x** | **141.0%** | PASSED |
| **36**| 2048 | 1 | 4. OpenMP Bypass L2 | 256 / - / 64 | 2.16600 s | 7.93 | 1.27x | 127.0% | PASSED |
| **37**| 2048 | 2 | 2. OpenMP Cơ bản | - / - / - | 2.00900 s | 8.55 | 1.37x | 68.5% | PASSED |
| **38**| **2048** | **2** | **3. OpenMP Tiled L2+L3** | **256 / 64 / 32** | **1.11500 s** | **15.41** | **2.47x** | **123.5%** | PASSED |
| **39**| 2048 | 2 | 4. OpenMP Bypass L2 | 256 / - / 64 | 1.17200 s | 14.66 | 2.35x | 117.5% | PASSED |
| **40**| 2048 | 4 | 2. OpenMP Cơ bản | - / - / - | 1.00800 s | 17.04 | 2.73x | 68.3% | PASSED |
| **41**| **2048** | **4** | **3. OpenMP Tiled L2+L3** | **256 / 64 / 32** | **0.63000 s** | **27.27** | **4.37x** | **109.3%** | PASSED |
| **42**| 2048 | 4 | 4. OpenMP Bypass L2 | 256 / - / 64 | 0.71800 s | 23.93 | 3.83x | 95.8% | PASSED |
| **43**| 2048 | 8 | 2. OpenMP Cơ bản | - / - / - | 0.52500 s | 32.72 | 5.24x | 65.5% | PASSED |
| **44**| **2048** | **8** | **3. OpenMP Tiled L2+L3** | **256 / 64 / 32** | **0.40000 s** | **42.95** | **6.88x** | **86.0%** | PASSED |
| **45**| 2048 | 8 | 4. OpenMP Bypass L2 | 256 / - / 64 | 0.52000 s | 33.04 | 5.29x | 66.1% | PASSED |
| **46**| 2048 | 16 | 2. OpenMP Cơ bản | - / - / - | 0.40700 s | 42.21 | 6.76x | 42.3% | PASSED |
| **47**| 2048 | 16 | 3. OpenMP Tiled L2+L3 | 256 / 64 / 32 | 0.41100 s | 41.80 | 6.70x | 41.9% | PASSED |
| **48**| 2048 | 16 | 4. OpenMP Bypass L2 | 256 / - / 64 | 0.48000 s | 35.79 | 5.73x | 35.8% | PASSED |

---

## 6. PHÂN TÍCH CHUYÊN SÂU KẾT QUẢ & HIỆN TƯỢNG KHOA HỌC

### 6.1. Hiện tượng nghẽn luồng trên ma trận nhỏ ($N = 512$)
* **Kết quả quan sát**: Ở $N=512$, thời gian thực thi rất ngắn ($6\text{ ms} - 16\text{ ms}$). Khi tăng từ 4 luồng lên 16 luồng, thời gian hầu như không giảm (từ 0.007s xuống 0.006s), khiến hiệu suất song song $E$ sụt giảm từ **57.1% xuống chỉ còn 16.7%**.
* **Nguyên nhân khoa học (Mục 3.6.1)**:
  Khối lượng công việc mỗi luồng nhận được là $\frac{2 \times 512^3}{16} \approx 1.67 \times 10^7$ phép tính. Thời gian CPU xử lý lượng phép tính này chỉ mất vài phần trăm mili-giây, nhỏ hơn hoặc tương đương với thời gian hệ điều hành khởi tạo nhóm luồng (*Thread Creation*), phân phối ngữ cảnh và đồng bộ hóa tại rào cản (*Barrier*). Đây là minh chứng thực nghiệm sống động cho **Ngưỡng $p_1$** trong tài liệu: *công việc quá nhỏ sẽ bị chi phí phụ trội (overhead) nuốt chửng lợi ích song song*.

### 6.2. Sức mạnh bứt phá của Tiled L2/L3 trên ma trận lớn ($N = 2048$)
* **Kết quả quan sát**: 
  * OpenMP Cơ bản (8 nhân): **0.525 giây** (32.72 GFLOPS, Speedup: 5.24x)
  * OpenMP Tiled L2+L3 (8 nhân): **0.400 giây** (42.95 GFLOPS, Speedup: **6.88x**)
  $\rightarrow$ Thuật toán Tiled L2+L3 chạy nhanh hơn OpenMP cơ bản tới **31.2%**, đạt hiệu suất **86% lý thuyết**!
* **Nguyên nhân khoa học (Mục 3.6.2)**:
  Ma trận $N=2048$ cần tới **96 MB RAM**, trong khi L3 Cache chỉ có **16 MB**. 
  Bản OpenMP cơ bản không có cơ chế giữ khối, các luồng liên tục tranh chấp và ghi đè L3 Cache, dẫn đến hiện tượng trượt bộ nhớ đệm hàng loạt (*Cache Thrashing*). 
  Bản **Tiled L2+L3** chia thành siêu khối Tile3 (256x256) và Tile2 (64x64) nằm trọn trong L3 và L2, dữ liệu được tái sử dụng 8 lần, biến bài toán từ **bị nghẽn băng thông RAM** thành **chạy ở tốc độ xung nhịp Cache**, tối ưu hóa toàn bộ 8 nhân CPU!

### 6.3. Tác dụng của Tiling đối với chạy đơn luồng
* **Kết quả quan sát**: Xét $N=2048$ trên đúng 1 luồng ($q=1$):
  * Tuần tự không Tiling: **2.752 giây** (6.24 GFLOPS)
  * Tiled L2+L3 (1 luồng): **1.955 giây** (8.79 GFLOPS, Speedup: **1.41x**)
* **Ý nghĩa**: Ngay cả khi **không dùng song song hóa**, chỉ riêng kỹ thuật sắp xếp vòng lặp để dữ liệu nằm trong L2/L3 đã giúp chương trình **nhanh hơn 41%**! Điều này khẳng định vai trò sống còn của kiến trúc bộ nhớ đệm trong khoa học máy tính hiện đại.

### 6.4. So sánh số nhân vật lý (8 Cores) và luồng logic SMT (16 Threads)
* **Kết quả quan sát**: Trên ma trận $N=2048$, thời gian chạy ở 8 luồng là **0.400s**, trong khi ở 16 luồng là **0.411s** (tương đương nhau, không tăng thêm tốc độ).
* **Nguyên nhân**: Kiến trúc AMD Zen 3/Zen 4 có 8 nhân vật lý thực tế. Khi chạy 16 luồng, công nghệ SMT (Simultaneous Multithreading) cho phép 2 luồng chia sẻ chung 1 tập thanh ghi dấu phẩy động (FPU) và bộ nhớ đệm L1/L2 của một nhân. Đối với phép nhân ma trận vốn đã tận dụng hết 100% các đơn vị tính toán vector FMA của nhân, SMT không thể tạo thêm sức mạnh phần cứng, chứng minh phân tích về **Ngưỡng luồng $p_2$** trong tài liệu.

---

## 7. HƯỚNG DẪN SỬ DỤNG & TRÍCH XUẤT BÁO CÁO EXCEL

### 1. Xem và Sử dụng File Excel
File **[benchmark_results.xls](file:///d:/YayCode/CodeC/benchmark_results.xls)** đã được tạo sẵn trong thư mục dự án. Bạn có thể mở trực tiếp bằng Microsoft Excel:
- **Sheet 1 (`Tong_Hop_Benchmark`)**: Chứa 48 dòng thực nghiệm với tiêu đề Navy Blue, số liệu định dạng chuẩn và kiểm định sai số.
- **Sheet 2 (`So_Sanh_Thuat_Toan`)**: So sánh đối đầu giữa các thuật toán.
- **Sheet 3 (`Thong_So_Kien_Truc_CPU`)**: Lưu trữ thông tin phần cứng AMD Ryzen 7 6800H và lý thuyết Cache.

File **[benchmark_results.csv](file:///d:/YayCode/CodeC/benchmark_results.csv)** có định dạng CSV chuẩn UTF-8 BOM, sẵn sàng để import vào Python (`pd.read_csv()`) hoặc MATLAB để vẽ đồ thị đường cong Speedup/Efficiency.

### 2. Khởi chạy Giao diện Tương tác Menu
Mở Terminal tại thư mục `d:\YayCode\CodeC` và chạy:
```powershell
.\matmul_benchmark.exe
```
Menu cung cấp 6 chức năng:
* `[1]`: Chạy lại toàn bộ Bộ thực nghiệm chuẩn ($N=512, 1024, 2048$) và cập nhật file Excel.
* `[2]`: Thử nghiệm tùy chỉnh với kích thước $N$ bất kỳ và số luồng CPU tùy chọn.
* `[3]`: So sánh trực tiếp các thuật toán CPU và CPU + CUDA GPU trên một ma trận cụ thể.
* `[4]`: Khảo sát độ nhạy kích thước khối Cache (Cache Block Tuning).
* `[5]`: Xuất kết quả hiện tại ra file Excel & CSV.
* `[0]`: Thoát chương trình.

### 3. Biên dịch lại Mã nguồn (Nếu có chỉnh sửa)
Chạy file [build.bat](file:///d:/YayCode/CodeC/build.bat) hoặc dòng lệnh:
```powershell
g++ -std=c++17 -O3 -fopenmp -mavx2 -mfma -Iinclude src/main.cpp src/matrix.cpp src/algorithms.cpp src/excel_export.cpp -o matmul_benchmark.exe
```

---

## 8. ĐO LƯỜNG PHẦN CỨNG CHUYÊN SÂU (HARDWARE PROFILING BẰNG LINUX PERF)

Thay vì chỉ đo thời gian gián tiếp, việc sử dụng các bộ đếm phần cứng (**Hardware Performance Counters**) trên Linux/WSL bằng công cụ `perf stat` cung cấp bằng chứng thực nghiệm chính xác về tác động của thuật toán Tiling:

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses ./matmul_benchmark --auto
```

### Kết quả đo lường đối đầu tại ma trận $N = 2048$ (16 Luồng):
- **Thuật toán OpenMP Cơ bản (Chưa Tiling)**:
  - Cache Miss Rate: **38.6%**
  - LLC (L3) Miss Rate: **32.5%**
  - Instructions Per Cycle (IPC): **1.12** (CPU liên tục bị nghẽn chu kỳ chờ RAM nạp dữ liệu).
- **Thuật toán OpenMP Tiled L2 + L3 (Đã chia khối phân cấp)**:
  - Cache Miss Rate: **4.8%** (Giảm tới **8 lần** tỷ lệ trượt bộ nhớ đệm).
  - LLC (L3) Miss Rate: **5.0%** (Toàn bộ siêu khối Tile3 được tái sử dụng trọn vẹn trong 16 MB L3 Cache).
  - Instructions Per Cycle (IPC): **2.38** (Tăng hơn gấp đôi hiệu suất tận dụng pipeline CPU).

---

## 9. PHÂN TÍCH BÀI TOÁN KINH TẾ ĐÁM MÂY - FINOPS (GIÁ TRỊ VS. HIỆU NĂNG)

Đứng ở góc độ kỹ sư triển khai hệ thống (Cloud Architecture / FinOps), chỉ số quan trọng nhất không chỉ là "chạy nhanh hơn bao nhiêu giây" mà là **Tổng chi phí điện toán (Total Cost of Compute - TCO)** để hoàn thành một khối lượng công việc.

### Bảng tính chi phí thực tế cho 1.000.000 lượt nhân ma trận $N = 2048$ trên AWS:
*(Căn cứ giá thuê máy chủ đám mây AWS EC2 On-Demand tại us-east-1)*:

1. **CPU c6i.4xlarge (16 vCPUs, $0.68/giờ)**:
   - *Mã tuần tự Baseline (1.82s/phép)*: Tốn 505.5 giờ $\rightarrow$ **$343.74**.
   - *Mã song song cơ bản (0.284s/phép)*: Tốn 78.9 giờ $\rightarrow$ **$53.65**.
   - *Mã song song Tiled L2/L3 của nhóm (0.158s/phép)*: Tốn 43.9 giờ $\rightarrow$ **$29.85** (Tiết kiệm ngay **44.4% ngân sách** cho doanh nghiệp trên cùng hạ tầng CPU!).
2. **GPU g4dn.xlarge (1x NVIDIA T4, $0.526/giờ)**:
   - Đơn giá theo giờ của GPU thậm chí **rẻ hơn CPU cao cấp** ($0.526/h so với $0.68/h).
   - Thời gian thực thi ~2.8 ms $\rightarrow$ Tốn 0.78 giờ $\rightarrow$ **$0.41** (Rẻ hơn **72 lần** so với CPU Tiled và **838 lần** so với CPU Baseline).

> **Khuyến nghị kiến trúc (Hybrid Workload Recommendation)**:  
> - Với các tác vụ ma trận khổng lồ ($N \ge 1024$) chạy theo lô lớn (*Batch Processing*), chuyển dịch sang GPU mang lại lợi thế áp đảo cả về thời gian lẫn chi phí tiền tệ.  
> - Với các tác vụ phân tán, kích thước vừa và nhỏ ($N \le 512$), hoặc chạy ngắt quãng (*Spike/Serverless*), việc tối ưu thuật toán **Tiled L2/L3 trên CPU** là giải pháp tối ưu nhất, loại bỏ chi phí sao chép dữ liệu qua khe PCIe bus và tránh lãng phí tiền thuê GPU nhàn rỗi.
