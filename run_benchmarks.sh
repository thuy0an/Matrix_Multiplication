#!/usr/bin/env bash
# ==============================================================================
# SCRIPT TỰ ĐỘNG HÓA BENCHMARK CHUẨN KỸ SƯ (CI / HPC AUTOMATION)
# Tự động biên dịch, quét kích thước N = 512, 1024, 2048, 4096,
# chạy lặp 5 lần lấy trung bình (loại trừ nhiễu OS) và xuất results.csv
# ==============================================================================

set -e

# Định nghĩa màu sắc Terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}====================================================================${NC}"
echo -e "${CYAN}     HỆ THỐNG BENCHMARK TỰ ĐỘNG HÓA CHUẨN KỸ SƯ (REPRODUCIBILITY)     ${NC}"
echo -e "${CYAN}====================================================================${NC}"
echo ""

# Luôn biên dịch lại để benchmark chắc chắn dùng backend CPU/CUDA mới nhất.
echo -e "${YELLOW}[INFO] Đang biên dịch mã nguồn mới nhất...${NC}"
if command -v make &> /dev/null; then
    if command -v nvcc &> /dev/null; then
        make CUDA_ENABLED=1 all
    else
        make CUDA_ENABLED=0 all
    fi
else
    echo -e "${YELLOW}[INFO] Biên dịch trực tiếp bằng g++...${NC}"
    if command -v nvcc &> /dev/null; then
        nvcc -std=c++17 -O3 -Xcompiler -fopenmp -Iinclude \
            src/main.cpp src/matrix.cpp src/algorithms.cpp src/excel_export.cpp \
            src/cuda_backend.cu -o matmul_benchmark
    else
        g++ -std=c++17 -O3 -fopenmp -mavx2 -mfma -Iinclude \
            src/main.cpp src/matrix.cpp src/algorithms.cpp src/excel_export.cpp \
            src/cuda_backend_stub.cpp -o matmul_benchmark
    fi
fi

if [ -f "./matmul_benchmark.exe" ]; then
    EXE="./matmul_benchmark.exe"
else
    EXE="./matmul_benchmark"
fi

echo -e "${GREEN}[+] File thực thi sẵn sàng: ${EXE}${NC}"
echo ""

# Thiết lập tham số đo lường
SIZES="512,1024,2048,4096"
THREADS="1,2,4,8,16"
RUNS=5
OUTPUT_CSV="results.csv"
OUTPUT_XLS="results.xls"

echo -e "${BLUE}>>> CẤU HÌNH THỰC NGHIỆM TỰ ĐỘNG:${NC}"
echo -e "  * Kích thước ma trận (N) : ${YELLOW}${SIZES}${NC}"
echo -e "  * Số luồng OpenMP (Threads): ${YELLOW}${THREADS}${NC}"
echo -e "  * Số lần lặp lấy trung bình: ${YELLOW}${RUNS} lần/cấu hình (Loại trừ Cold Cache / OS Noise)${NC}"
echo -e "  * File dữ liệu đầu ra    : ${GREEN}${OUTPUT_CSV}${NC} và ${GREEN}${OUTPUT_XLS}${NC}"
echo ""

START_TIME=$(date +%s)

# Khởi chạy thực thi
echo -e "${CYAN}>>> ĐANG TIẾN HÀNH CHẠY TOÀN BỘ KỊCH BẢN THỰC NGHIỆM...${NC}"
${EXE} -N "${SIZES}" -t "${THREADS}" -r "${RUNS}" -o "${OUTPUT_CSV}"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}====================================================================${NC}"
echo -e "${GREEN}[THÀNH CÔNG] Toàn bộ kịch bản Benchmark tự động đã hoàn thành!${NC}"
echo -e "${GREEN}Tổng thời gian thực hiện: ${TOTAL_TIME} giây${NC}"
echo -e "${GREEN}[+] Đã xuất file CSV: ${OUTPUT_CSV}${NC}"
echo -e "${GREEN}[+] Đã xuất file Excel: ${OUTPUT_XLS}${NC}"
echo -e "${GREEN}====================================================================${NC}"
