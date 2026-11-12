# ==============================================================================
# MAKEFILE CHUẨN CÔNG NGHIỆP: NHÂN MA TRẬN SONG SONG (C++/OpenMP/AVX2)
# Hỗ trợ đa nền tảng: Linux, macOS, Windows (MinGW/UCRT64)
# ==============================================================================

CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp -mavx2 -mfma -Wall -Wextra
INCLUDES = -Iinclude
CUDA_ENABLED ?= 0
NVCC = nvcc
CUDA_ARCH ?= sm_75
CUDAFLAGS = -arch=$(CUDA_ARCH) -std=c++17 -O3 -Xcompiler -fopenmp

SRCDIR = src
OBJDIR = obj
BINDIR = .

SRCS = $(SRCDIR)/main.cpp \
       $(SRCDIR)/matrix.cpp \
       $(SRCDIR)/algorithms.cpp \
       $(SRCDIR)/excel_export.cpp

ifeq ($(CUDA_ENABLED),1)
    CUDA_SRCS = $(SRCDIR)/cuda_backend.cu
    CUDA_OBJS = $(CUDA_SRCS:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
else
    SRCS += $(SRCDIR)/cuda_backend_stub.cpp
    CUDA_OBJS =
endif

OBJS = $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# Phát hiện hệ điều hành (Windows vs Linux/Unix)
ifeq ($(OS),Windows_NT)
    TARGET = $(BINDIR)/matmul_benchmark.exe
    RM = del /Q /F
    MKDIR = if not exist $(OBJDIR) mkdir $(OBJDIR)
    CUDAFLAGS = -arch=$(CUDA_ARCH) -std=c++17 -O3 -Xcompiler /openmp:experimental
else
    TARGET = $(BINDIR)/matmul_benchmark
    RM = rm -rf
    MKDIR = mkdir -p $(OBJDIR)
endif

# Target mặc định
.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS) $(CUDA_OBJS)
	@echo [LINK] Tạo file thực thi: $(TARGET)
ifeq ($(CUDA_ENABLED),1)
	$(NVCC) $(CUDAFLAGS) $(OBJS) $(CUDA_OBJS) -o $(TARGET)
else
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)
endif
	@echo [DONE] Biên dịch hoàn tất thành công!

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@$(MKDIR)
	@echo [CXX] Biên dịch: $<
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@$(MKDIR)
	@echo [NVCC] Biên dịch: $<
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) -c $< -o $@

# Chạy tự động toàn bộ Benchmark
.PHONY: benchmark
benchmark: $(TARGET)
	@echo [BENCHMARK] Bắt đầu chạy Benchmark tự động...
	$(TARGET) --auto

# Chạy tương tác
.PHONY: run
run: $(TARGET)
	$(TARGET)

# Dọn dẹp file nhị phân
.PHONY: clean
clean:
	@echo [CLEAN] Dọn dẹp file tạm...
ifeq ($(OS),Windows_NT)
	if exist $(OBJDIR) rmdir /S /Q $(OBJDIR)
	if exist $(TARGET) del /Q /F $(TARGET)
else
	$(RM) $(OBJDIR) $(TARGET)
endif

# Hướng dẫn
.PHONY: help
help:
	@echo "Danh sách lệnh make khả dụng:"
	@echo "  make            : Biên dịch chương trình (tạo $(TARGET))"
	@echo "  make benchmark  : Biên dịch và chạy bộ benchmark tự động"
	@echo "  make run        : Biên dịch và chạy chế độ tương tác Menu"
	@echo "  make CUDA_ENABLED=1 CUDA_ARCH=sm_75 : Bật CUDA (sm_75 mặc định cho NVIDIA T4)"
	@echo "  make clean      : Xóa toàn bộ file object và binary"
