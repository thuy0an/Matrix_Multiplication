#include "cuda_backend.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

__global__ void matrix_multiply_tiled(const double* a, const double* b,
                                      double* c, int n, int gpu_rows,
                                      int row_offset) {
    extern __shared__ double shared[];
    double* tile_a = shared;
    double* tile_b = shared + blockDim.x * blockDim.x;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + ty;
    const int col = blockIdx.x * blockDim.x + tx;
    double sum = 0.0;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int a_col = tile + tx;
        const int b_row = tile + ty;
        tile_a[ty * blockDim.x + tx] =
            (row < gpu_rows && a_col < n) ? a[row * n + a_col] : 0.0;
        tile_b[ty * blockDim.x + tx] =
            (b_row < n && col < n) ? b[b_row * n + col] : 0.0;
        __syncthreads();

        for (int k = 0; k < blockDim.x && tile + k < n; ++k) {
            sum += tile_a[ty * blockDim.x + k] *
                   tile_b[k * blockDim.x + tx];
        }
        __syncthreads();
    }

    if (row < gpu_rows && col < n) {
        c[(row_offset + row) * n + col] = sum;
    }
}

__global__ void matrix_multiply_global(const double* a, const double* b,
                                       double* c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }

    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

bool matmul_cuda_full(const double* a, const double* b, double* c, int n,
                      int tile_size, bool use_shared_memory) {
    if (!cuda_backend_available() || n <= 0) {
        return false;
    }
    if (tile_size != 16 && tile_size != 32) {
        throw std::invalid_argument("CUDA block/tile size must be 16 or 32");
    }

    const size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(double);
    double* device_a = nullptr;
    double* device_b = nullptr;
    double* device_c = nullptr;

    try {
        check_cuda(cudaMalloc(&device_a, matrix_bytes), "cudaMalloc(d_A)");
        check_cuda(cudaMalloc(&device_b, matrix_bytes), "cudaMalloc(d_B)");
        check_cuda(cudaMalloc(&device_c, matrix_bytes), "cudaMalloc(d_C)");
        check_cuda(cudaMemcpy(device_a, a, matrix_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(A)");
        check_cuda(cudaMemcpy(device_b, b, matrix_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(B)");

        const int block_size = tile_size;
        const dim3 block(static_cast<unsigned>(block_size),
                         static_cast<unsigned>(block_size));
        const dim3 grid((n + block_size - 1) / block_size,
                        (n + block_size - 1) / block_size);
        if (use_shared_memory) {
            const size_t shared_bytes =
                2ULL * block_size * block_size * sizeof(double);
            matrix_multiply_tiled<<<grid, block, shared_bytes>>>(
                device_a, device_b, device_c, n, n, 0);
            check_cuda(cudaGetLastError(), "matrix_multiply_tiled launch");
        } else {
            matrix_multiply_global<<<grid, block>>>(device_a, device_b, device_c,
                                                    n);
            check_cuda(cudaGetLastError(), "matrix_multiply_global launch");
        }
        check_cuda(cudaMemcpy(c, device_c, matrix_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(C)");
    } catch (...) {
        if (device_a != nullptr) cudaFree(device_a);
        if (device_b != nullptr) cudaFree(device_b);
        if (device_c != nullptr) cudaFree(device_c);
        throw;
    }

    check_cuda(cudaFree(device_a), "cudaFree(d_A)");
    check_cuda(cudaFree(device_b), "cudaFree(d_B)");
    check_cuda(cudaFree(device_c), "cudaFree(d_C)");
    return true;
}
}  // namespace

bool cuda_backend_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void cuda_backend_device_info(std::string& name, int& compute_major,
                             int& compute_minor) {
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    name = properties.name;
    compute_major = properties.major;
    compute_minor = properties.minor;
}

bool matmul_cuda_rows(const double* a, const double* b, double* c,
                      int n, int gpu_rows, int tile_size) {
    if (!cuda_backend_available() || gpu_rows <= 0) {
        return false;
    }
    if (tile_size != 16 && tile_size != 32) {
        throw std::invalid_argument("CUDA tile size must be 16 or 32");
    }

    const size_t a_bytes = static_cast<size_t>(gpu_rows) * n * sizeof(double);
    const size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(double);
    const size_t c_bytes = a_bytes;
    double* device_a = nullptr;
    double* device_b = nullptr;
    double* device_c = nullptr;
    cudaStream_t stream = nullptr;

    try {
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        check_cuda(cudaMalloc(&device_a, a_bytes), "cudaMalloc(d_A)");
        check_cuda(cudaMalloc(&device_b, matrix_bytes), "cudaMalloc(d_B)");
        check_cuda(cudaMalloc(&device_c, c_bytes), "cudaMalloc(d_C)");
        check_cuda(cudaMemcpyAsync(device_a, a, a_bytes, cudaMemcpyHostToDevice,
                                   stream), "cudaMemcpyAsync(A)");
        check_cuda(cudaMemcpyAsync(device_b, b, matrix_bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync(B)");

        const dim3 block(static_cast<unsigned>(tile_size),
                         static_cast<unsigned>(tile_size));
        const dim3 grid((n + tile_size - 1) / tile_size,
                        (gpu_rows + tile_size - 1) / tile_size);
        const size_t shared_bytes =
            2ULL * tile_size * tile_size * sizeof(double);
        matrix_multiply_tiled<<<grid, block, shared_bytes, stream>>>(
            device_a, device_b, device_c, n, gpu_rows, 0);
        check_cuda(cudaGetLastError(), "matrix_multiply_tiled launch");
        check_cuda(cudaMemcpyAsync(c, device_c, c_bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(C)");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    } catch (...) {
        if (stream != nullptr) cudaStreamDestroy(stream);
        if (device_a != nullptr) cudaFree(device_a);
        if (device_b != nullptr) cudaFree(device_b);
        if (device_c != nullptr) cudaFree(device_c);
        throw;
    }

    cudaStreamDestroy(stream);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    return true;
}

bool matmul_cuda_global(const double* a, const double* b, double* c, int n,
                        int block_size) {
    return matmul_cuda_full(a, b, c, n, block_size, false);
}

bool matmul_cuda_shared(const double* a, const double* b, double* c, int n,
                        int tile_size) {
    return matmul_cuda_full(a, b, c, n, tile_size, true);
}
