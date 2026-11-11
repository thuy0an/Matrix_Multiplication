#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

bool cuda_backend_available();

bool matmul_cuda_rows(const double* a, const double* b, double* c,
                      int n, int gpu_rows, int tile_size);

#endif
