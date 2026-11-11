#include "cuda_backend.h"

bool cuda_backend_available() {
    return false;
}

bool matmul_cuda_rows(const double*, const double*, double*, int, int, int) {
    return false;
}
