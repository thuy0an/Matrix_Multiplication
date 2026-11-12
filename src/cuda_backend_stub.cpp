#include "cuda_backend.h"

bool cuda_backend_available() {
    return false;
}

void cuda_backend_device_info(std::string&, int&, int&) {}

bool matmul_cuda_rows(const double*, const double*, double*, int, int, int) {
    return false;
}

bool matmul_cuda_global(const double*, const double*, double*, int, int) {
    return false;
}

bool matmul_cuda_shared(const double*, const double*, double*, int, int) {
    return false;
}
