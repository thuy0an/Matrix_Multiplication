#include "matrix.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(size, align) _aligned_malloc((size), (align))
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
static void* aligned_alloc_wrapper(size_t size, size_t align) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, align, size) != 0) return nullptr;
    return ptr;
}
#define ALIGNED_ALLOC(size, align) aligned_alloc_wrapper((size), (align))
#define ALIGNED_FREE(ptr) free(ptr)
#endif

Matrix::Matrix() : n_(0), data_(nullptr) {}

Matrix::Matrix(int n, bool zero_init) : n_(n), data_(nullptr) {
    if (n > 0) {
        size_t bytes = static_cast<size_t>(n) * n * sizeof(double);
        data_ = static_cast<double*>(ALIGNED_ALLOC(bytes, 64));
        if (zero_init && data_) {
            std::memset(data_, 0, bytes);
        }
    }
}

Matrix::~Matrix() {
    if (data_) {
        ALIGNED_FREE(data_);
        data_ = nullptr;
    }
}

Matrix::Matrix(const Matrix& other) : n_(other.n_), data_(nullptr) {
    if (n_ > 0 && other.data_) {
        size_t bytes = static_cast<size_t>(n_) * n_ * sizeof(double);
        data_ = static_cast<double*>(ALIGNED_ALLOC(bytes, 64));
        if (data_) {
            std::memcpy(data_, other.data_, bytes);
        }
    }
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        if (data_) {
            ALIGNED_FREE(data_);
            data_ = nullptr;
        }
        n_ = other.n_;
        if (n_ > 0 && other.data_) {
            size_t bytes = static_cast<size_t>(n_) * n_ * sizeof(double);
            data_ = static_cast<double*>(ALIGNED_ALLOC(bytes, 64));
            if (data_) {
                std::memcpy(data_, other.data_, bytes);
            }
        }
    }
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept : n_(other.n_), data_(other.data_) {
    other.n_ = 0;
    other.data_ = nullptr;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        if (data_) {
            ALIGNED_FREE(data_);
        }
        n_ = other.n_;
        data_ = other.data_;
        other.n_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

void Matrix::resize(int n, bool zero_init) {
    if (n == n_ && data_) {
        if (zero_init) {
            std::memset(data_, 0, static_cast<size_t>(n) * n * sizeof(double));
        }
        return;
    }
    if (data_) {
        ALIGNED_FREE(data_);
        data_ = nullptr;
    }
    n_ = n;
    if (n_ > 0) {
        size_t bytes = static_cast<size_t>(n_) * n_ * sizeof(double);
        data_ = static_cast<double*>(ALIGNED_ALLOC(bytes, 64));
        if (zero_init && data_) {
            std::memset(data_, 0, bytes);
        }
    }
}

void Matrix::zero() {
    if (data_ && n_ > 0) {
        std::memset(data_, 0, static_cast<size_t>(n_) * n_ * sizeof(double));
    }
}

void Matrix::fill_random(double min_val, double max_val, unsigned int seed) {
    if (!data_ || n_ <= 0) return;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    size_t total = static_cast<size_t>(n_) * n_;
    for (size_t i = 0; i < total; ++i) {
        data_[i] = dist(gen);
    }
}

double Matrix::max_diff(const Matrix& ref) const {
    if (n_ != ref.n_ || !data_ || !ref.data_) return 1e9;
    size_t total = static_cast<size_t>(n_) * n_;
    double max_d = 0.0;
    for (size_t i = 0; i < total; ++i) {
        double d = std::abs(data_[i] - ref.data_[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

double Matrix::relative_error(const Matrix& ref) const {
    if (n_ != ref.n_ || !data_ || !ref.data_) return 1e9;
    size_t total = static_cast<size_t>(n_) * n_;
    double diff_sq = 0.0;
    double ref_sq = 0.0;
    for (size_t i = 0; i < total; ++i) {
        double d = data_[i] - ref.data_[i];
        diff_sq += d * d;
        ref_sq += ref.data_[i] * ref.data_[i];
    }
    double norm_diff = std::sqrt(diff_sq);
    double norm_ref = std::sqrt(ref_sq);
    if (norm_ref == 0.0) return (norm_diff == 0.0) ? 0.0 : 1e9;
    return norm_diff / norm_ref;
}
