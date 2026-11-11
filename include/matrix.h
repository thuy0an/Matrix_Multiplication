#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string>
#include <iostream>

class Matrix {
private:
    int n_;
    double* data_;

public:
    Matrix();
    explicit Matrix(int n, bool zero_init = true);
    ~Matrix();

    // Copy and Move semantics
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    void resize(int n, bool zero_init = true);
    void zero();
    void fill_random(double min_val = 0.5, double max_val = 5.0, unsigned int seed = 42);

    inline int size() const { return n_; }
    inline double* data() { return data_; }
    inline const double* data() const { return data_; }

    inline double& at(int i, int j) { return data_[i * n_ + j]; }
    inline double at(int i, int j) const { return data_[i * n_ + j]; }

    // Kiểm định sai số tương đối theo chuẩn Frobenius: ||C_par - C_ref||_F / ||C_ref||_F
    double relative_error(const Matrix& ref) const;
    double max_diff(const Matrix& ref) const;
};

#endif // MATRIX_H
