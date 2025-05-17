#include <omp.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <immintrin.h>
#include "linalg.h"

namespace linalg {
template <typename T>
Vector<T>::Vector(size_t size) : data_(size) {
    if (size == 0) {
        throw std::invalid_argument("Size must be greater than zero.");
    }
}

template <typename T>
Vector<T>::Vector(size_t size, const T &value) : data_(size, value) {
    if (size == 0) {
        throw std::invalid_argument("Size must be greater than zero.");
    }
}

template <typename T>
Vector<T>::Vector(const std::vector<T> &vec) : data_(vec) {}

template <typename T>
Vector<T>::Vector(const Vector<T> &vec) : data_(vec.data_) {}

// Accessors
template <typename T>
T &Vector<T>::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("Index out of bounds.");
    }
    return data_[index];
}

template <typename T>
const T &Vector<T>::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Index out of bounds.");
    }
    return data_[index];
}

template <typename T>
Vector<T> &Vector<T>::operator=(const Vector<T> &vec) {
    if (this != &vec) {
        data_ = vec.data_;
    }
    return *this;
}

template <typename T>
Vector<T> Vector<T>::operator+(const Vector<T> &vec) const {
    if (size() != vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data_[i] + vec[i];
    }
    return result;
}

template <typename T>
Vector<T> Vector<T>::operator-(const Vector<T> &vec) const {
    if (size() != vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data_[i] - vec[i];
    }
    return result;
}

template <typename T>
Vector<T> Vector<T>::operator*(const T scalar) const {
    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data_[i] * scalar;
    }
    return result;
}

template <typename T>
Vector<T> Vector<T>::operator/(const T scalar) const {
    if (scalar == 0) {
        throw std::invalid_argument("Division by zero.");
    }

    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data_[i] / scalar;
    }
    return result;
}

template <typename T>
Vector<T> &Vector<T>::operator+=(const Vector<T> &vec) {
    if (size() != vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    for (size_t i = 0; i < size(); ++i) {
        data_[i] += vec[i];
    }
    return *this;
}

template <typename T>
Vector<T> &Vector<T>::operator-=(const Vector<T> &vec) {
    if (size() != vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    for (size_t i = 0; i < size(); ++i) {
        data_[i] -= vec[i];
    }
    return *this;
}

template <typename T>
Vector<T> &Vector<T>::operator*=(const T scalar) {
    for (size_t i = 0; i < size(); ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

template <typename T>
Vector<T> &Vector<T>::operator/=(const T scalar) {
    if (scalar == 0) {
        throw std::invalid_argument("Division by zero.");
    }

    for (size_t i = 0; i < size(); ++i) {
        data_[i] /= scalar;
    }
    return *this;
}

template <typename T>
Vector<T> Vector<T>::operator*(const Vector<T> &vec) const {
    if (size() != vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data_[i] * vec[i];
    }
    return result;
}

template <typename T>
Vector<T> Vector<T>::operator/(const Vector<T> &vec) const {
    if (size() != vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        if (vec[i] == 0) {
            throw std::invalid_argument("Division by zero.");
        }
        result[i] = data_[i] / vec[i];
    }
    return result;
}

template <typename T>
Vector<T> &Vector<T>::operator*=(const Vector<T> &vec) {
    if (size() != vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    for (size_t i = 0; i < size(); ++i) {
        data_[i] *= vec[i];
    }
    return *this;
}

template <typename T>
Vector<T> &Vector<T>::operator/=(const Vector<T> &vec) {
    if (size() != vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    for (size_t i = 0; i < size(); ++i) {
        if (vec[i] == 0) {
            throw std::invalid_argument("Division by zero.");
        }
        data_[i] /= vec[i];
    }
    return *this;
}

template <typename T>
bool Vector<T>::operator==(const Vector<T> &vec) const {
    if (size() != vec.size()) {
        return false;
    }

    for (size_t i = 0; i < size(); ++i) {
        if (data_[i] != vec[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool Vector<T>::operator!=(const Vector<T> &vec) const {
    return !(*this == vec);
}

template <typename T>
typename Vector<T>::iterator Vector<T>::begin() {
    return data_.begin();
}

template <typename T>
typename Vector<T>::const_iterator Vector<T>::begin() const {
    return data_.begin();
}

template <typename T>
typename Vector<T>::iterator Vector<T>::end() {
    return data_.end();
}

template <typename T>
typename Vector<T>::const_iterator Vector<T>::end() const {
    return data_.end();
}

template <typename T>
size_t Vector<T>::size() const {
    return data_.size();
}

template <typename T>
bool Vector<T>::empty() const {
    return data_.empty();
}

template <typename T>
void *Vector<T>::data() {
    return data_.data();
}

template <typename T>
const void *Vector<T>::data() const {
    return data_.data();
}

template <typename T>
template <typename U>
typename std::enable_if<std::is_same<U, double>::value, double>::type
Vector<T>::dot(const Vector<U> &vec) const {
    if (size() != vec.size()) {
        throw std::invalid_argument(
            "Vectors must be of the same size for dot product.");
    }

    const size_t size = this->size();
    const double *a = data();
    const double *b = vec.data();

    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;

    // Process in chunks of 4 doubles (256 bits / 64 bits per double)
    for (; i + 3 < size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        sum = _mm256_fmadd_pd(va, vb, sum);
    }

    // Horizontal sum of the 256-bit register
    double temp[4];
    _mm256_storeu_pd(temp, sum);
    double result = temp[0] + temp[1] + temp[2] + temp[3];

    // Handle remaining elements (size not a multiple of 4)
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

template <typename T>
T Vector<T>::dot(const Vector<T> &vec) const {
    if (size() != vec.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    T result = 0;

#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < size(); ++i) {
        result += data_[i] * vec[i];
    }

    return result;
}

template <typename T>
T Vector<T>::norm(int p) const {
    if (p < 1) {
        throw std::invalid_argument("Norm order must be greater than zero.");
    }
    T result = 0;

    for (size_t i = 0; i < size(); ++i) {
        result += std::pow(std::abs(data_[i]), p);
    }
    return std::pow(result, 1.0 / p);
}

template <typename T>
void Vector<T>::normalize(int p) {
    if (p < 1) {
        throw std::invalid_argument("Norm order must be greater than zero.");
    }

    T norm_value = norm(p);

    if (norm_value == 0) {
        throw std::runtime_error("Cannot normalize a zero-norm vector.");
    }

    for (size_t i = 0; i < size(); ++i) {
        data_[i] /= norm_value;
    }
}

template <typename T>
T Vector<T>::min() const {
    if (empty()) {
        throw std::runtime_error("Vector is empty.");
    }

    return *std::min_element(begin(), end());
}

template <typename T>
T Vector<T>::max() const {
    if (empty()) {
        throw std::runtime_error("Vector is empty.");
    }

    return *std::max_element(begin(), end());
}

template <typename T>
T Vector<T>::sum() const {
    return std::accumulate(data_.begin(), data_.end(), T{});
}

template <typename T>
T Vector<T>::mean() const {
    if (empty()) {
        throw std::runtime_error("Vector is empty.");
    }
    return sum() / static_cast<T>(data_.size());
}

template <typename T>
void Vector<T>::print(std::ostream &os) const {
    os << "[";
    for (size_t i = 0; i < size(); ++i) {
        os << data_[i];
        if (i < size() - 1) {
            os << ", ";
        }
    }
    os << "]";
}

template <typename T>
Matrix<T>::Matrix(size_t size)
    : rows_(size), cols_(size), data_(size, std::vector<T>(size)) {
    if (size == 0) {
        throw std::invalid_argument("Size must be greater than zero.");
    }
}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows, std::vector<T>(cols)) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(
            "Rows and columns must be greater than zero.");
    }
}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T &value)
    : rows_(rows), cols_(cols), data_(rows, std::vector<T>(cols, value)) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(
            "Rows and columns must be greater than zero.");
    }
}

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>> &mat)
    : rows_(mat.size()), cols_(mat[0].size()), data_(mat) {
    for (const auto &row : mat) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must be of the same size.");
        }
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &mat)
    : rows_(mat.rows_), cols_(mat.cols_), data_(mat.data_) {}

template <typename T>
Matrix<T>::Matrix(const Vector<T> &vec)
    : rows_(vec.size()), cols_(1), data_(vec.size(), std::vector<T>(1)) {
    if (vec.size() == 0) {
        throw std::invalid_argument("Vector size must be greater than zero.");
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        data_[i][0] = vec[i];
    }
}

template <typename T>
Matrix<T>::Matrix(const Vector<Vector<T>> &vec)
    : rows_(vec.size()),
      cols_(vec[0].size()),
      data_(vec.size(), std::vector<T>(vec[0].size())) {
    throw std::runtime_error(
        "Matrix constructor from Vector<Vector<T>> is not implemented.");
}

template <typename T>
T &Matrix<T>::operator()(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Index out of bounds.");
    }
    return data_[row][col];
}

template <typename T>
const T &Matrix<T>::operator()(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Index out of bounds.");
    }
    return data_[row][col];
}

template <typename T>
size_t Matrix<T>::rows() const {
    return rows_;
}

template <typename T>
size_t Matrix<T>::cols() const {
    return cols_;
}

template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &mat) {
    if (this != &mat) {
        rows_ = mat.rows_;
        cols_ = mat.cols_;
        data_ = mat.data_;
    }
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &mat) const {
    if (rows_ != mat.rows_ || cols_ != mat.cols_) {
        throw std::invalid_argument("Matrices must be of the same size.");
    }

    Matrix<T> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(i, j) = data_[i][j] + mat(i, j);
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &mat) const {
    if (rows_ != mat.rows_ || cols_ != mat.cols_) {
        throw std::invalid_argument("Matrices must be of the same size.");
    }

    Matrix<T> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(i, j) = data_[i][j] - mat(i, j);
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator/(const T scalar) const {
    if (scalar == 0) {
        throw std::invalid_argument("Division by zero.");
    }

    Matrix<T> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(i, j) = data_[i][j] / scalar;
        }
    }
    return result;
}

template <typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &mat) {
    if (rows_ != mat.rows_ || cols_ != mat.cols_) {
        throw std::invalid_argument("Matrices must be of the same size.");
    }

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] += mat(i, j);
        }
    }
    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &mat) {
    if (rows_ != mat.rows_ || cols_ != mat.cols_) {
        throw std::invalid_argument("Matrices must be of the same size.");
    }

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] -= mat(i, j);
        }
    }
    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator*=(const T scalar) {
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] *= scalar;
        }
    }
    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator/=(const T scalar) {
    if (scalar == 0) {
        throw std::invalid_argument("Division by zero.");
    }

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] /= scalar;
        }
    }
    return *this;
}

template <typename T>
bool Matrix<T>::operator==(const Matrix<T> &mat) const {
    if (rows_ != mat.rows_ || cols_ != mat.cols_) {
        return false;
    }

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (data_[i][j] != mat(i, j)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool Matrix<T>::operator!=(const Matrix<T> &mat) const {
    return !(*this == mat);
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T scalar) const {
    Matrix<T> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(i, j) = data_[i][j] * scalar;
        }
    }
    return result;
}

template <typename T>
void Matrix<T>::transpose() {
    Matrix<T> result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = data_[i][j];
        }
    }
    *this = result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &mat) const {
    if (cols_ != mat.rows_) {
        throw std::invalid_argument(
            "Matrices must be compatible for multiplication.");
    }

    Matrix<T> result(rows_, mat.cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < mat.cols_; ++j) {
            result(i, j) = 0;
            for (size_t k = 0; k < cols_; ++k) {
                result(i, j) += data_[i][k] * mat(k, j);
            }
        }
    }
    return result;
}

template <typename T>
Vector<T> Matrix<T>::operator*(const Vector<T> &vec) const {
    if (cols_ != vec.size()) {
        throw std::invalid_argument(
            "Matrix and vector sizes are incompatible.");
    }

    Vector<T> result(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        result[i] = 0;
        for (size_t j = 0; j < cols_; ++j) {
            result[i] += data_[i][j] * vec[j];
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::transposed() const {
    Matrix<T> result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = data_[i][j];
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square.");
    }

    return _Gauss_Jordan_elimination();
}

template <typename T>
Matrix<T> Matrix<T>::_Gauss_Jordan_elimination() const {
    // Create an augmented matrix [A|I]
    Matrix<T> augmented(rows_, 2 * cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            augmented.data_[i][j] = data_[i][j];
        }
        // Set the identity matrix part
        augmented.data_[i][i + cols_] = static_cast<T>(1);
    }

    // Perform Gauss-Jordan elimination
    for (size_t i = 0; i < rows_; ++i) {
        // Find pivot
        T pivot = augmented.data_[i][i];
        if (pivot == static_cast<T>(0)) {
            // Try to find non-zero pivot in the same column
            bool found = false;
            for (size_t k = i + 1; k < rows_; ++k) {
                if (augmented.data_[k][i] != static_cast<T>(0)) {
                    // Swap rows
                    std::swap(augmented.data_[i], augmented.data_[k]);
                    pivot = augmented.data_[i][i];
                    found = true;
                    break;
                }
            }

            if (!found) {
                throw std::runtime_error("Matrix is not invertible.");
            }
        }

        // Scale the pivot row to make pivot = 1
        if (pivot != static_cast<T>(1)) {
            for (size_t j = 0; j < 2 * cols_; ++j) {
                augmented.data_[i][j] /= pivot;
            }
        }

        // Eliminate other rows
        for (size_t k = 0; k < rows_; ++k) {
            if (k != i) {
                T factor = augmented.data_[k][i];
                for (size_t j = 0; j < 2 * cols_; ++j) {
                    augmented.data_[k][j] -= factor * augmented.data_[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix from the right side of the augmented
    // matrix
    Matrix<T> inverse(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            inverse.data_[i][j] = augmented.data_[i][j + cols_];
        }
    }

    return inverse;
}

template <typename T>
T Matrix<T>::spectral_norm() const {
    // ||A||_2 = max(sqrt(lambda_i))
    // where lambda_i are the eigenvalues of A^T * A.
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square.");
    }
    Matrix<T> AtA = transposed() * (*this);
    Vector<T> eigenvalues = std::get<0>(AtA.eigen_decomposition());
    T max_eigenvalue = eigenvalues.max();
    return std::sqrt(max_eigenvalue);
}

template <typename T>
T Matrix<T>::frobenius_norm() const {
    // ||A||_F = sqrt(sum(A_ij^2))
    // where A_ij is the element at row i and column j of matrix A.
    T sum = 0;
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            sum += data_[i][j] * data_[i][j];
        }
    }
    return std::sqrt(sum);
}

template <typename T>
T Matrix<T>::condition_number() const {
    // K = ||A|| * ||A^(-1)||
    // where ||A|| is the spectral norm and ||A^(-1)|| is the spectral norm
    // of the inverse of A.
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square.");
    }
    Matrix<T> inv = inverse();
    T norm_A = spectral_norm();
    T norm_A_inv = inv.spectral_norm();
    return norm_A * norm_A_inv;
}

template <typename T>
T Matrix<T>::determinant() const {
    // |A| = product of eigenvalues
    // where A is the matrix and |A| is the determinant.
    // The determinant is only defined for square matrices.
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square.");
    }

    Vector<T> eigenvalues = std::get<0>(eigen_decomposition());
    T det = 1;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        det *= eigenvalues[i];
    }
    return det;
}

template <typename T>
T Matrix<T>::trace() const {
    // Tr(A) = sum(A_ii)
    // where A_ii is the element at row i and column i of matrix A.

    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square.");
    }

    T sum = 0;
    for (size_t i = 0; i < rows_; ++i) {
        sum += data_[i][i];
    }
    return sum;
}

template <typename T>
std::tuple<Vector<T>, Matrix<T>> Matrix<T>::eigen_decomposition() const {
    // Placeholder for eigen decomposition
    // This is a complex operation and would typically require
    // specialized libraries or algorithms.
    throw std::runtime_error("Eigen decomposition not implemented.");
}

template <typename T>
std::tuple<Matrix<T>, Matrix<T>> Matrix<T>::qr_decomposition() const {
    // Placeholder for QR decomposition
    // This is a complex operation and would typically require
    // specialized libraries or algorithms.
    throw std::runtime_error("QR decomposition not implemented.");
}

template <typename T>
std::tuple<Matrix<T>, Matrix<T>> Matrix<T>::lu_decomposition() const {
    // Placeholder for LU decomposition
    // This is a complex operation and would typically require
    // specialized libraries or algorithms.
    throw std::runtime_error("LU decomposition not implemented.");
}

template <typename T>
Matrix<T> Matrix<T>::cholesky_decomposition() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square.");
    }

    // Placeholder for Cholesky decomposition
    // This is a complex operation and would typically require
    // specialized libraries or algorithms.
    throw std::runtime_error("Cholesky decomposition not implemented.");
}

template <typename T>
std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> Matrix<T>::svd() const {
    // For matrix A of size m x n,
    // U is m x m, S is m x n, and V^T is n x n.
    // A = U * S * V^T

    throw std::runtime_error("SVD decomposition not implemented.");
}

// enable arbitrary scalar multiplication
template <typename T>
Vector<T> operator*(const T scalar, const Vector<T> &vec) {
    return vec * scalar;
}

template <typename T>
Matrix<T> operator*(const T scalar, const Matrix<T> &mat) {
    return mat * scalar;
}

}  // namespace linalg

// Explicit instantiation for double type
template class linalg::Vector<double>;
template class linalg::Matrix<double>;