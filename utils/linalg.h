#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <tuple>
#include <iostream>

namespace linalg {
/**
 * @brief A class for representing vectors.
 *
 * This class provides a simple implementation of a vector with basic
 * arithmetic operations and methods for computing norms and normalizing
 * vectors.
 *
 */
template <typename T = double>
class Vector {
   private:
    std::vector<T> data_;

   public:
    // Constructor
    Vector() = delete;

    /*
     * Constructor for a vector of given size.
     * @param size Size of the vector.
     * @throws std::invalid_argument if size is zero.
     */
    Vector(size_t size);

    /*
     * Constructor for a vector of given size and initial value.
     * @param size Size of the vector.
     * @param value Initial value for all elements.
     * @throws std::invalid_argument if size is zero.
     */
    Vector(size_t size, const T &value);

    /*
     * Constructor for a vector from a std::vector.
     * @param vec std::vector to initialize from.
     */
    Vector(const std::vector<T> &vec);

    /*
     * Copy constructor for a vector.
     * @param vec Vector to copy from.
     */
    Vector(const Vector<T> &vec);

    // Destructor
    ~Vector() = default;

    // Accessors
    /*
     * Accessor for the vector element at given index.
     * @param index Index of the element to access.
     * @return Reference to the element at the given index.
     * @throws std::out_of_range if index is out of bounds.
     */
    T &operator[](size_t index);
    const T &operator[](size_t index) const;

    // Operators
    Vector<T> &operator=(const Vector<T> &vec);
    Vector<T> operator+(const Vector<T> &vec) const;
    Vector<T> operator-(const Vector<T> &vec) const;
    Vector<T> operator*(const T scalar) const;
    Vector<T> operator/(const T scalar) const;
    Vector<T> operator*(
        const Vector<T> &vec) const;  // Element-wise multiplication
    Vector<T> operator/(const Vector<T> &vec) const;  // Element-wise division
    Vector<T> &operator*=(const Vector<T> &vec);
    Vector<T> &operator/=(const Vector<T> &vec);
    Vector<T> &operator+=(const Vector<T> &vec);
    Vector<T> &operator-=(const Vector<T> &vec);
    Vector<T> &operator*=(const T scalar);
    Vector<T> &operator/=(const T scalar);
    bool operator==(const Vector<T> &vec) const;
    bool operator!=(const Vector<T> &vec) const;

    // Iterators
    using iterator = std::vector<T>::iterator;
    using const_iterator = std::vector<T>::const_iterator;
    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;

    // Methods

    /*
     * Returns the size of the vector.
     * @return Size of the vector.
     */
    size_t size() const;

    /*
     * Check if vector is empty.
     * @return True if the vector is empty, false otherwise.
     */
    bool empty() const;

    // Accessor for the underlying data.
    void *data();
    const void *data() const;

    /*
     * Computes the dot product of this vector with another vector.
     * Uses std::enable_if to enable this function only for double type.
     * @param vec Vector to compute the dot product with.
     * @return Dot product of the two vectors.
     * @throws std::invalid_argument if the vectors are not of the same size.
     */
    template <typename U = T>
    typename std::enable_if<std::is_same<U, double>::value, double>::type dot(
        const Vector<U> &vec) const;

    /*
     * Computes the dot product of this vector with another vector.
     * @param vec Vector to compute the dot product with.
     * @return Dot product of the two vectors.
     * @throws std::invalid_argument if the vectors are not of the same
     * size.
     */
    T dot(const Vector<T> &vec) const;

    /*
     * Computes the norm (magnitude) of the vector.
     * @return Norm of the vector.
     */
    T norm(int p = 2) const;

    /*
     * Normalizes the vector to unit length.
     * @throws std::runtime_error if the vector is zero-length.
     */
    void normalize(int p = 2);

    /*
     * Computes the minimum value in the vector.
     * @return Minimum value in the vector.
     */
    T min() const;

    /*
     * Computes the maximum value in the vector.
     * @return Maximum value in the vector.
     */
    T max() const;

    /*
     * Computes the total sum of the vector.
     * @return Total sum of the vector.
     */
    T sum() const;

    /*
     * Computes the mean value of the vector.
     * @return Mean value of the vector.
     */
    T mean() const;

    // printing
    /*
     * Prints the vector to ostream.
     * @param os Output stream to print to (default is std::cout).
     */
    void print(std::ostream &os = std::cout) const;
};

template <typename T = double>
/**
 * @brief A class for representing matrices.
 *
 * This class provides a simple implementation of a matrix with basic
 * arithmetic operations and methods for computing norms and normalizing
 * matrices.
 *
 */
class Matrix {
   private:
    size_t rows_;
    size_t cols_;
    std::vector<std::vector<T>> data_;

   public:
    // Constructor
    Matrix() = delete;

    /*
     * Constructor for a square matrix of given size.
     * @param size Size of the matrix (number of rows and columns).
     * @throws std::invalid_argument if size is zero.
     */
    Matrix(size_t size);

    /*
     * Constructor for a matrix of given size.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @throws std::invalid_argument if rows or cols is zero.
     */
    Matrix(size_t rows, size_t cols);

    /*
     * Constructor for a matrix of given size and initial value.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param value Initial value for all elements.
     * @throws std::invalid_argument if rows or cols is zero.
     */
    Matrix(size_t rows, size_t cols, const T &value);

    /*
     * Constructor for a matrix from a std::vector of std::vector.
     * @param mat std::vector of std::vector to initialize from.
     * @throws std::invalid_argument if the inner vectors are not of the same
     * size.
     */
    Matrix(const std::vector<std::vector<T>> &mat);

    /*
     * Constructor for a matrix from a vector.
     * @param vec Vector to initialize from.
     * @throws std::invalid_argument if the vector is empty.
     */
    Matrix(const Vector<T> &vec);

    /*
     * Constructor for a matrix from a vector of vectors.
     * @param vec Vector of vectors to initialize from.
     * @throws std::invalid_argument if the inner vectors are not of the same
     * size.
     */
    Matrix(const Vector<Vector<T>> &vec);

    /*
     * Copy constructor for a matrix.
     * @param mat Matrix to copy from.
     */
    Matrix(const Matrix<T> &mat);

    // Destructor
    ~Matrix() = default;

    // Accessors
    /*
     * Accessor for the matrix element at given row and column.
     * @param row Row index of the element to access.
     * @param col Column index of the element to access.
     * @return Reference to the element at the given row and column.
     * @throws std::out_of_range if row or col is out of bounds.
     */
    T &operator()(size_t row, size_t col);
    const T &operator()(size_t row, size_t col) const;
    size_t rows() const;
    size_t cols() const;

    // Operators
    Matrix<T> &operator=(const Matrix<T> &mat);
    Matrix<T> operator+(const Matrix<T> &mat) const;
    Matrix<T> operator-(const Matrix<T> &mat) const;
    Matrix<T> operator*(const T scalar) const;
    Matrix<T> operator/(const T scalar) const;
    Matrix<T> &operator+=(const Matrix<T> &mat);
    Matrix<T> &operator-=(const Matrix<T> &mat);
    Matrix<T> &operator*=(const T scalar);
    Matrix<T> &operator/=(const T scalar);
    bool operator==(const Matrix<T> &mat) const;
    bool operator!=(const Matrix<T> &mat) const;
    Matrix<T> operator*(const Matrix<T> &mat) const;
    Vector<T> operator*(const Vector<T> &vec) const;

    // Methods
    /*
     * Transposes the matrix in place.
     */
    void transpose();

    /*
     * Computes the transpose of the matrix.
     * @return Transposed matrix.
     */
    Matrix<T> transposed() const;

    /*
     * Computes the inverse of the matrix.
     * @return Inverse of the matrix.
     * @throws std::runtime_error if the matrix is not invertible.
     */
    Matrix<T> inverse() const;

    /*
     * Computes the spectral norm of the matrix.
     * @return Spectral norm of the matrix.
     */
    T spectral_norm() const;

    /*
     * Computes the Frobenius norm of the matrix.
     * @return Frobenius norm of the matrix.
     */
    T frobenius_norm() const;

    /*
     * Computes the condition number of the matrix.
     * @return Condition number of the matrix.
     */
    T condition_number() const;

    /*
     * Computes the determinant of the matrix.
     * @return Determinant of the matrix.
     */
    T determinant() const;

    /*
     * Computes the trace of the matrix.
     * @return Trace of the matrix.
     */
    T trace() const;

    /*
     * Computes the eigen decomposition of the matrix.
     * @return Eigenvalues and eigenvectors of the matrix.
     */
    std::tuple<Vector<T>, Matrix<T>> eigen_decomposition() const;

    /*
     * Computes the QR decomposition of the matrix.
     * @return Q and R matrices of the QR decomposition.
     */
    std::tuple<Matrix<T>, Matrix<T>> qr_decomposition() const;

    /*
     * Computes the LU decomposition of the matrix.
     * @return L and U matrices of the LU decomposition.
     */
    std::tuple<Matrix<T>, Matrix<T>> lu_decomposition() const;

    /*
     * Computes the Cholesky decomposition of the matrix.
     * @return L matrix of the Cholesky decomposition.
     */
    Matrix<T> cholesky_decomposition() const;

    /*
     * Computes the singular value decomposition of the matrix.
     * @return SVD of the matrix.
     */
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd() const;

   private:
    /*
     * Helper function to perform Gauss-Jordan elimination. Assumes the matrix
     * is square.
     * @return Inverse of the matrix.
     * @throws std::runtime_error if the matrix is not invertible.
     */
    Matrix<T> _Gauss_Jordan_elimination() const;
};

// enable arbitrary scalar multiplication
template <typename T>
Vector<T> operator*(const T scalar, const Vector<T> &vec);

template <typename T>
Matrix<T> operator*(const T scalar, const Matrix<T> &mat);
};  // namespace linalg

// Type definitions
typedef linalg::Vector<double> Vec;
typedef linalg::Matrix<double> Mat;

#endif  // TYPES_H