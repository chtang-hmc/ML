#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <tuple>

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
    size_t size() const;

    // Operators
    Vector<T> &operator=(const Vector<T> &vec);
    Vector<T> operator+(const Vector<T> &vec) const;
    Vector<T> operator-(const Vector<T> &vec) const;
    Vector<T> operator*(const T scalar) const;
    Vector<T> operator/(const T scalar) const;
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
     * Computes the dot product of this vector with another vector.
     * @param vec Vector to compute the dot product with.
     * @return Dot product of the two vectors.
     * @throws std::invalid_argument if the vectors are not of the same size.
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
     * Computes the singular value decomposition of the matrix.
     * @return SVD of the matrix.
     */
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd() const;
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