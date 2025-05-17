#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "types.h"

// Vector Class Tests
TEST(VectorTest, Constructor) {
    // Test size constructor
    linalg::Vector<double> v1(5);
    EXPECT_EQ(v1.size(), 5);

    // Test size and value constructor
    linalg::Vector<double> v2(3, 2.5);
    EXPECT_EQ(v2.size(), 3);
    for (size_t i = 0; i < v2.size(); ++i) {
        EXPECT_DOUBLE_EQ(v2[i], 2.5);
    }

    // Test std::vector constructor
    std::vector<double> stdVec = {1.0, 2.0, 3.0};
    linalg::Vector<double> v3(stdVec);
    EXPECT_EQ(v3.size(), 3);
    EXPECT_DOUBLE_EQ(v3[0], 1.0);
    EXPECT_DOUBLE_EQ(v3[1], 2.0);
    EXPECT_DOUBLE_EQ(v3[2], 3.0);

    // Test copy constructor
    linalg::Vector<double> v4(v3);
    EXPECT_EQ(v4.size(), 3);
    EXPECT_DOUBLE_EQ(v4[0], 1.0);
    EXPECT_DOUBLE_EQ(v4[1], 2.0);
    EXPECT_DOUBLE_EQ(v4[2], 3.0);

    // Test invalid size
    EXPECT_THROW(linalg::Vector<double>(0), std::invalid_argument);
}

TEST(VectorTest, AccessOperators) {
    linalg::Vector<double> v({1.0, 2.0, 3.0});

    // Test operator[]
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);

    // Test operator[] const
    const linalg::Vector<double> constV(v);
    EXPECT_DOUBLE_EQ(constV[0], 1.0);
    EXPECT_DOUBLE_EQ(constV[1], 2.0);
    EXPECT_DOUBLE_EQ(constV[2], 3.0);

    // Test operator[] out of bounds
    EXPECT_THROW(v[3], std::out_of_range);
    EXPECT_THROW(constV[3], std::out_of_range);

    // Test operator[] modification
    v[0] = 5.0;
    EXPECT_DOUBLE_EQ(v[0], 5.0);
}

TEST(VectorTest, AssignmentOperator) {
    linalg::Vector<double> v1({1.0, 2.0, 3.0});
    linalg::Vector<double> v2(5, 0.0);

    v2 = v1;
    EXPECT_EQ(v2.size(), 3);
    EXPECT_DOUBLE_EQ(v2[0], 1.0);
    EXPECT_DOUBLE_EQ(v2[1], 2.0);
    EXPECT_DOUBLE_EQ(v2[2], 3.0);

    // Self-assignment
    v1 = v1;
    EXPECT_EQ(v1.size(), 3);
    EXPECT_DOUBLE_EQ(v1[0], 1.0);
    EXPECT_DOUBLE_EQ(v1[1], 2.0);
    EXPECT_DOUBLE_EQ(v1[2], 3.0);
}

TEST(VectorTest, ArithmeticOperators) {
    linalg::Vector<double> v1({1.0, 2.0, 3.0});
    linalg::Vector<double> v2({4.0, 5.0, 6.0});

    // Test operator+
    linalg::Vector<double> v3 = v1 + v2;
    EXPECT_EQ(v3.size(), 3);
    EXPECT_DOUBLE_EQ(v3[0], 5.0);
    EXPECT_DOUBLE_EQ(v3[1], 7.0);
    EXPECT_DOUBLE_EQ(v3[2], 9.0);

    // Test operator-
    linalg::Vector<double> v4 = v2 - v1;
    EXPECT_EQ(v4.size(), 3);
    EXPECT_DOUBLE_EQ(v4[0], 3.0);
    EXPECT_DOUBLE_EQ(v4[1], 3.0);
    EXPECT_DOUBLE_EQ(v4[2], 3.0);

    // Test operator*
    linalg::Vector<double> v5 = v1 * 2.0;
    EXPECT_EQ(v5.size(), 3);
    EXPECT_DOUBLE_EQ(v5[0], 2.0);
    EXPECT_DOUBLE_EQ(v5[1], 4.0);
    EXPECT_DOUBLE_EQ(v5[2], 6.0);

    // Test operator/
    linalg::Vector<double> v6 = v1 / 2.0;
    EXPECT_EQ(v6.size(), 3);
    EXPECT_DOUBLE_EQ(v6[0], 0.5);
    EXPECT_DOUBLE_EQ(v6[1], 1.0);
    EXPECT_DOUBLE_EQ(v6[2], 1.5);

    // Test operator+=
    linalg::Vector<double> v7 = v1;
    v7 += v2;
    EXPECT_EQ(v7.size(), 3);
    EXPECT_DOUBLE_EQ(v7[0], 5.0);
    EXPECT_DOUBLE_EQ(v7[1], 7.0);
    EXPECT_DOUBLE_EQ(v7[2], 9.0);

    // Test operator-=
    linalg::Vector<double> v8 = v2;
    v8 -= v1;
    EXPECT_EQ(v8.size(), 3);
    EXPECT_DOUBLE_EQ(v8[0], 3.0);
    EXPECT_DOUBLE_EQ(v8[1], 3.0);
    EXPECT_DOUBLE_EQ(v8[2], 3.0);

    // Test operator*=
    linalg::Vector<double> v9 = v1;
    v9 *= 2.0;
    EXPECT_EQ(v9.size(), 3);
    EXPECT_DOUBLE_EQ(v9[0], 2.0);
    EXPECT_DOUBLE_EQ(v9[1], 4.0);
    EXPECT_DOUBLE_EQ(v9[2], 6.0);

    // Test operator/=
    linalg::Vector<double> v10 = v1;
    v10 /= 2.0;
    EXPECT_EQ(v10.size(), 3);
    EXPECT_DOUBLE_EQ(v10[0], 0.5);
    EXPECT_DOUBLE_EQ(v10[1], 1.0);
    EXPECT_DOUBLE_EQ(v10[2], 1.5);

    // Test mismatched size operations
    linalg::Vector<double> v11(2, 1.0);
    EXPECT_THROW(v1 + v11, std::invalid_argument);
    EXPECT_THROW(v1 - v11, std::invalid_argument);
    EXPECT_THROW(v1 += v11, std::invalid_argument);
    EXPECT_THROW(v1 -= v11, std::invalid_argument);

    // Test division by zero
    EXPECT_THROW(v1 / 0.0, std::invalid_argument);
    EXPECT_THROW(v1 /= 0.0, std::invalid_argument);
}

TEST(VectorTest, ComparisonOperators) {
    linalg::Vector<double> v1({1.0, 2.0, 3.0});
    linalg::Vector<double> v2({1.0, 2.0, 3.0});
    linalg::Vector<double> v3({3.0, 2.0, 1.0});
    std::vector<double> vec4 = {1.0, 2.0};
    linalg::Vector<double> v4(vec4);

    // Test operator==
    EXPECT_TRUE(v1 == v2);
    EXPECT_FALSE(v1 == v3);
    EXPECT_FALSE(v1 == v4);

    // Test operator!=
    EXPECT_FALSE(v1 != v2);
    EXPECT_TRUE(v1 != v3);
    EXPECT_TRUE(v1 != v4);
}

TEST(VectorTest, DotProduct) {
    linalg::Vector<double> v1({1.0, 2.0, 3.0});
    linalg::Vector<double> v2({4.0, 5.0, 6.0});

    // Test dot product
    EXPECT_DOUBLE_EQ(v1.dot(v2), 32.0);  // 1*4 + 2*5 + 3*6 = 32

    // Test mismatched size
    linalg::Vector<double> v3(2, 1.0);
    EXPECT_THROW(v1.dot(v3), std::invalid_argument);
}

TEST(VectorTest, Norm) {
    std::vector<double> vec1 = {3.0, 4.0};
    linalg::Vector<double> v1(vec1);
    linalg::Vector<double> v2({1.0, 2.0, 2.0});

    // Test L2 norm (default)
    EXPECT_DOUBLE_EQ(v1.norm(), 5.0);  // sqrt(3^2 + 4^2) = 5
    EXPECT_DOUBLE_EQ(v2.norm(), 3.0);  // sqrt(1^2 + 2^2 + 2^2) = 3

    // Test L1 norm
    EXPECT_DOUBLE_EQ(v1.norm(1), 7.0);  // |3| + |4| = 7
    EXPECT_DOUBLE_EQ(v2.norm(1), 5.0);  // |1| + |2| + |2| = 5

    // Test Lp norm (p=3)
    double expected = std::pow(
        std::pow(1.0, 3) + std::pow(2.0, 3) + std::pow(2.0, 3), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(v2.norm(3), expected);

    // Test invalid p
    EXPECT_THROW(v1.norm(-1), std::invalid_argument);
    EXPECT_THROW(v1.norm(0), std::invalid_argument);
}

TEST(VectorTest, Normalize) {
    std::vector<double> vec1 = {3.0, 4.0};
    linalg::Vector<double> v1(vec1);
    v1.normalize();
    EXPECT_DOUBLE_EQ(v1[0], 0.6);  // 3/5
    EXPECT_DOUBLE_EQ(v1[1], 0.8);  // 4/5
    EXPECT_DOUBLE_EQ(v1.norm(), 1.0);

    linalg::Vector<double> v2({1.0, 2.0, 2.0});
    v2.normalize();
    EXPECT_DOUBLE_EQ(v2[0], 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(v2[1], 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(v2[2], 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(v2.norm(), 1.0);

    // Test L1 normalization
    std::vector<double> vec3 = {3.0, 4.0};
    linalg::Vector<double> v3(vec3);
    v3.normalize(1);
    EXPECT_DOUBLE_EQ(v3[0], 3.0 / 7.0);
    EXPECT_DOUBLE_EQ(v3[1], 4.0 / 7.0);
    EXPECT_DOUBLE_EQ(v3.norm(1), 1.0);

    // Test zero vector
    linalg::Vector<double> v4(3, 0.0);
    EXPECT_THROW(v4.normalize(), std::runtime_error);
}

// Matrix Class Tests
TEST(MatrixTest, Constructor) {
    // Test size constructor
    linalg::Matrix<double> m1(3);
    EXPECT_EQ(m1.rows(), 3);
    EXPECT_EQ(m1.cols(), 3);

    // Test rows and cols constructor
    linalg::Matrix<double> m2(2, 3);
    EXPECT_EQ(m2.rows(), 2);
    EXPECT_EQ(m2.cols(), 3);

    // Test rows, cols, and value constructor
    linalg::Matrix<double> m3(2, 3, 1.5);
    EXPECT_EQ(m3.rows(), 2);
    EXPECT_EQ(m3.cols(), 3);
    double entry;
    for (size_t i = 0; i < m3.rows(); ++i) {
        for (size_t j = 0; j < m3.cols(); ++j) {
            entry = m3[i, j];
            EXPECT_DOUBLE_EQ(entry, 1.5);
        }
    }

    // Test std::vector constructor
    std::vector<std::vector<double>> stdMat = {
        {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    linalg::Matrix<double> m4(stdMat);
    EXPECT_EQ(m4.rows(), 3);
    EXPECT_EQ(m4.cols(), 2);
    EXPECT_DOUBLE_EQ(m4[0, 0], 1.0);
    EXPECT_DOUBLE_EQ(m4[0, 1], 2.0);
    EXPECT_DOUBLE_EQ(m4[1, 0], 3.0);
    EXPECT_DOUBLE_EQ(m4[1, 1], 4.0);
    EXPECT_DOUBLE_EQ(m4[2, 0], 5.0);
    EXPECT_DOUBLE_EQ(m4[2, 1], 6.0);

    // Test copy constructor
    linalg::Matrix<double> m5(m4);
    EXPECT_EQ(m5.rows(), 3);
    EXPECT_EQ(m5.cols(), 2);
    EXPECT_DOUBLE_EQ(m5[0, 0], 1.0);
    EXPECT_DOUBLE_EQ(m5[2, 1], 6.0);

    // Test invalid dimensions
    EXPECT_THROW(linalg::Matrix<double>(0), std::invalid_argument);
    EXPECT_THROW(linalg::Matrix<double>(0, 2), std::invalid_argument);
    EXPECT_THROW(linalg::Matrix<double>(2, 0), std::invalid_argument);

    // Test inconsistent inner vector sizes
    std::vector<std::vector<double>> inconsistentMat = {{1.0, 2.0}, {3.0}};
    EXPECT_THROW(linalg::Matrix<double>(inconsistentMat),
                 std::invalid_argument);
}

TEST(MatrixTest, AccessOperators) {
    linalg::Matrix<double> m({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});

    // Test operator[]
    EXPECT_DOUBLE_EQ(m[0, 0], 1.0);
    EXPECT_DOUBLE_EQ(m[0, 1], 2.0);
    EXPECT_DOUBLE_EQ(m[0, 2], 3.0);
    EXPECT_DOUBLE_EQ(m[1, 0], 4.0);
    EXPECT_DOUBLE_EQ(m[1, 1], 5.0);
    EXPECT_DOUBLE_EQ(m[1, 2], 6.0);

    // Test operator[] const
    const linalg::Matrix<double> constM(m);
    EXPECT_DOUBLE_EQ(constM[0, 0], 1.0);
    EXPECT_DOUBLE_EQ(constM[1, 2], 6.0);

    // Test operator[] out of bounds
    EXPECT_THROW(m[2, 0], std::out_of_range);
    EXPECT_THROW(m[0, 3], std::out_of_range);
    EXPECT_THROW(constM[2, 0], std::out_of_range);

    // Test operator[] modification
    m[0, 0] = 10.0;
    EXPECT_DOUBLE_EQ(m[0, 0], 10.0);
}

TEST(MatrixTest, AssignmentOperator) {
    linalg::Matrix<double> m1({{1.0, 2.0}, {3.0, 4.0}});
    linalg::Matrix<double> m2(3, 3, 0.0);

    m2 = m1;
    EXPECT_EQ(m2.rows(), 2);
    EXPECT_EQ(m2.cols(), 2);
    EXPECT_DOUBLE_EQ(m2[0, 0], 1.0);
    EXPECT_DOUBLE_EQ(m2[0, 1], 2.0);
    EXPECT_DOUBLE_EQ(m2[1, 0], 3.0);
    EXPECT_DOUBLE_EQ(m2[1, 1], 4.0);

    // Self-assignment
    m1 = m1;
    EXPECT_EQ(m1.rows(), 2);
    EXPECT_EQ(m1.cols(), 2);
    EXPECT_DOUBLE_EQ(m1[0, 0], 1.0);
    EXPECT_DOUBLE_EQ(m1[1, 1], 4.0);
}

TEST(MatrixTest, ArithmeticOperators) {
    linalg::Matrix<double> m1({{1.0, 2.0}, {3.0, 4.0}});
    linalg::Matrix<double> m2({{5.0, 6.0}, {7.0, 8.0}});

    // Test operator+
    linalg::Matrix<double> m3 = m1 + m2;
    EXPECT_EQ(m3.rows(), 2);
    EXPECT_EQ(m3.cols(), 2);
    EXPECT_DOUBLE_EQ(m3[0, 0], 6.0);
    EXPECT_DOUBLE_EQ(m3[0, 1], 8.0);
    EXPECT_DOUBLE_EQ(m3[1, 0], 10.0);
    EXPECT_DOUBLE_EQ(m3[1, 1], 12.0);

    // Test operator-
    linalg::Matrix<double> m4 = m2 - m1;
    EXPECT_EQ(m4.rows(), 2);
    EXPECT_EQ(m4.cols(), 2);
    EXPECT_DOUBLE_EQ(m4[0, 0], 4.0);
    EXPECT_DOUBLE_EQ(m4[0, 1], 4.0);
    EXPECT_DOUBLE_EQ(m4[1, 0], 4.0);
    EXPECT_DOUBLE_EQ(m4[1, 1], 4.0);

    // Test scalar multiplication operator*
    linalg::Matrix<double> m5 = m1 * 2.0;
    EXPECT_EQ(m5.rows(), 2);
    EXPECT_EQ(m5.cols(), 2);
    EXPECT_DOUBLE_EQ(m5[0, 0], 2.0);
    EXPECT_DOUBLE_EQ(m5[0, 1], 4.0);
    EXPECT_DOUBLE_EQ(m5[1, 0], 6.0);
    EXPECT_DOUBLE_EQ(m5[1, 1], 8.0);

    // Test matrix-matrix multiplication
    linalg::Matrix<double> m_mult = m1 * m2;
    EXPECT_EQ(m_mult.rows(), 2);
    EXPECT_EQ(m_mult.cols(), 2);
    // m1 * m2 = [1 2] * [5 6] = [19 22]
    //           [3 4]   [7 8]   [43 50]
    EXPECT_DOUBLE_EQ(m_mult[0, 0], 19.0);  // 1*5 + 2*7 = 19
    EXPECT_DOUBLE_EQ(m_mult[0, 1], 22.0);  // 1*6 + 2*8 = 22
    EXPECT_DOUBLE_EQ(m_mult[1, 0], 43.0);  // 3*5 + 4*7 = 43
    EXPECT_DOUBLE_EQ(m_mult[1, 1], 50.0);  // 3*6 + 4*8 = 50

    // Test matrix-vector multiplication
    linalg::Vector<double> v({5.0, 6.0});
    linalg::Vector<double> mv = m1 * v;
    EXPECT_EQ(mv.size(), 2);
    // [1 2] * [5] = [17]
    // [3 4]   [6]   [39]
    EXPECT_DOUBLE_EQ(mv[0], 17.0);  // 1*5 + 2*6 = 17
    EXPECT_DOUBLE_EQ(mv[1], 39.0);  // 3*5 + 4*6 = 39

    // Test incompatible dimensions for matrix-matrix multiplication
    linalg::Matrix<double> m_incompatible(3, 2, 1.0);
    EXPECT_THROW(m1 * m_incompatible, std::invalid_argument);

    // Test incompatible dimensions for matrix-vector multiplication
    linalg::Vector<double> v_incompatible(3, 1.0);
    EXPECT_THROW(m1 * v_incompatible, std::invalid_argument);

    // Test operator/
    linalg::Matrix<double> m6 = m1 / 2.0;
    EXPECT_EQ(m6.rows(), 2);
    EXPECT_EQ(m6.cols(), 2);
    EXPECT_DOUBLE_EQ(m6[0, 0], 0.5);
    EXPECT_DOUBLE_EQ(m6[0, 1], 1.0);
    EXPECT_DOUBLE_EQ(m6[1, 0], 1.5);
    EXPECT_DOUBLE_EQ(m6[1, 1], 2.0);

    // Test operator+=
    linalg::Matrix<double> m7 = m1;
    m7 += m2;
    EXPECT_EQ(m7.rows(), 2);
    EXPECT_EQ(m7.cols(), 2);
    EXPECT_DOUBLE_EQ(m7[0, 0], 6.0);
    EXPECT_DOUBLE_EQ(m7[0, 1], 8.0);
    EXPECT_DOUBLE_EQ(m7[1, 0], 10.0);
    EXPECT_DOUBLE_EQ(m7[1, 1], 12.0);

    // Test operator-=
    linalg::Matrix<double> m8 = m2;
    m8 -= m1;
    EXPECT_EQ(m8.rows(), 2);
    EXPECT_EQ(m8.cols(), 2);
    EXPECT_DOUBLE_EQ(m8[0, 0], 4.0);
    EXPECT_DOUBLE_EQ(m8[0, 1], 4.0);
    EXPECT_DOUBLE_EQ(m8[1, 0], 4.0);
    EXPECT_DOUBLE_EQ(m8[1, 1], 4.0);

    // Test operator*=
    linalg::Matrix<double> m9 = m1;
    m9 *= 2.0;
    EXPECT_EQ(m9.rows(), 2);
    EXPECT_EQ(m9.cols(), 2);
    EXPECT_DOUBLE_EQ(m9[0, 0], 2.0);
    EXPECT_DOUBLE_EQ(m9[0, 1], 4.0);
    EXPECT_DOUBLE_EQ(m9[1, 0], 6.0);
    EXPECT_DOUBLE_EQ(m9[1, 1], 8.0);

    // Test operator/=
    linalg::Matrix<double> m10 = m1;
    m10 /= 2.0;
    EXPECT_EQ(m10.rows(), 2);
    EXPECT_EQ(m10.cols(), 2);
    EXPECT_DOUBLE_EQ(m10[0, 0], 0.5);
    EXPECT_DOUBLE_EQ(m10[0, 1], 1.0);
    EXPECT_DOUBLE_EQ(m10[1, 0], 1.5);
    EXPECT_DOUBLE_EQ(m10[1, 1], 2.0);

    // Test mismatched size operations
    linalg::Matrix<double> m11(3, 2, 1.0);
    EXPECT_THROW(m1 + m11, std::invalid_argument);
    EXPECT_THROW(m1 - m11, std::invalid_argument);
    EXPECT_THROW(m1 += m11, std::invalid_argument);
    EXPECT_THROW(m1 -= m11, std::invalid_argument);

    // Test division by zero
    EXPECT_THROW(m1 / 0.0, std::invalid_argument);
    EXPECT_THROW(m1 /= 0.0, std::invalid_argument);
}

TEST(MatrixTest, ComparisonOperators) {
    linalg::Matrix<double> m1({{1.0, 2.0}, {3.0, 4.0}});
    linalg::Matrix<double> m2({{1.0, 2.0}, {3.0, 4.0}});
    linalg::Matrix<double> m3({{4.0, 3.0}, {2.0, 1.0}});
    linalg::Matrix<double> m4({{1.0, 2.0}, {3.0, 5.0}});
    linalg::Matrix<double> m5({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});

    // Test operator==
    EXPECT_TRUE(m1 == m2);
    EXPECT_FALSE(m1 == m3);
    EXPECT_FALSE(m1 == m4);
    EXPECT_FALSE(m1 == m5);

    // Test operator!=
    EXPECT_FALSE(m1 != m2);
    EXPECT_TRUE(m1 != m3);
    EXPECT_TRUE(m1 != m4);
    EXPECT_TRUE(m1 != m5);
}

TEST(MatrixTest, Transpose) {
    linalg::Matrix<double> m1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});

    // Test transposed()
    linalg::Matrix<double> m2 = m1.transposed();
    EXPECT_EQ(m2.rows(), 3);
    EXPECT_EQ(m2.cols(), 2);
    EXPECT_DOUBLE_EQ(m2[0, 0], 1.0);
    EXPECT_DOUBLE_EQ(m2[0, 1], 4.0);
    EXPECT_DOUBLE_EQ(m2[1, 0], 2.0);
    EXPECT_DOUBLE_EQ(m2[1, 1], 5.0);
    EXPECT_DOUBLE_EQ(m2[2, 0], 3.0);
    EXPECT_DOUBLE_EQ(m2[2, 1], 6.0);

    // Test transpose()
    m1.transpose();
    EXPECT_EQ(m1.rows(), 3);
    EXPECT_EQ(m1.cols(), 2);
    EXPECT_DOUBLE_EQ(m1[0, 0], 1.0);
    EXPECT_DOUBLE_EQ(m1[0, 1], 4.0);
    EXPECT_DOUBLE_EQ(m1[1, 0], 2.0);
    EXPECT_DOUBLE_EQ(m1[1, 1], 5.0);
    EXPECT_DOUBLE_EQ(m1[2, 0], 3.0);
    EXPECT_DOUBLE_EQ(m1[2, 1], 6.0);
}

TEST(MatrixTest, MatrixVectorMultiplication) {
    linalg::Matrix<double> m1({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    linalg::Vector<double> v1({1.0, 2.0, 3.0});

    // Test matrix-vector multiplication
    linalg::Vector<double> result = m1 * v1;
    EXPECT_EQ(result.size(), 2);
    // [1 2 3]   [1]   [14]
    // [4 5 6] * [2] = [32]
    //           [3]
    EXPECT_DOUBLE_EQ(result[0], 14.0);  // 1*1 + 2*2 + 3*3 = 14
    EXPECT_DOUBLE_EQ(result[1], 32.0);  // 4*1 + 5*2 + 6*3 = 32

    // Test with different dimensions
    linalg::Matrix<double> m2({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    linalg::Vector<double> v2({1.0, 2.0});

    linalg::Vector<double> result2 = m2 * v2;
    EXPECT_EQ(result2.size(), 3);
    // [1 2]   [1]   [5]
    // [3 4] * [2] = [11]
    // [5 6]         [17]
    EXPECT_DOUBLE_EQ(result2[0], 5.0);   // 1*1 + 2*2 = 5
    EXPECT_DOUBLE_EQ(result2[1], 11.0);  // 3*1 + 4*2 = 11
    EXPECT_DOUBLE_EQ(result2[2], 17.0);  // 5*1 + 6*2 = 17

    // Test incompatible dimensions
    linalg::Vector<double> v3({1.0, 2.0});
    EXPECT_THROW(m1 * v3, std::invalid_argument);
}

TEST(MatrixTest, MatrixMatrixMultiplication) {
    linalg::Matrix<double> m1({{1.0, 2.0}, {3.0, 4.0}});
    linalg::Matrix<double> m2({{5.0, 6.0}, {7.0, 8.0}});

    // Test matrix-matrix multiplication
    linalg::Matrix<double> result = m1 * m2;
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 2);
    // [1 2]   [5 6]   [19 22]
    // [3 4] * [7 8] = [43 50]
    EXPECT_DOUBLE_EQ(result[0, 0], 19.0);  // 1*5 + 2*7 = 19
    EXPECT_DOUBLE_EQ(result[0, 1], 22.0);  // 1*6 + 2*8 = 22
    EXPECT_DOUBLE_EQ(result[1, 0], 43.0);  // 3*5 + 4*7 = 43
    EXPECT_DOUBLE_EQ(result[1, 1], 50.0);  // 3*6 + 4*8 = 50

    // Test non-square matrix multiplication
    linalg::Matrix<double> m3({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    linalg::Matrix<double> m4({{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}});

    linalg::Matrix<double> result2 = m3 * m4;
    EXPECT_EQ(result2.rows(), 2);
    EXPECT_EQ(result2.cols(), 2);
    // [1 2 3]   [7  8]   [58  64]
    // [4 5 6] * [9  10] = [139 154]
    //           [11 12]
    EXPECT_DOUBLE_EQ(result2[0, 0], 58.0);   // 1*7 + 2*9 + 3*11 = 58
    EXPECT_DOUBLE_EQ(result2[0, 1], 64.0);   // 1*8 + 2*10 + 3*12 = 64
    EXPECT_DOUBLE_EQ(result2[1, 0], 139.0);  // 4*7 + 5*9 + 6*11 = 139
    EXPECT_DOUBLE_EQ(result2[1, 1], 154.0);  // 4*8 + 5*10 + 6*12 = 154

    // Test incompatible dimensions
    linalg::Matrix<double> m5({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    EXPECT_THROW(m1 * m5, std::invalid_argument);
}

TEST(MatrixTest, Inverse) {
    linalg::Matrix<double> m1({{4.0, 7.0}, {2.0, 6.0}});

    // Test inverse()
    linalg::Matrix<double> m2 = m1.inverse();
    EXPECT_EQ(m2.rows(), 2);
    EXPECT_EQ(m2.cols(), 2);

    // Check that m1 * m2 = I
    linalg::Matrix<double> identity = m1 * m2;
    EXPECT_NEAR(identity[0, 0], 1.0, 1e-10);
    EXPECT_NEAR(identity[0, 1], 0.0, 1e-10);
    EXPECT_NEAR(identity[1, 0], 0.0, 1e-10);
    EXPECT_NEAR(identity[1, 1], 1.0, 1e-10);

    // Test non-invertible matrix
    linalg::Matrix<double> m3({{1.0, 2.0}, {2.0, 4.0}});  // det = 0
    EXPECT_THROW(m3.inverse(), std::runtime_error);

    // Test non-square matrix
    linalg::Matrix<double> m4(2, 3, 1.0);
    EXPECT_THROW(m4.inverse(), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}