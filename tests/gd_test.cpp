#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <iostream>

#include "optimization/gd/GradientDescentOptimizer.h"

TEST(GradientDescentTest, Basic) {
    GradientDescentOptimizer optimizer;
    Vec initial_point = std::vector<double>{1.0, 2.0};
    GradientDescentOptimizer::ObjectiveFunction objective = [](const Vec& x) {
        return x[0] * x[0] + x[1] * x[1];
    };
    GradientDescentOptimizer::GradientFunction gradient = [](const Vec& x) {
        return Vec{std::vector<double>{2 * x[0], 2 * x[1]}};
    };

    Vec result = optimizer.minimize(objective, gradient, initial_point);
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    return 0;
}