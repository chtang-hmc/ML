#include "GD.h"

GD::GD(double learning_rate, size_t max_iterations, double tolerance)
    : Optimizer(learning_rate, max_iterations, tolerance) {}

Vec GD::minimize(const ObjectiveFunction& objective,
                 const GradientFunction& gradient,
                 const Vec& initial_point) override {
    Vec current_point = initial_point;
    double prev_value = objective(current_point);

    for (size_t iter = 0; iter < max_iterations_; ++iter) {
        // Compute gradient at current point
        Vec grad = gradient(current_point);

        // Update parameters: x = x - learning_rate * gradient
        current_point = current_point - learning_rate_ * grad;

        // Check convergence
        double current_value = objective(current_point);
        if (std::abs(current_value - prev_value) < tolerance_) {
            break;
        }
        prev_value = current_value;
    }

    return current_point;
}