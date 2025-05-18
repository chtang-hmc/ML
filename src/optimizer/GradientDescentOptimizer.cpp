#include "GradientDescentOptimizer.h"
#include <spdlog/spdlog.h>

GradientDescentOptimizer::GradientDescentOptimizer(double learning_rate,
                                                   size_t max_iterations,
                                                   double tolerance)
    : Optimizer(learning_rate, max_iterations, tolerance) {}

Vec GradientDescentOptimizer::minimize(const ObjectiveFunction& objective,
                                       const GradientFunction& gradient,
                                       const Vec& initial_point, bool verbose) {
    Vec current_point = initial_point;
    double prev_value = objective(current_point);

    if (verbose) {
        spdlog::set_level(spdlog::level::info);
    } else {
        spdlog::set_level(spdlog::level::warn);
    }

    for (size_t iter = 0; iter < max_iterations_; ++iter) {
        if (iter % 10 == 0) {
            spdlog::info("Iteration {}: Current point: {}, Objective value: {}",
                         iter, current_point, prev_value);
        }

        // Compute gradient at current point
        Vec grad = gradient(current_point);

        // Update parameters: x = x - learning_rate * gradient
        current_point = current_point - learning_rate_ * grad;

        // Check convergence
        double current_value = objective(current_point);
        if (std::abs(current_value - prev_value) < tolerance_) {
            spdlog::info(
                "Convergence reached at iteration {} with tolerance {}.", iter,
                tolerance_);
            break;
        }
        prev_value = current_value;
    }

    spdlog::info("Gradient Descent converged after {} iterations.",
                 max_iterations_);

    return current_point;
}

std::vector<std::tuple<Vec, double>> GradientDescentOptimizer::minimize_history(
    const ObjectiveFunction& objective, const GradientFunction& gradient,
    const Vec& initial_point, bool verbose) {
    std::vector<std::tuple<Vec, double>> history;
    Vec current_point = initial_point;
    double prev_value = objective(current_point);

    if (verbose) {
        spdlog::set_level(spdlog::level::info);
    } else {
        spdlog::set_level(spdlog::level::warn);
    }

    for (size_t iter = 0; iter < max_iterations_; ++iter) {
        if (iter % 10 == 0) {
            spdlog::info("Iteration {}: Current point: {}, Objective value: {}",
                         iter, current_point, prev_value);
        }

        // Compute gradient at current point
        Vec grad = gradient(current_point);

        // Update parameters: x = x - learning_rate * gradient
        current_point -= learning_rate_ * grad;

        // Store the current point and its function value
        history.emplace_back(current_point, objective(current_point));

        // Check convergence
        double current_value = objective(current_point);
        if (std::abs(current_value - prev_value) < tolerance_) {
            spdlog::info(
                "Convergence reached at iteration {} with tolerance {}.", iter,
                tolerance_);
            break;
        }
        prev_value = current_value;
    }

    spdlog::info("Gradient Descent converged after {} iterations.",
                 max_iterations_);

    return history;
}