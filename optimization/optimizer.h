#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <functional>
#include "../utils/linalg.h"

/**
 * @brief Base class for optimization algorithms
 *
 * This abstract class defines the interface for various optimization
 * algorithms. Derived classes should implement specific optimization strategies
 * like gradient descent, Newton's method, etc.
 */
class Optimizer {
   public:
    // Type definitions for clarity
    using ObjectiveFunction = std::function<double(const Vec&)>;
    using GradientFunction = std::function<Vec(const Vec&)>;

    /**
     * @brief Constructor for the optimizer
     * @param learning_rate Initial learning rate/step size
     * @param max_iterations Maximum number of iterations
     * @param tolerance Convergence tolerance
     */
    Optimizer(double learning_rate = 0.01, size_t max_iterations = 1000,
              double tolerance = 1e-6)
        : learning_rate_(learning_rate),
          max_iterations_(max_iterations),
          tolerance_(tolerance) {}

    virtual ~Optimizer() = default;

    /**
     * @brief Minimize the objective function starting from initial point
     * @param objective The objective function to minimize
     * @param gradient The gradient function of the objective
     * @param initial_point Starting point for optimization
     * @return The optimal point found
     */
    virtual Vec minimize(const ObjectiveFunction& objective,
                         const GradientFunction& gradient,
                         const Vec& initial_point) = 0;

    // Getters and setters for optimization parameters
    double get_learning_rate() const { return learning_rate_; }
    void set_learning_rate(double rate) { learning_rate_ = rate; }

    size_t get_max_iterations() const { return max_iterations_; }
    void set_max_iterations(size_t max_iter) { max_iterations_ = max_iter; }

    double get_tolerance() const { return tolerance_; }
    void set_tolerance(double tol) { tolerance_ = tol; }

   protected:
    double learning_rate_;   // Step size/learning rate
    size_t max_iterations_;  // Maximum number of iterations
    double tolerance_;       // Convergence tolerance
};

#endif  // OPTIMIZER_H