#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include "optimizer.h"
#include <cmath>

/**
 * @brief Implementation of simple gradient descent optimization
 *
 * This class implements the basic gradient descent algorithm where
 * parameters are updated by moving in the direction of the negative gradient
 * scaled by the learning rate.
 */
class GradientDescent : public Optimizer {
   public:
    /**
     * @brief Constructor for gradient descent optimizer
     * @param learning_rate Step size for gradient updates
     * @param max_iterations Maximum number of iterations
     * @param tolerance Convergence tolerance
     */
    GradientDescent(double learning_rate = 0.01, size_t max_iterations = 1000,
                    double tolerance = 1e-6);

    /**
     * @brief Minimize the objective function using gradient descent
     * @param objective The objective function to minimize
     * @param gradient The gradient function of the objective
     * @param initial_point Starting point for optimization
     * @return The optimal point found
     */
    Vec minimize(const ObjectiveFunction& objective,
                 const GradientFunction& gradient,
                 const Vec& initial_point) override;
};

#endif  // GRADIENT_DESCENT_H