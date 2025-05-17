#ifndef NEWTON_H
#define NEWTON_H

#include "optimizer.h"

/**
 * @brief Newton's method
 *
 * This class implements Newton's method for optimization.
 *
 */
class Newton : public Optimizer {
   public:
    using HessianFunction = std::function<Mat(const Vec&)>;

    Newton(double learning_rate = 0.01, size_t max_iterations = 1000,
           double tolerance = 1e-6)
        : Optimizer(learning_rate, max_iterations, tolerance) {}

    Vec minimize(const ObjectiveFunction& objective,
                 const GradientFunction& gradient,
                 const HessianFunction& hessian,
                 const Vec& initial_point) override;
};

#endif
