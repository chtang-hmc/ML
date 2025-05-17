#ifndef SGD_H
#define SGD_H

#include "optimizer.h"

/**
 * @brief Stochastic gradient descent optimizer.
 *
 * This class implements the stochastic gradient descent optimizer.
 *
 */
class SGD : public Optimizer {
   public:
    SGD(double learning_rate = 0.01, size_t max_iterations = 1000,
        double tolerance = 1e-6)
        : Optimizer(learning_rate, max_iterations, tolerance) {}

    Vec minimize(const ObjectiveFunction& objective,
                 const GradientFunction& gradient,
                 const Vec& initial_point) override;
};
#endif
