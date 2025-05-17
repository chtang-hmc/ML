#ifndef MOMENTUM_H
#define MOMENTUM_H

#include "optimizer.h"

/**
 * @brief Momentum gradient descent optimizer.
 *
 * This class implements the momentum gradient descent optimizer.
 *
 */
class Momentum : public Optimizer {
   public:
    Momentum(double learning_rate = 0.01, double momentum = 0.9,
             size_t max_iterations = 1000, double tolerance = 1e-6)
        : Optimizer(learning_rate, max_iterations, tolerance),
          momentum_(momentum) {}

    Vec minimize(const ObjectiveFunction& objective,
                 const GradientFunction& gradient,
                 const Vec& initial_point) override;

   private:
    double momentum_;

    Vec update_velocity(const Vec& velocity, const Vec& gradient);
    Vec update_position(const Vec& position, const Vec& velocity);
};

#endif
