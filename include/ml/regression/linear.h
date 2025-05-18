// Class for general linear regression

#ifndef ML_REGRESSION_LINEAR_H
#define ML_REGRESSION_LINEAR_H

#include "linalg.h"
#include "./model.h"
#include <string>

/*
 * Class for linear regression model.
 */
class LinearModel : public MLModel {
   public:
    // Constructor
    LinearModel(double alpha = 0.01, int max_iter = 1000,
                const std::string &solver = "sgd");

    // Destructor
    ~LinearModel() = default;

    // Public methods for training and predicting
    void fit(const linalg::Matrix<double> &X,
             const linalg::Vector<double> &y) override;
    linalg::Vector<double> predict(
        const linalg::Matrix<double> &X) const override;

   private:
    double alpha_;        // Learning rate
    int max_iter_;        // Maximum number of iterations
    std::string solver_;  // Solver type (e.g., "sgd", "normal")
};

#endif  // ML_REGRESSION_LINEAR_H