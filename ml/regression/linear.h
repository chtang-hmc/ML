// Class for general linear regression

#ifndef ML_REGRESSION_LINEAR_H
#define ML_REGRESSION_LINEAR_H

#include "linalg.h"
#include <string>

/*
 * Class for linear regression model. For storing coefficients and showing
 * results. This class provides methods for predicting and evaluating the model.
 * It does not contain hyperparameters or methods for training.
 */
class LinearModel {
   public:
    // Constructor
    LinearModel(const Vec &coefficients);

    // Destructor
    ~LinearModel() = default;

    // Public methods for predicting and evaluating
    Vec predict(const Mat &X) const;
    double score(const Vec &y_true, const Vec &y_pred) const;

   private:
    Vec coefficients_;  // Coefficients of the model
};

/*
 * Class for general linear regression model. For computation only.
 * This class contains hyperparameters and methods for training.
 * It does not store the coefficients or show results.
 * Largely a private class.
 */
class LinearRegressor {
   public:
    // Constructor
    LinearRegressor(double alpha = 0.01, int max_iter = 1000, double tol = 1e-6,
                    const std::string &solver = "sgd");

    // Destructor
    ~LinearRegressor() = default;

    // Public methods for training and predicting
    LinearModel fit(const Mat &X, const Vec &y);

   private:
    double alpha_;        // Learning rate
    int max_iter_;        // Maximum number of iterations
    double tol_;          // Tolerance for convergence
    std::string solver_;  // Solver type (e.g., "sgd", "normal"); if "normal",
                          // use normal equation to compute coefficients and
                          // previous coefficients are not used
};

#endif  // ML_REGRESSION_LINEAR_H