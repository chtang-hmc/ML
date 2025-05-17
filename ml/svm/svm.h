// header file for SVM

#ifndef ML_SVM_H
#define ML_SVM_H

#include <string>

#include "linalg.h"
#include "./model.h"

class SVM : public MLModel {
   public:
    // Constructor
    SVM(double C = 1.0, double tol = 1e-3, int max_iter = 1000,
        const std::string &kernel = "linear");

    // Destructor
    ~SVM() = default;

    // Public methods for training and predicting
    void fit(const linalg::Matrix<double> &X,
             const linalg::Vector<double> &y) override;
    linalg::Vector<double> predict(
        const linalg::Matrix<double> &X) const override;

   private:
    double C_;            // Regularization parameter
    double tol_;          // Tolerance for convergence
    int max_iter_;        // Maximum number of iterations
    std::string kernel_;  // Kernel type (e.g., "linear", "rbf")
};

#endif