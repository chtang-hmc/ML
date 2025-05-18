// abstract class for ML model

#ifndef ML_MODEL_H
#define ML_MODEL_H

#include "linalg.h"

class MLModel {
   public:
    // Constructor
    MLModel() = default;

    // Destructor
    virtual ~MLModel() = default;

    // Pure virtual methods for training and predicting
    virtual void fit(const linalg::Matrix<double> &X,
                     const linalg::Vector<double> &y) = 0;
    virtual linalg::Vector<double> predict(
        const linalg::Matrix<double> &X) const = 0;
};

#endif  // ML_MODEL_H