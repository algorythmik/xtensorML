#include "xtensor_ml/models/linear/logistic.h"
#include <xtensor/xarray.hpp>
#include <xtensor/math.hpp>

namespace xtensor_ml {
namespace linear_models {
// constructor
LogisticRegression::LogisticRegression(
        double gamma, Penalty Penalty::l1, bool fit_intercpet_)
    :gamma_(gamma), penalty_(Penalty::l1),
    fit_intercept_(fit_intercept), is_fit_(false){}

// sigmoid function
xt::xarray<double> LogisticRegression::sigmoid_(const xt:xarray<double>& z){
    return 1.0 / (1 + xt::exp(-z));
}

// Negative log likelihood
double NLL_(
        const xt::xarray<double>& y,
        const xt::xarray<double>& y_pred) const
    {
    auto log_y_pred = xt::log(y_pred);
    auto log_1_minus_y_pred = xt::log(1 - y_pred);
    return -xt::sum(y * log_y_pred + (1 - y)* log_1_minus_y_pred);
}


} // end of linear_models
} // end of xtensor_ml namespace
