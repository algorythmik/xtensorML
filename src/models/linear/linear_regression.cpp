#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xview.hpp>
#include <iostream>

#include "xtensor_ml/models/linear/linear_regression.hpp"

namespace xtensor_ml{
namespace linear_model{
LinearRegression::LinearRegression()
        : is_fit_(false){}

LinearRegression& LinearRegression::fit(
    const xt::xarray<double>& X,
    const xt::xarray<double>& y
   )
    {
        auto X_t = xt::transpose(X);
        auto sigma_inv = xt::linalg::pinv(xt::linalg::dot(X_t, X));
        beta_ = xt::linalg::dot(sigma_inv, xt::linalg::dot(X_t, y));
        is_fit_ = true;
        return *this;


    }

xt::xarray<double> LinearRegression::predict(const xt::xarray<double>& X) const{
    if (!is_fit_)
        throw std::runtime_error("Model must be fitted before prediction");

    return xt::linalg::dot(X, beta_);
}

} // namespace linear_model
} // namespace xtensor_ml
