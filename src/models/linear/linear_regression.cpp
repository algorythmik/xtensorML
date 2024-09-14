#include <ostream>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xarray.hpp>
#include "xtensor_ml/models/linear/linear_regression.hpp"
#include "xtensor/xmanipulation.hpp"
#include <xtensor/xbuilder.hpp>
#include <xtensor/xadapt.hpp>

namespace xtensor_ml{
namespace linear_model{
LinearRegression::LinearRegression(bool fit_intercept)
        : is_fit_(false), fit_intercept_(fit_intercept){}

LinearRegression& LinearRegression::fit(
    const xt::xarray<double>& X,
    const xt::xarray<double>& y
   )
    {
        auto X_ = LinearRegression::update_(X);
        auto X_t = xt::transpose(X_);
        auto sigma_inv = xt::linalg::pinv(xt::linalg::dot(X_t, X_));
        auto y_ = xt::atleast_2d(y);
        auto XTy = xt::linalg::dot(X_t, xt::transpose(y_));
        beta_ = xt::linalg::dot(sigma_inv, XTy);
        is_fit_ = true;
        return *this;


    }

xt::xarray<double> LinearRegression::predict(const xt::xarray<double>& X) const{
    if (!is_fit_)
        throw std::runtime_error("Model must be fitted before prediction");

    auto X_ = LinearRegression::update_(X);
    return xt::linalg::dot(X_, beta_);
}

xt::xarray<double> LinearRegression::update_(const xt::xarray<double>& X) const {
    if (fit_intercept_){
        int N = (int)X.shape()[0];
        auto ones = xt::ones<double>({N, 1});
        return  xt::concatenate(xt::xtuple(ones, X), 1);
    }
    else {
        return X;
    }
}
xt::xarray<double> LinearRegression::get_beta() const {
       return beta_;
   }
} // namespace linear_model
} // namespace xtensor_ml
