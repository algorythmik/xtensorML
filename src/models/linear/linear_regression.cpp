#include <algorithm>
#include <cwchar>
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
        auto X_ = X;
        if (fit_intercept_){
            int N = (int)X.shape()[0];
            auto ones = xt::ones<double>({N, 1});
            X_ = xt::concatenate(xt::xtuple(ones, X), 1);
        }
        auto X_t = xt::transpose(X_);
        auto sigma_inv = xt::linalg::pinv(xt::linalg::dot(X_t, X_));
        auto y_ = xt::atleast_2d(y);
        std::cout << "Shape of y_: "<<y_.shape()[0] <<"," << y_.shape()[1] <<std::endl;
        auto inter = xt::linalg::dot(X_t, xt::transpose(y_));
        std::cout << "inter:" << inter<<std::endl;
        std::cout << "sigma_inv: "<< sigma_inv<<std::endl;
        std::cout <<"Shape of inter: " << inter.shape()[0] <<","<< inter.shape()[1]<<std::endl;
        std::cout <<"Shape of sigma_inv: " << sigma_inv.shape()[0] <<","<< sigma_inv.shape()[1]<<std::endl;
        beta_ = xt::linalg::dot(sigma_inv, inter);
        is_fit_ = true;
        return *this;


    }

xt::xarray<double> LinearRegression::predict(const xt::xarray<double>& X) const{
    if (!is_fit_)
        throw std::runtime_error("Model must be fitted before prediction");
    auto X_ = X;
    if (fit_intercept_){
        int N = (int)X.shape()[0];
        auto ones = xt::ones<double>({N, 1});
        X_ = xt::concatenate(xt::xtuple(ones, X), 1);
    }
    return xt::linalg::dot(X_, beta_);
}

} // namespace linear_model
} // namespace xtensor_ml
