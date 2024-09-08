#pragma once

#include <xtensor/xarray.hpp>

namespace xtensor_ml {
namespace linear_model {
class LinearRegression {
    public:
        // constructor
        explicit LinearRegression(bool fit_intercept=true);

        // fit method to train the model
        LinearRegression& fit(
            const xt::xarray<double>& X,
            const xt::xarray<double>& y
        );

          xt::xarray<double> predict(const xt::xarray<double>& X) const;

    private:
        bool fit_intercept_;
        xt::xarray<double> beta_;
        bool is_fit_;

};
}
}
