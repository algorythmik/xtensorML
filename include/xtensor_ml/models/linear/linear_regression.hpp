#pragma once

#include <xtensor/xarray.hpp>

namespace xtensor_ml {
namespace linear_model {
class LinearRregression {
    public:
        // constructor
        explicit LinearRregression(bool fit_intercept=true);

        // fit method to train the model
            LinearRregression& fit(
                const xt::xarray<double>& X,
                const xt::xarray<double>& y,
                const xt::xarray<double>* weights = nullptr
              );

            xt::xarray<double> predict(const xt::xarray<double>& X) const;

    private:
        bool fit_intercept_;
        xt::xarray<double> beta_;
        xt::xarray<double> sigma_inv_;
        bool is_fit_;

    void update_1d(const xt::xarray<double>& X, const xt::xarray<double>& y, const xt::xarray<double>& w);
    void update_2d(const xt::xarray<double>& X, const xt::xarray<double>& y, const xt::xarray<double>& w);

};
}
}
