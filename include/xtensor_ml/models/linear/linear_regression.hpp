#pragma once

#include <xtensor/xarray.hpp>

namespace xtensor_ml {
namespace linear_model {
class LinearRregression {
    public:
        // constructor
        explicit LinearRregression();

        // fit method to train the model
        LinearRregression& fit(
            const xt::xarray<double>& X,
            const xt::xarray<double>& y
        );

          xt::xarray<double> predict(const xt::xarray<double>& X) const;

    private:
        xt::xarray<double> beta_;
        bool is_fit_;

};
}
}
