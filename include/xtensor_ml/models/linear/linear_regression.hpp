#pragma once

#include "xtensor/xtensor_forward.hpp"
#include <xtensor/xarray.hpp>

namespace xtensor_ml {
namespace linear_models {
class LinearRegression {
public:
  explicit LinearRegression(bool fit_intercept = true);

  LinearRegression &fit(const xt::xarray<double> &X,
                        const xt::xarray<double> &y);

  xt::xarray<double> predict(const xt::xarray<double> &X) const;
  xt::xarray<double> get_beta() const;

private:
  bool fit_intercept_;
  xt::xarray<double> beta_;
  bool is_fit_;
  xt::xarray<double> update_(const xt::xarray<double> &X) const;
};
} // namespace linear_models
} // namespace xtensor_ml
