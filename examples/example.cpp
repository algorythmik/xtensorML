#include "xtensor/xtensor_forward.hpp"
#include "xtensor_ml/matrix.hpp"
#include "xtensor_ml/models/linear/linear_regression.hpp"
#include "xtensor_ml/trees/dt.hpp"
#include <iostream>
#include <ostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

int main() {
  // xt::xarray<double> X = {{1.0, 2.0},
  //                        {2.0, 3.0},
  //                        {4.0, 5.0},
  //                        {6.0, 7.0},
  //                        {8.0, 8.0}};
  // xt::xarray<double> y = {4.0, 6.0, 10.0, 14.0, 17.0};
  // auto model = xtensor_ml::linear_models::LinearRegression(true);
  // model.fit(X, y);
  // xt::xarray<double> X_new = {{3.0, 4.0}, {6.0, 7.0}};
  // auto y_pred = model.predict(X_new);
  // std::cout <<"Preicted values:" <<std::endl;
  // std::cout << y_pred <<std::endl;
  // std::cout << "Estimated coefficients are: " << model.get_beta()<<
  // std::endl;
  //
  xt::xarray<int> labels = {1, 1, 1, 1};
  double result = xtensor_ml::trees::entropy(labels);
  std::cout << "entropy for the uniform distribution: " << result << std::endl;
  labels = {1, 1, 0, 0};
  result = xtensor_ml::trees::entropy(labels);
  std::cout << "entropy for the non uniform distribution: " << result
            << std::endl;
  result = xtensor_ml::trees::gini(labels);
  std::cout << "entropy for the non uniform distribution: " << result
            << std::endl;

  return 0;
}
