#include <iostream>
#include "xtensor_ml/matrix.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include "xtensor_ml/models/linear/linear_regression.hpp"

int main() {
    xt::xarray<double> X = {{1.0, 2.0},
                           {2.0, 3.0},
                           {4.0, 5.0}};
    xt::xarray<double> y = {3.0, 5.0, 9.0};
    auto model = xtensor_ml::linear_model::LinearRegression();
    model.fit(X, y);
    xt::xarray<double> X_new = {{3.0, 4.0}, {6.0, 7.0}};
    auto y_pred = model.predict(X_new);
    std::cout <<"Preicted values:" <<std::endl;
    std::cout << y_pred <<std::endl;

    return 0;
}
