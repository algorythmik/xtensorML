#include "xtensor_ml/matrix.hpp"
#include <xtensor/xarray.hpp>

int main() {
    // Create a 2x2 matrix with some values
    xt::xarray<double> mat = {{1.0, 2.0}, {3.0, 4.0}};

    xtensor_ml::matrix::print_matrix(mat);
    return 0;
}
