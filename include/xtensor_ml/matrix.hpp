#pragma once
#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>   // For matrix output

namespace xtensor_ml{
namespace matrix {

// A utility function to print a matrix (xtensor array) - for illustration
template <typename T>
void print_matrix(const xt::xarray<T>& mat) {
    std::cout << mat << std::endl;
}

}  // namespace matrix
}  // namespace your_library
