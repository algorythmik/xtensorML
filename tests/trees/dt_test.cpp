#include "xtensor/xtensor_forward.hpp"
#include <gtest/gtest.h>
#include <xtensor_ml/trees/dt.hpp>

using xt::xarray;

TEST(entropyTESTS, entropyUniform){
    xarray<int> labels = {1, 1, 1, 1};
    double entropy = xtensor_ml::trees::entropy(labels);
    ASSERT_NEAR(entropy, 0.0, 1e-7);
};
