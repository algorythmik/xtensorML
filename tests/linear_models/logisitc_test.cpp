#include "xtensor_ml/models/linear/logistic.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <gtest/gtest.h>
namespace xtensor_ml{

namespace linear_models{

// Define the test class
class LogisticRegressionTest : public ::testing::Test {
};
//Test for sigmoid function

TEST(LogisticRegressionTest, SigmoidTest) {
    xt::xarray<double> z = {-1000.0, 0.0, 1000.0};

    xt::xarray<double> expected = {0.0, 0.5, 1.0}; // Sigmoid of large negative -> 0, 0 -> 0.5, large positive -> 1

    xt::xarray<double> actual = LogisticRegression::sigmoid_(z);

    // Use a small tolerance because sigmoid will return floating-point numbers
    EXPECT_TRUE(xt::allclose(actual, expected, 1e-6, 1e-6));
}

// Test for Negative Log-Likelihood (NLL)
TEST(LogisticRegressionTest, NLLTest) {
    // Define a small dataset
    xt::xarray<double> y = {0.0, 1.0, 1.0};          // Actual class labels
    xt::xarray<double> y_pred = {0.1, 0.9, 0.8};     // Predicted probabilities

    // Create a LogisticRegression object
    LogisticRegression model(0.1, Penalty::l2);

    // Calculate NLL manually for comparison
    double expected_nll = -(xt::sum(y * xt::log(y_pred) + (1.0 - y) * xt::log(1.0 - y_pred)))();

    // Call NLL_ method of the LogisticRegression class
    double actual_nll = model.NLL_(y, y_pred);

    EXPECT_NEAR(actual_nll, expected_nll, 1e-6);
}

}
}
