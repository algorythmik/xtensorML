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

class LogisticRegressionFTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code before each test
        X_train = xt::xarray<double>{
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {4.0, 5.0}
        };
        y_train = xt::xarray<double>{0, 0, 1, 1}; // Binary classification labels
    }

    xt::xarray<double> X_train;
    xt::xarray<double> y_train;
};

// Test for fit method
TEST_F(LogisticRegressionFTest, FitTest) {
    xtensor_ml::linear_models::LogisticRegression model(
            0.01, xtensor_ml::linear_models::Penalty::l2, true);
    model.fit(X_train, y_train, 0.01, 1e-7, 10000); // Fitting the model

    // Check that the model is fit by inspecting the coefficients
    xt::xarray<double> coefs = model.get_coefs();

    // Expected size of coefficients
    ASSERT_EQ(coefs.shape()[0], X_train.shape()[1] + 1); // +1 for intercept

    // Optionally, print the coefficients
}

// Test for predict method
TEST_F(LogisticRegressionFTest, PredictTest) {
    xtensor_ml::linear_models::LogisticRegression model(0.01, xtensor_ml::linear_models::Penalty::l2, true);
    model.fit(X_train, y_train, 0.01, 1e-7, 10000); // Fit the model

    // Make predictions
    xt::xarray<double> predictions = model.predict(X_train);


    // Check if the predicted values are reasonable (close to 0 or 1)
    for (size_t i = 0; i < predictions.size(); ++i) {
        EXPECT_TRUE(predictions(i) >= 0.0 && predictions(i) <= 1.0); // Logistic regression predicts probabilities
    }
}
} // linear_models
} // xtensor_ml
