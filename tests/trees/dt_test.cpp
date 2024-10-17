#include "xtensor/xtensor_forward.hpp"
#include <gtest/gtest.h>
#include <xtensor_ml/trees/dt.hpp>

using xt::xarray;

TEST(entropyTESTS, entropyUniform) {
  xarray<int> labels = {1, 1, 1, 1};
  double entropy = xtensor_ml::trees::Entropy(labels);
  ASSERT_NEAR(entropy, 0.0, 1e-7);
};

TEST(giniTESTS, giniUniform) {
  xarray<int> labels = {1, 1, 1, 1};
  double entropy = xtensor_ml::trees::Gini(labels);
  ASSERT_NEAR(entropy, 0.0, 1e-7);
};

TEST(DecisionTreeTest, FitMethodWorks) {
  xt::xarray<double> X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 6.0}, {4.0, 8.0}};
xt:
  xarray<int> y = {0, 0, 1, 1};
  int expected_depth = 2;
  xtensor_ml::trees::DecisionTree tree;
  EXPECT_NO_THROW(tree.fit(X, y));
};

TEST(DecisionTreeTest, PredictTest) {
  // Create a small dataset
  xt::xarray<double> X = {{2.0, -.5}, {1.0, .5}, {4.0, 0.5}, {5.0, -0.5}};
  xt::xarray<int> y = {0, 0, 1, 1};

  xtensor_ml::trees::DecisionTree tree(3,
                                       xtensor_ml::trees::Criterion::entropy);

  tree.fit(X, y);

  xt::xarray<int> predictions = tree.predict(X);

  xt::xarray<int> expected = {0, 0, 1, 1};
  EXPECT_EQ(predictions, expected);
}
