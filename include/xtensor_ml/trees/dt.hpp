#pragma once

#include "xtensor/xtensor_forward.hpp"
#include <cstddef>
#include <memory>
#include <stdlib.h>
#include <xtensor/xarray.hpp>
using xt::xarray;
namespace xtensor_ml {
namespace trees {
class Node {
public:
  std::shared_ptr<Node> left;
  std::shared_ptr<Node> right;
  size_t feature;
  double threshold;

  Node(const std::shared_ptr<Node> left_child,
       const std::shared_ptr<Node> right_child, size_t feature,
       double threshold)
      : left(left_child), right(right_child), feature(feature),
        threshold(threshold) {};
};

class Leaf : public Node {

public:
  xarray<double> value;
  explicit Leaf(xarray<double> &value_)
      : Node(nullptr, nullptr, -1, 0), value(value_) {};
};
enum class Criterion {
  gini,
  entropy,
};

// for now classifier is always is assumed to be true,
class DecisionTree {
public:
  explicit DecisionTree(int max_depth = 100,
                        Criterion criterion = Criterion::gini);
  DecisionTree &fit(const xarray<double> &X, const xarray<double> &y);
  xarray<double> predict(const xarray<double> &X);

private:
  std::shared_ptr<Node> root_;
  int max_depth_;
  Criterion criterion_;
  size_t depth_;

  /// Recursively grow the decison tree.
  std::shared_ptr<Node> Grow(const xarray<double> &X, const xarray<int> &y,
                             size_t cur_depth);
  /// Finds the best feature and trheshold to split the data
  std::pair<size_t, double> Segment(const xarray<double> &X,
                                    const xarray<int> &y,
                                    const xarray<size_t> &feat_id);
  /// Calculates the impurity gain of a split
  double ImpurityGain(const xarray<double> &y, double split_thresh,
                      const xarray<double> &feat_values);
  double Impurity(const xarray<int> &y);
  int Traverse(const xarray<double> &X, const std::shared_ptr<Node> &node);
};

double Entropy(const xarray<int> &y);
double Gini(const xarray<int> &y);
} // namespace trees
} // namespace xtensor_ml
