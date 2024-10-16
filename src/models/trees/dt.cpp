#include "xtensor_ml/trees/dt.hpp"
#include "xtensor/xhistogram.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xoperation.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xtensor_forward.hpp"
#include <cstddef>
#include <limits>
#include <memory>
// using xtensor_ml::trees::DecisionTree;
using xt::xarray;
namespace xtensor_ml {
namespace trees {

DecisionTree::DecisionTree(int max_depth, Criterion criterion)
    : max_depth_(max_depth), criterion_(criterion), root_(nullptr),
      depth_(0) {};
DecisionTree &DecisionTree::fit(const xarray<double> &X,
                                const xarray<double> &y) {
  root_ = Grow(X, y, 0);
  return *this;
};

std::shared_ptr<Node> DecisionTree::Grow(const xarray<double> &X,
                                         const xarray<int> &y,
                                         size_t cur_depth) {
  if (xt::unique(y).size() == 1 || cur_depth >= max_depth_) {
    xarray<double> proba = xt::bincount(y) / static_cast<double>(y.size());
    return std::make_shared<Leaf>(proba);
  }
  cur_depth++;
  depth_ = std::max(cur_depth, depth_);
  auto range = xt::xarray<size_t>::from_shape({X.shape(1)});
  std::iota(range.begin(), range.end(), 0); // Fill wi
  xarray<size_t> feat_idxs = xt::random::choice(range, X.shape(1), false)();
  auto result = Segment(X, y, feat_idxs);
  size_t best_feat = result.first;
  double best_thresh = result.second;
  auto feat_values = xt::view(X, xt::all(), best_feat);
  auto left_idx = xt::flatten_indices(xt::where(feat_values <= best_thresh));
  auto right_idx = xt::flatten_indices(xt::where(feat_values > best_thresh));
  auto left = Grow(xt::view(X, xt::keep(left_idx)), xt::view(y, xt::keep(left_idx)), cur_depth);
  auto right = Grow(xt::view(X, xt::keep(right_idx)), xt::view(y, xt::keep(right_idx)), cur_depth);
  return std::make_shared<Node>(left, right, best_feat, best_thresh);
};

std::pair<size_t, double>
DecisionTree::Segment(const xarray<double> &X, const xarray<int> &y,
                      const xarray<size_t> &feat_idxs) {
  double best_gain = -std::numeric_limits<double>::infinity();
  int best_feat = -1;
  double best_thresh = 0.0;

  for (size_t feat_idx : feat_idxs) {
    xarray<double> vals = xt::view(X, xt::all(), feat_idx);
    xarray<double> levels = xt::unique(vals);
    xarray<double> thresholds =
        (xt::view(levels, xt::range(0, -1))) +
        xt::view(levels, xt::range(1, levels.size() + 1)) / 2.0;

    for (auto &threshold : thresholds) {
      double gain = ImpurityGain(y, threshold, vals);
      if (gain > best_gain) {
        best_gain = gain;
        best_feat = feat_idx;
        best_thresh = threshold;
      }
    }
  }
  return {best_feat, best_thresh};
};

double DecisionTree::ImpurityGain(const xarray<double> &y, double split_thresh,
                                  const xarray<double> &feat_values) {
  auto left = xt::where(feat_values <= split_thresh);
  auto right = xt::where(feat_values > split_thresh);
  if (left.size() == 0 || right.size() == 0)
    return 0.0;

  double n = y.size();
  double n_left = left.size();
  double n_right = right.size();

  double imp_left = Impurity(xt::index_view(y, left));
  double imp_right = Impurity(xt::index_view(y, right));
  double child_imp = (n_left * imp_left) / n + (n_right * imp_right) / n;
  double parent_imp = Impurity(y);
  return parent_imp - child_imp;
};

double DecisionTree::Impurity(const xt::xarray<int> &y) {
  if (criterion_ == Criterion::gini) {
    return Gini(y);
  } else if (criterion_ == Criterion::entropy) {
    return Entropy(y);
  }
  return 0.0;
}
double Entropy(const xarray<int> &y) {
  xarray<double> bin_count = xt::bincount(y);
  xarray<double> total = xt::sum(bin_count);
  xarray<double> proba = bin_count / total;
  auto log_probs = proba * xt::log2(proba);
  auto values = xt::where(proba > 0, -proba * log_probs, 0);
  return xt::sum(values)();
}
double Gini(const xarray<int> &y) {
  // Gini impurity measure
  xarray<double> bin_count = xt::bincount(y);
  xarray<double> total = xt::sum(bin_count);
  xarray<double> proba = bin_count / total;
  return 1.0 - xt::sum(xt::square(proba))();
}
} // namespace trees
} // namespace xtensor_ml
