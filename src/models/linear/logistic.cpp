#include "xtensor_ml/models/linear/logistic.hpp"
#include "xtensor/xtensor_forward.hpp"
#include <limits>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
namespace xtensor_ml {
namespace linear_models {
// constructor
LogisticRegression::LogisticRegression(double gamma, Penalty penalty,
                                       bool fit_intercept)
    : gamma_(gamma), penalty_(penalty), fit_intercept_(fit_intercept),
      is_fit_(false) {}

LogisticRegression &LogisticRegression::fit(const xt::xarray<double> &X,
                                            const xt::xarray<double> &y,
                                            double lr, double tol,
                                            size_t max_iter) {
  auto X_ = update_(X);
  std::cout << X_ << std::endl;
  auto prev_loss = std::numeric_limits<double>::infinity();
  size_t num_features = X_.shape()[1];
  beta_ = xt::random::randn<double>({num_features});

  for (size_t i = 0; i <= max_iter; i++) {
    auto logit = xt::linalg::dot(X_, beta_);
    auto y_pred = sigmoid_(logit);
    double loss = NLL_(y, y_pred);
    if (prev_loss - loss < tol)
      break;
    beta_ = beta_ - lr * NLL_grad_(X_, y, y_pred);
  }

  return *this;
}

xt::xarray<double> LogisticRegression::get_coefs() const { return beta_; }

xt::xarray<double>
LogisticRegression::predict(const xt::xarray<double> &X) const {
  auto X_ = update_(X);
  return sigmoid_(xt::linalg::dot(X_, beta_));
}
// sigmoid function
xt::xarray<double> LogisticRegression::sigmoid_(const xt::xarray<double> &z) {
  return 1.0 / (1 + xt::exp(-z));
}

// Negative log likelihood
double LogisticRegression::NLL_(const xt::xarray<double> &y,
                                const xt::xarray<double> &y_pred) const {
  auto log_y_pred = xt::log(y_pred);
  auto log_1_minus_y_pred = xt::log(1 - y_pred);
  return -xt::sum(y * log_y_pred + (1 - y) * log_1_minus_y_pred)();
}
// gradient of negative log likelihood
xt::xarray<double>
LogisticRegression::NLL_grad_(const xt::xarray<double> &X,
                              const xt::xarray<double> &y,
                              const xt::xarray<double> &Y_pred) const {
  // Compute the gradient: X^T * (Y_pred - y)
  auto error = Y_pred - y;
  return xt::linalg::dot(xt::transpose(X), error);
}

xt::xarray<double>
LogisticRegression::update_(const xt::xarray<double> &X) const {
  if (fit_intercept_) {
    int N = (int)X.shape()[0];
    auto ones = xt::ones<double>({N, 1});
    return xt::concatenate(xt::xtuple(ones, X), 1);
  } else {
    return X;
  }
}
} // namespace linear_models
} // namespace xtensor_ml
