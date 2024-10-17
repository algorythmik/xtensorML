# xtensorML

**xtensorML** is a modern C++ library for machine learning that prioritizes simplicity and clarity for educational purposes.
It aims to provide minimalistic implementations of core ML algorithms using [`xtensor`](https://xtensor.readthedocs.io/en/latest/), a high-performance, NumPy-like library for numerical computation in C++.

# Goal

The goal of xtensorML is to provide straightforward implementations of machine learning algorithms that are easy to read and understand. Each algorithm is implemented in just one .cpp and one .hpp file, making it easy to read and understand.
It is inspired by [numpy-ml](https://github.com/ddbourgin/numpy-ml).

## Key Features:

- **Minimalistic Design**: Each machine learning method is implemented across one `.cpp` and one `.hpp` file, allowing users to easily read and understand the full implementation of an algorithm.
- **Leverage of `xtensor`**: Benefit from the expressive power and performance of `xtensor`, enabling efficient tensor operations with a NumPy-like API.

## Current Algorithms:

- Decision Trees (Classification)
- Linear Regression
- Logistic Regression

# Usage

Here’s a quick example of how to use the Decision Tree classifier:

```C
#include "xtensor_ml/trees/dt.hpp"

xt::xarray<double> X = {{2.3, 1.9}, {1.5, 2.6}, {3.1, 2.9}};
xt::xarray<int> y = {0, 1, 0};

xtensor_ml::trees::DecisionTree tree;
tree.fit(X, y);
auto predictions = tree.predict(X);
```
