#pragma once

#include "xtensor/xtensor_forward.hpp"
#include <memory>
#include <xtensor/xarray.hpp>
#include <stdlib.h>
using xt::xarray;
namespace xtensor_ml{
namespace trees{
class Leaf{

    public:
        xarray<double> value;
        Leaf(xarray<double>& value_):
            value(value_){};
};
class Node{
    public:
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;
        size_t feature;
        double threshold;

        Node(
             const std::shared_ptr<Node> left_child,
             const std::shared_ptr<Node> right_child,
             size_t feature,
             double threshold
        ):
            left(left_child), right(right_child), feature(feature), threshold(threshold){};
};

// for now classifier is always is assumed to be true,
class DecisionTree{
    public:
        DecisionTree(bool classifier, int max_depth):
            max_depth(max_depth){};
        DecisionTree& fit(const xarray<double>& X, const xarray<double>& y);
        xarray<double> predict(const xarray<double>& X);

    private:
        std::shared_ptr<Node> root;
        int max_depth;
};

double entropy(const xarray<int>& y);
}// xtensor_ml
}// trees
