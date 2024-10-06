// #include "xtensor_ml/trees/dt.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor_forward.hpp"
#include "xtensor/xhistogram.hpp"
// using xtensor_ml::trees::DecisionTree;
using xt::xarray;
namespace xtensor_ml {
namespace trees {

// DecisionTree& fit(const xarray<double> &X, const xarray<double>& y){
// };
double entropy(const xarray<int>& y){
    std::cout<<"y: "<<y<<std::endl;
    xarray<double> bin_count = xt::bincount(y);
    xarray<double> total = xt::sum(bin_count);
    xarray<double> proba = bin_count / total;
    auto log_probs = proba * xt::log2(proba);
    auto values = xt::where(proba > 0, -proba * log_probs, 0);
    return xt::sum(values)();

}
}
}


