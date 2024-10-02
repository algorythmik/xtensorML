#pragma once

#include "xtensor/xtensor_forward.hpp"
#include <xtensor/xarray.hpp>
namespace xtensor_ml{
namespace linear_models{
class LogisticRegressionTest;

enum class Penalty{
    l1,
    l2
};

class LogisticRegression {
    public:
        explicit LogisticRegression(double gamma, Penalty penalty, bool fit_intercept=true);

        LogisticRegression& fit(
            const xt::xarray<double>& X,
            const xt::xarray<double>& y
        );

        xt::xarray<double> predict(const xt::xarray<double>& X) const;
        xt::xarray<double> get_coefs() const;

    private:
        double gamma_;
        bool fit_intercept_;
        Penalty penalty_;
        bool is_fit_;

        friend LogisticRegressionTest;

        double NLL_(
                const xt::xarray<double>& y,
                const xt::xarray<double>& y_pred
        ) const;
        xt::xarray<double> NLL_grad_(
                const xt::xarray<double>& X,
                const xt::xarray<double>& y,
                const xt::xarray<double>& Y_pred
        ) const;
        static xt::xarray<double> sigmoid_(
                const xt::xarray<double>& z);


};
} // end of linear_modles namespace
} // end of xtensor_ml
