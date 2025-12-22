#pragma once
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace ArcML {

    enum class Activations {
        None,
        ReLU,
        // add more later: Sigmoid, Tanh, LeakyReLU, ...
    };

    inline double activate_scalar(double x, Activations a) {
        switch (a) {
        case Activations::None: return x;
        case Activations::ReLU: return std::max(0.0, x);
        default: throw std::runtime_error("Unsupported Activation Function");
        }
    }

    inline void apply_activation_inplace(std::vector<double>& v, Activations a) {
        if (a == Activations::None) return;
        for (double& x : v) x = activate_scalar(x, a);
    }

    inline std::vector<double> apply_activation_copy(const std::vector<double>& v, Activations a) {
        std::vector<double> out = v;
        apply_activation_inplace(out, a);
        return out;
    }

} // namespace ArcML
