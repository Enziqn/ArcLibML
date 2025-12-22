#pragma once
#include <vector>
#include <random>
#include <stdexcept>
#include <ostream>

#include "Activations.hpp"   // <-- split activations here

namespace ArcML {

    enum class DenseInitialization {
        UniformSigned
        // add more later
    };

    struct Dense {
        std::size_t input_size = 0;
        std::size_t neuron_count = 0;
        std::vector<std::vector<double>> Weights;
        std::vector<double> Biases;

        // Pre-activation output (useful for backprop later)
        std::vector<double> Out;

        ArcML::DenseInitialization Initial_Type = ArcML::DenseInitialization::UniformSigned;
        ArcML::Activations Activation_Function = ArcML::Activations::ReLU;

        Dense() = default;

        Dense(std::size_t input_sz, std::size_t Neuron_ct)
            : input_size(input_sz),
            neuron_count(Neuron_ct),
            Weights(Neuron_ct, std::vector<double>(input_sz)),
            Biases(Neuron_ct),
            Out(Neuron_ct)
        {
            const int SEED = 42;
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            std::mt19937 rng(SEED);

            for (std::size_t i = 0; i < neuron_count; i++) {
                for (std::size_t j = 0; j < input_size; j++) {
                    Weights[i][j] = dist(rng);
                }
                Biases[i] = dist(rng);
            }
        }

        Dense(std::size_t input_sz, std::size_t Neuron_ct, ArcML::DenseInitialization typeofInitialization)
            : input_size(input_sz),
            neuron_count(Neuron_ct),
            Weights(Neuron_ct, std::vector<double>(input_sz)),
            Biases(Neuron_ct),
            Out(Neuron_ct),
            Initial_Type(typeofInitialization)
        {
            const int SEED = 42;

            switch (Initial_Type) {
            case ArcML::DenseInitialization::UniformSigned: {
                std::uniform_real_distribution<double> dist(-1.0, 1.0);
                std::mt19937 rng(SEED);

                for (std::size_t i = 0; i < neuron_count; i++) {
                    for (std::size_t j = 0; j < input_size; j++) {
                        Weights[i][j] = dist(rng);
                    }
                    Biases[i] = dist(rng);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported Dense Initialization Type");
            }
        }

        // Keep your call signature the same (Activation passed in)
        std::vector<double> fowardPropagation(const std::vector<double>& input, ArcML::Activations Activation) {
            if (input.size() != input_size) {
                throw std::runtime_error("Input size does not match layer input size.");
            }

            //reset after each call
            std::fill(Out.begin(), Out.end(), 0.0);

            for (std::size_t i = 0; i < neuron_count; ++i) {
                double sum = 0.0;
                for (std::size_t j = 0; j < input_size; ++j) {
                    sum += Weights[i][j] * input[j];
                }
                Out[i] = sum + Biases[i];
            }

            // Activation out = activated copy of Out
            return ArcML::apply_activation_copy(Out, Activation);
        }

        friend std::ostream& operator<<(std::ostream& os, const Dense& d);
    };

    inline std::ostream& operator<<(std::ostream& os, const Dense& d) {
        os << "\nDense Layer:\n";
        for (std::size_t i = 0; i < d.neuron_count; ++i) {
            for (std::size_t j = 0; j < d.input_size; ++j) {
                os << d.Weights[i][j] << (j + 1 < d.input_size ? ' ' : '\n');
            }
        }

        os << "\nBiases:\n";
        for (std::size_t i = 0; i < d.neuron_count; ++i) {
            os << d.Biases[i] << (i + 1 < d.neuron_count ? ' ' : '\n');
        }
        return os;
    }

} // namespace ArcML
