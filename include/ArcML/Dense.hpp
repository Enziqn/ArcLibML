#pragma once
#include <vector>
#include <stdexcept>
#include <random>
#include <ostream>
#include "ArcKeywords.hpp"

// Dense: Y = X * W^T + b

namespace ArcML {

	struct Dense {
		std::size_t input_size = 0;
		std::size_t neuron_count = 0; //number of neurons in the layer
		std::vector<std::vector<double>> Weights; //Weights matrix
		std::vector<double> Biases; //Bias vector
		ArcML::DenseInitialization Initial_Type = ArcML::DenseInitialization::UniformSigned;

		Dense() = default;

		Dense(std::size_t input_sz, std::size_t Neuron_ct)
			: input_size(input_sz), neuron_count(Neuron_ct), Weights(Neuron_ct, std::vector<double>(input_sz)), Biases(Neuron_ct) {

			const int SEED = 42; // Fixed seed for reproducibility
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
			: input_size(input_sz), neuron_count(Neuron_ct), Weights(Neuron_ct, std::vector<double>(input_sz)), Biases(Neuron_ct), Initial_Type(typeofInitialization) {
			
			const int SEED = 42; // Fixed seed for reproducibility

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

		//friend function to print the Dense layer
		friend std::ostream& operator<<(std::ostream& os, const Dense& d);

	}; // struct Dense

	inline std::ostream& operator<<(std::ostream& os, const Dense& d)
	{
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