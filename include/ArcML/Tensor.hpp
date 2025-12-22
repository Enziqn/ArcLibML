#pragma once
#include <vector>
#include <stdexcept>

namespace ArcML {

	struct Tensor1D {
		std::vector<double> data;

		Tensor1D() = default;
		Tensor1D(std::size_t size) : data(size) {}


	};

	struct Tensor2D {
	
	};

	struct Tensor3D {

	};
}