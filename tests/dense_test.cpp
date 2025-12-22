#include <ArcML/ArcML.hpp>

int main() {

	std::vector<double> input = { 0.5, -0.2, 0.1 };
	ArcML::Dense d(3, 40);
	ArcML::Dense a(20, 40, ArcML::DenseInitialization::UniformSigned);

	d.fowardPropagation(input, ArcML::Activations::ReLU);
	

	return 0;
}
