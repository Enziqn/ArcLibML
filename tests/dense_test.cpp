#include <ArcML/ArcML.hpp>

int main() {

	ArcML::Dense d(20, 40);
	ArcML::Dense a(20, 40, ArcML::DenseInitialization::UniformSigned);

	return 0;
}
