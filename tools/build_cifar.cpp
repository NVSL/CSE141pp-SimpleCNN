#include <iostream>
#include "CNN/dataset_t.hpp"
#include "util/cifar.hpp"

int main(int argc, char *argv[])
{
	if (argc != 4) {
		std::cerr << "Usage: build_cifar.exe <batch_file> <10 for cifar10; 100 for cifar100> <output file>\n";
		exit(1);
	}
	
	dataset_t cifar = load_cifar(argv[1], argv[2] == std::string("100"), argv[2] == std::string("100") ? 1000 : 1000);

	std::ofstream out (argv[3],std::ofstream::binary);
	cifar.write(out);
	out.close();

	return 0;
}
