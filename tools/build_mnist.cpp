#include <iostream>
#include "CNN/dataset_t.h"
#include "util/mnist.h"

int main(int argc, char *argv[])
{
	if (argc != 4) {
		std::cerr << "Usage; build_mnist.exe <image file> <labels file> <output file>\n";
		exit(1);
	}
	
	dataset_t mnist = load_mnist(argv[1], argv[2]);

	std::ofstream out (argv[3],std::ofstream::binary);
	mnist.write(out);
	out.close();

	return 0;
}
