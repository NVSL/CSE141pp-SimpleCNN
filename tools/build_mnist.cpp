#include <iostream>
#include "CNN/dataset_t.hpp"
#include "util/mnist.hpp"
#include <cstring>

int main(int argc, char *argv[])
{
	if (argc != 5) {
		std::cerr << "Usage; build_mnist.exe <image file> <labels file> <output file> <digits|extended>\n";
		exit(1);
	}
	
	dataset_t mnist = load_mnist(argv[1], argv[2], !strcmp(argv[4], "extended"), 10000);

	std::ofstream out (argv[3],std::ofstream::binary);
	mnist.write(out);
	out.close();

	return 0;
}
