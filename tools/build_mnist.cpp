#include <iostream>
#include "CNN/dataset_t.h"
#include "util/mnist.h"

int main()
{
	dataset_t mnist = load_mnist("train-images.idx3-ubyte",
				     "train-labels.idx1-ubyte");

	std::ofstream out ("mnist.dataset",std::ofstream::binary);
	mnist.write(out);
	out.close();

	return 0;
}
