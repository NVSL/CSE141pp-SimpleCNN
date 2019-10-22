#include <iostream>
#include <algorithm>
#include <vector>
#include "CNN/canela.hpp"
#include "CNN/dataset_t.hpp"
#include "util/mnist.hpp"

int main()
{
	std::vector<test_case_t> cases = load_mnist("train-images.idx3-ubyte",
					       "train-labels.idx1-ubyte");

	for ( test_case_t& t : cases )
	{
		std::cout << t.out.argmax() << " :\n" << t.data << "\n";
	} 

	return 0;
}
