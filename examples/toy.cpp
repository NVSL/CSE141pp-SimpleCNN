#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "CNN/cnn.h"
#include "CNN/dataset_t.h"
#include "util/mnist.h"
#include "util/tensor_util.h"

using namespace std;

int main()
{
	
//	dataset_t mnist = load_mnist("../datasets/mnist/train-images.idx3-ubyte",
//				     "../datasets/mnist/train-labels.idx1-ubyte");
	std::ifstream in("../tools/mnist.dataset",std::ofstream::binary);
	throw_assert(in.good(), "Couldn't open mnist.dataset");
	dataset_t toy;
	model_t model;

	tensor_t<float> d(3,3,1);
	tensor_t<float> l(2,2,1);

	d(0,0,0) = 1;
	d(0,1,0) = 0;
	d(0,2,0) = 0;

	d(1,0,0) = 0;
	d(1,1,0) = 1;
	d(1,2,0) = 0;

	d(2,0,0) = 0;
	d(2,1,0) = 0;
	d(2,2,0) = 1;

	
	l(0,0,0) = 0;
	l(0,1,0) = 1;

	l(1,0,0) = 0;
	l(1,1,0) = 0;

	toy.add(d, l);
	
	conv_layer_t  layer1(2, 2, 1, 0, {3,3,1});
	layer1.filters[0] = tensor_t<float>(2,2,1);
	layer1.filters[0](0,0,0) = 1;
	layer1.filters[0](0,1,0) = 1;
	layer1.filters[0](1,0,0) = 1;
	layer1.filters[0](1,1,0) = 1;
	model.add_layer(layer1);
	model.train(toy.test_cases[0],
		    true);

	return 0;
}
