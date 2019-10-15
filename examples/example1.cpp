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

using namespace std;

int main()
{
	
//	dataset_t mnist = load_mnist("../datasets/mnist/train-images.idx3-ubyte",
//				     "../datasets/mnist/train-labels.idx1-ubyte");
	std::ifstream in("../tools/mnist.dataset",std::ofstream::binary);
	throw_assert(in.good(), "Couldn't open mnist.dataset");
	dataset_t mnist = dataset_t::read(in);
	model_t model;

	conv_layer_t  layer1( 1, 5, 8, 0, mnist.data_size);
	relu_layer_t  layer2( layer1.out.size );
	pool_layer_t layer3( 2, 2, 0, layer2.out.size );				// 24 * 24 * 8 -> 12 * 12 * 8
	fc_layer_t   layer4(layer3.out.size, 10);					// 4 * 4 * 16 -> 10
	softmax_layer_t  layer5(layer4.out.size);					// 4 * 4 * 16 -> 10

	model.add_layer(layer1 );
	model.add_layer(layer2 );
	model.add_layer(layer3 );
	model.add_layer(layer4 );
	model.add_layer(layer5 );

	float amse = 0;
	int ic = 0;
	int ep = 0;
#define COUNT 100
	do {
		for ( test_case_t& t : mnist.test_cases )
		{
			float xerr = model.train(t.data, t.label );
			amse += xerr;
			
			ep++;
			ic++;
			
			if ( ep % 1000 == 0 )
				cout << "case " << ep << " err=" << amse/ic << endl;
			if ( ep > COUNT) {
				break;
			}
		}
	} while (ep <= COUNT);

	int correct  = 0, incorrect = 0;
	ep = 0;
	for ( test_case_t& t : mnist.test_cases )
	{
		tensor_t<float>& out = model.apply(t.data);

		
		tdsize guess = out.argmax();
	        tdsize answer = t.label.argmax();
		if (guess == answer) {
			correct++;
		} else {
			incorrect++;
		}
		ep++;
		if (ep > COUNT) {
			break;
		}
	}

	std::cout << "Accuracy: " << (correct+0.0)/(correct+ incorrect +0.0) << ": " << correct << "/" << correct + incorrect << "\n";
	return 0;
}
