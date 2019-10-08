#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "CNN/cnn.h"
#include "CNN/test_case.h"
#include "util/mnist.h"

using namespace std;

int main()
{
	vector<case_t> cases = load_mnist("train-images.idx3-ubyte",
					  "train-labels.idx1-ubyte");

	vector<layer_t*> layers;

	conv_layer_t * layer1 = new conv_layer_t( 1, 5, 8, cases[0].data.size );		// 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t * layer2 = new relu_layer_t( layer1->out.size );
	pool_layer_t * layer3 = new pool_layer_t( 2, 2, layer2->out.size );				// 24 * 24 * 8 -> 12 * 12 * 8
	fc_layer_t * layer4 = new fc_layer_t(layer3->out.size, 10);					// 4 * 4 * 16 -> 10

	layers.push_back( (layer_t*)layer1 );
	layers.push_back( (layer_t*)layer2 );
	layers.push_back( (layer_t*)layer3 );
	layers.push_back( (layer_t*)layer4 );

	float amse = 0;
	int ic = 0;
	int ep = 0;
	do {
		for ( case_t& t : cases )
		{
			float xerr = train( layers, t.data, t.out );
			amse += xerr;
			
			ep++;
			ic++;
			
			if ( ep % 1000 == 0 )
				cout << "case " << ep << " err=" << amse/ic << endl;
			if ( ep > 100) {
				break;
			}
		}
	} while (ep <= 100);

	int correct  = 0, incorrect = 0;
	ep = 0;
	for ( case_t& t : cases )
	{
		forward(layers, t.data);

		tensor_t<float>& out = layers.back()->out;
		tdsize guess = layers.back()->out.argmax();
	        tdsize answer = t.out.argmax();
		if (guess == answer) {
			correct++;
		} else {
			incorrect++;
			std::cout << guess << " != " << answer << "\n";
		}
		ep++;
		if (ep > 100) {
			break;
		}
	}

	std::cout << "Accuracy: " << (correct+0.0)/(correct+ incorrect +0.0) << ": " << correct << "/" << correct + incorrect << "\n";
	return 0;
}
