#include <iostream>
#include "CNN/canela.hpp"
#include "util/mnist.hpp"

using namespace std;

int main()
{

	dataset_t mnist = load_mnist("../datasets/mnist/train-images.idx3-ubyte",
				     "../datasets/mnist/train-labels.idx1-ubyte");

	model_t model;

	conv_layer_t  layer1( 1, 5, 32, tdsize(mnist.test_cases[0].data.size.x*16,
					       mnist.test_cases[0].data.size.y*16,
					       mnist.test_cases[0].data.size.z*16));
	relu_layer_t layer2( layer1.out.size );
	pool_layer_t layer3( 2, 2, layer2.out.size );	
	conv_layer_t layer4( 1, 3, 10, layer3.out.size );
	relu_layer_t layer5( layer4.out.size );
	pool_layer_t layer6( 2, 2, layer5.out.size );

	fc_layer_t layer7( layer6.out.size, 120 );
	fc_layer_t layer8( layer7.out.size, 36 );

	model.add_layer(layer1 );
	model.add_layer(layer2 );
	model.add_layer(layer3 );
	model.add_layer(layer4 );
	model.add_layer(layer5 );
	model.add_layer(layer6 );
	model.add_layer(layer7 );
	model.add_layer(layer8 );

	
	std::cout << model.geometry() << "\n";
	std::cout << "Training data size: " << (mnist.get_total_memory_size()+0.0)/(1024*1024)  << " MB" << std::endl;
	double amse = 0;
	int ic = 0;
	int ep = 0;
#define COUNT 10000
	do {
		for ( test_case_t& t : mnist.test_cases )

		{
			double xerr = model.train(t.data, t.label );
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
		tensor_t<double>& out = model.apply(t.data);
		
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
