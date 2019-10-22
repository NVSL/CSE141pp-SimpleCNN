#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "CNN/cnn.hpp"
#include "CNN/dataset_t.hpp"
#include "util/mnist.hpp"
#include "CNN/throw_assert.hpp"

using namespace std;

model_t * build_two_layer(const dataset_t & ds)  {
	model_t * model = new model_t();
	fc_layer_t *layer1 = new fc_layer_t(ds.data_size, 100);
	fc_layer_t *layer2 = new fc_layer_t(layer1->out.size, ds.label_size.x);
	model->add_layer(*layer1 );
	model->add_layer(*layer2 );
	return model;
}

model_t * build_perceptron(const dataset_t & ds)  {
	model_t * model = new model_t();
	fc_layer_t *layer2 = new fc_layer_t(ds.data_size, ds.label_size.x);
	model->add_layer(*layer2 );
	return model;
}

model_t * build_conv(const dataset_t & ds)  {
	model_t * model = new model_t();

	layer_t *layer1 = new conv_layer_t(1, 5, 5, 1, ds.data_size);
	layer_t * layer2 = new relu_layer_t( layer1->out.size );
	layer_t *layer3 = new fc_layer_t(layer2->out.size, ds.label_size.x);
	model->add_layer(*layer1 );
	model->add_layer(*layer2 );
	model->add_layer(*layer3 );
	return model;
}

model_t * build_deep(const dataset_t & ds) {
	model_t * model = new model_t();

	conv_layer_t *layer1 = new conv_layer_t ( 1, 5, 8, 0, ds.data_size);
	relu_layer_t *layer2= new relu_layer_t ( layer1->out.size );
	pool_layer_t *layer3= new pool_layer_t ( 2, 2, 0, layer2->out.size );				// 24 24 8 -> 12 12 8
			                
	conv_layer_t *layer4= new conv_layer_t ( 1, 3, 10, 0, layer3->out.size );			// 12 12 6 -> 10 10 10
	relu_layer_t *layer5= new relu_layer_t ( layer4->out.size );
	pool_layer_t *layer6= new pool_layer_t ( 2, 2, 0, layer5->out.size );				// 10 10 10 -> 5 5 10
			                
	fc_layer_t   *layer7= new fc_layer_t   ( layer6->out.size, ds.label_size.x );					// 4 * 4 * 16 -> 10

	model->add_layer(*layer1 );
	model->add_layer(*layer2 );
	model->add_layer(*layer3 );
	model->add_layer(*layer4 );
	model->add_layer(*layer5 );
	model->add_layer(*layer6 );
	model->add_layer(*layer7 );

	return model;
}

float simple(int which, int samples) 
{
	
	dataset_t mnist_train = dataset_t::read("../datasets/mnist/mnist-train.dataset", samples);
	dataset_t mnist_test = dataset_t::read("../datasets/mnist/mnist-test.dataset", samples);
	       
	if (samples == 0) {
		samples = mnist_train.test_cases.size();
	}

	model_t * model = nullptr;
	
	switch(which)  {
	case 0:
		model = build_perceptron(mnist_train);
		break;
	case 1:
		model = build_two_layer(mnist_train);
		break;
	case 2:
		model = build_conv(mnist_train);
		break;
	case 3:
		model = build_deep(mnist_train);
		break;
	default:
		throw_assert(false, "Illegal model number: " << which << "\n");
	}

	float amse = 0;
	int ic = 0;
	int ep = 0;
	float error;

	std::cout << model->geometry() << "\n";
	std::cout << "Training data size: " << (mnist_train.get_total_memory_size()+0.0)/(1024*1024)  << " MB" << std::endl;
	std::cout << "Training cases    : " << mnist_train.size() << std::endl;
	std::cout << "Testing data size : " << (mnist_test.get_total_memory_size()+0.0)/(1024*1024)  << " MB" << std::endl;
	std::cout << "Testing cases     : " << mnist_test.size() << std::endl;

	do {
		for ( test_case_t& t : mnist_train.test_cases )
		{
			float xerr = model->train(t);
			amse += xerr;
			
			ep++;
			ic++;
			error = amse/ic;
			if ( ep % 1000 == 0 ) 
				cout << "case " << ep << " err=" << error << endl;
			if ( ep > samples) {
				break;
			}
		}
	} while (ep <= samples);

	int correct  = 0, incorrect = 0;
	ep = 0;
	for ( test_case_t& t : mnist_test.test_cases )
	{
		tensor_t<float>& out = model->apply(t.data);
		
		tdsize guess = out.argmax();
	        tdsize answer = t.label.argmax();
		if (guess == answer) {
			correct++;
		} else {
			incorrect++;
		}
		ep++;
		if (ep > samples) {
			break;
		}
	}

	float total_error = (correct+0.0)/(correct+ incorrect +0.0);
	std::cout << "Accuracy: " << total_error << ": " << correct << "/" << correct + incorrect << "\n";
	return total_error;
}


#ifndef EXCLUDE_MAIN
int main() {
	simple(3, 1000);
	return 0;
}
#endif
