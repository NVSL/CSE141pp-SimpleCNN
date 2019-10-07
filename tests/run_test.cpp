#define INCLUDE_TESTS
#include "../CNN/cnn.h"
#include <iostream>
#include "gtest/gtest.h"

namespace CNNTest{
	TEST_F(CNNTest, simple_model_opt) {
		tensor_t<float> data(32, 32, 3);
		randomize(data);
		tensor_t<float> expected(10, 1, 10);
		randomize(expected);

		std::vector<layer_t*> layers;
		srand(42);
		conv_layer_t * layer1 = new conv_layer_t( 1, 5, 8, data.size );
		relu_layer_t * layer2 = new relu_layer_t( layer1->out.size );
		pool_layer_t * layer3 = new pool_layer_t( 2, 2, layer2->out.size );
		fc_layer_t * layer4 = new fc_layer_t(layer3->out.size, 10);
		
		layers.push_back(layer1 );
		layers.push_back(layer2 );
		layers.push_back(layer3 );
		layers.push_back(layer4 );

		train(layers, data, expected);

		std::vector<layer_t*> layers_o;
		srand(42);
		conv_layer_opt_t * layer1_o = new conv_layer_opt_t( 1, 5, 8, data.size );
		relu_layer_opt_t * layer2_o = new relu_layer_opt_t( layer1_o->out.size );
		pool_layer_opt_t * layer3_o = new pool_layer_opt_t( 2, 2, layer2_o->out.size );
		fc_layer_opt_t * layer4_o = new fc_layer_opt_t(layer3_o->out.size, 10);
		
		layers_o.push_back(layer1_o );
		layers_o.push_back(layer2_o );
		layers_o.push_back(layer3_o );
		layers_o.push_back(layer4_o );

		train(layers_o, data, expected);
		
	}
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
