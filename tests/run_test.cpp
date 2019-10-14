#define INCLUDE_TESTS
#include "../CNN/cnn.h"
#include "../util/tensor_util.h"
#include <iostream>
#include "gtest/gtest.h"
#include "util/png_util.h"
#include "util/jpeg_util.h"

namespace CNNTest{

	TEST_F(CNNTest, simple_model_math) {
		tensor_t<float> data(4, 4, 2);
		tensor_t<float> expected(2,1,1);
		model_t model;
		srand(42);
		conv_layer_t layer1( 1, 2, 2, data.size );
		relu_layer_t layer2(layer1.out.size );
		pool_layer_t layer3( 1, 2, layer2.out.size );
		fc_layer_t layer4(layer3.out.size, 2);
		
		model.add_layer(layer1 );
		model.add_layer(layer2 );
		model.add_layer(layer3 );
		model.add_layer(layer4 );

		for(int i = 0; i < 100; i++) {
			randomize(data);
			randomize(expected);
			model.train(data, expected); 
		}
		
		//std::cout << layer4.weights << "\n";
		//for(auto & k: layer1.filters) {
		//std::cout << k << "\n";
		//}

		randomize(expected);
		//std::cout <<  model.apply(data) << "\n";
	}


	TEST_F(CNNTest, simple_model_opt) {
		tensor_t<float> data(32, 32, 3);
		randomize(data);
		tensor_t<float> expected(10, 1, 1);
		randomize(expected);

		srand(42);
		model_t model;
		conv_layer_t layer1( 1, 5, 8, data.size );
		relu_layer_t layer2( layer1.out.size );
		pool_layer_t layer3( 2, 2, layer2.out.size );
		fc_layer_t layer4(layer3.out.size, 10);
		
		model.add_layer(layer1 );
		model.add_layer(layer2 );
		model.add_layer(layer3 );
		model.add_layer(layer4 );

		srand(42);
		model_t model_o;
		conv_layer_opt_t layer1_o( 1, 5, 8, data.size );
		relu_layer_opt_t layer2_o( layer1_o.out.size );
		pool_layer_opt_t layer3_o( 2, 2, layer2_o.out.size );
		fc_layer_opt_t   layer4_o(layer3_o.out.size, 10);
		
		model_o.add_layer(layer1_o );
		model_o.add_layer(layer2_o );
		model_o.add_layer(layer3_o );
		model_o.add_layer(layer4_o );

		for(int i = 0; i < 100; i++) {
			randomize(data);
			randomize(expected);
			model.train(data, expected); 
			model_o.train(data, expected); 
		}
		
		randomize(data);
		EXPECT_EQ(model.apply(data), model_o.apply(data));
		
	}
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
