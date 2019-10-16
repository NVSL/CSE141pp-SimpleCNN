#define INCLUDE_TESTS
#include "../CNN/cnn.h"
#include "../util/tensor_util.h"
#include <iostream>
#include "gtest/gtest.h"
#include "util/png_util.h"
#include "util/jpeg_util.h"
#include "util/mnist.h"
#include "CNN/dataset_t.h"

namespace CNNTest{

	class SimpleCNNTest : public ::testing::Test {
	protected:
		dataset_t rand_ds;
		void SetUp() override {
			srand(42);
			tensor_t<float> data(32, 32, 3);
			tensor_t<float> label(10,1,1);
			for(int i = 0; i < 100; i++) {
				randomize(data);
				randomize(label);
				rand_ds.add(data, label);
			}
		}
		
	};

	
	TEST_F(SimpleCNNTest, simple_model_math) {
		model_t model;
		srand(42);
		
		conv_layer_t layer1( 1, 2, 2, 0, rand_ds.data_size);
		relu_layer_t layer2(layer1.out.size );
		pool_layer_t layer3( 1, 2, 0, layer2.out.size );
		fc_layer_t layer4(layer3.out.size, 10);
		
		model.add_layer(layer1 );
		model.add_layer(layer2 );
		model.add_layer(layer3 );
		model.add_layer(layer4 );
		
		for(auto & c: rand_ds.test_cases) {
			model.train(c.data, c.label);
		}

		// batch training
		for(auto i = rand_ds.begin(); i != rand_ds.end();) 
			model.train_batch(rand_ds, i, 11);

	}


	TEST_F(SimpleCNNTest, simple_model_opt) {
		srand(42);
		model_t model;
		conv_layer_t layer1( 1, 5, 8, 0, rand_ds.data_size );
		relu_layer_t layer2( layer1.out.size );
		pool_layer_t layer3( 2, 2, 0, layer2.out.size );
		fc_layer_t layer4(layer3.out.size, 10);
		
		model.add_layer(layer1 );
		model.add_layer(layer2 );
		model.add_layer(layer3 );
		model.add_layer(layer4 );

		srand(42);
		model_t model_o;
		conv_layer_opt_t layer1_o( 1, 5, 8, 0, rand_ds.data_size);
		relu_layer_opt_t layer2_o( layer1_o.out.size );
		pool_layer_opt_t layer3_o( 2, 2, 0, layer2_o.out.size );
		fc_layer_opt_t   layer4_o(layer3_o.out.size, 10);
		
		model_o.add_layer(layer1_o );
		model_o.add_layer(layer2_o );
		model_o.add_layer(layer3_o );
		model_o.add_layer(layer4_o );

		for(auto & c: rand_ds.test_cases) {
			model.train(c.data, c.label); 
			model_o.train(c.data, c.label); 
		}
		
		tensor_t<float> t(rand_ds.data_size);
		for(int i = 0;i < 10; i++) {
			randomize(t);
			EXPECT_TENSOR_EQ(model.apply(t), model_o.apply(t));
		}
		
	}
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
