#define INCLUDE_TESTS
#include "../CNN/cnn.h"
#include "../util/tensor_util.h"
#include <iostream>
#include "gtest/gtest.h"
#include "util/png_util.h"
#include "util/jpeg_util.h"
#include "util/mnist.h"
#include "CNN/dataset_t.h"

#define EXCLUDE_MAIN
#include "../examples/simple.cpp"
#undef  EXCLUDE_MAIN

void EXPECT_TENSOR_EQ(const tensor_t<float> & a,const tensor_t<float> & b) {
	EXPECT_EQ(a.size, b.size);
	TENSOR_FOR(a, x,y,z) {
		EXPECT_FLOAT_EQ(a(x,y,z), b(x,y,z));
	}
}		


namespace CNNTest {

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
	}


	TEST_F(CNNTest, learning_SLOW) {
		float accuracy1 = simple(3, 1000);
		float accuracy2 = simple(3, 2000);
		EXPECT_NEAR(accuracy2 - accuracy1, 0.222, 0.005);
	}
}

#define PREFIX(x) opt_##x
#include "optimization_tests_inc.cpp"
#undef PREFIX


int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
