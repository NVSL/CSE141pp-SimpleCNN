#define INCLUDE_TESTS
#define DEBUG_OUTPUT "output/"
#include "../CNN/canela.hpp"
#include "../util/tensor_util.hpp"
#include <iostream>
#include <gtest/gtest.h>
#include "util/png_util.hpp"
#include "util/jpeg_util.hpp"
#include "util/mnist.hpp"
#include "util/cifar.hpp"
#include "CNN/dataset_t.hpp"
#include "../CNN/optimized.hpp"
#define EXCLUDE_MAIN
#include "../examples/simple.cpp"
#undef  EXCLUDE_MAIN


void EXPECT_TENSOR_EQ(const tensor_t<double> & a,const tensor_t<double> & c) {
	EXPECT_EQ(a.size, c.size);
	TENSOR_FOR(a, x,y,z,b) {
		EXPECT_FLOAT_EQ(a(x,y,z,b), c(x,y,z,b));
	}
}		


namespace CNNTest {

	class SimpleCNNTest : public ::testing::Test {
	protected:
		dataset_t rand_ds;
		void SetUp() override {
			srand(42);
			tensor_t<double> data(32, 32, 3);
			tensor_t<double> label(10,1,1);
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
		double accuracy1 = simple("deep", "mnist", 5);
		double accuracy2 = simple("deep", "mnist", 10);
		EXPECT_NEAR(accuracy2 - accuracy1, 0.222, 0.005);
	}


	// These are tests of the per-function test functions
	class FunctionTests :  public ::testing::Test {
	};

	
	TEST_F(FunctionTests, fc_activate) {
	fc_test_activate<fc_layer_t>(1,1,1,1,1,1);
}
	
	TEST_F(FunctionTests, fc_calc_grads) {
	fc_test_calc_grads<fc_layer_t>(1,1,1,1,1,1);
}

	TEST_F(FunctionTests, fc_fix_weights) {
	fc_test_fix_weights<fc_layer_t>(1,1,1,1,1,1);
}

	
	
	TEST_F(FunctionTests, conv_activate) {
	conv_test_activate<conv_layer_t>(1,1,1,1,1,1,1,1,1);
	
}
			  
	TEST_F(FunctionTests, conv_calc_grads) {
	conv_test_calc_grads<conv_layer_t>(1,1,1,1,1,1,1,1,1);
}

	TEST_F(FunctionTests, conv_fix_weights) {
	conv_test_fix_weights<conv_layer_t>(1,1,1,1,1,1,1,1,1);
}

	
	
	TEST_F(FunctionTests, pool_activate) {
	pool_test_activate<pool_layer_t>(1,1,1,1,1,1,1,1);
}
			  
	TEST_F(FunctionTests, pool_calc_grads) {
	pool_test_calc_grads<pool_layer_t>(1,1,1,1,1,1,1,1);
}

	TEST_F(FunctionTests, pool_fix_weights) {
	pool_test_fix_weights<pool_layer_t>(1,1,1,1,1,1,1,1);
}

	
	
	TEST_F(FunctionTests, relu_activate) {
	relu_test_activate<relu_layer_t>(1,1,1,1,1);
}
			  
	TEST_F(FunctionTests, relu_calc_grads) {
	relu_test_calc_grads<relu_layer_t>(1,1,1,1,1);
}

	TEST_F(FunctionTests, relu_fix_weights) {
	relu_test_fix_weights<relu_layer_t>(1,1,1,1,1);
}

	
}

#if(0)
#define PREFIX(x) opt_##x
#include "optimization_tests_inc.cpp"
#undef PREFIX
#endif

	int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
		return RUN_ALL_TESTS();
	}
       
