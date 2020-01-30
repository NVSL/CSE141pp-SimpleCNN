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


void EXPECT_TENSOR_EQ(const tensor_t<double> & a,const tensor_t<double> & b) {
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

      	
	class SimplificationTests :  public ::testing::Test {
		
	};

	TEST_F(SimplificationTests, level_0_fc) {
		fc_test<simple_fc_layer_t>(1,1,1,1,1);
	}	
			  
	TEST_F(SimplificationTests, level_1_fc) {
		fc_test<simple_fc_layer_t>(4,  4,  4,  4, 1);
		fc_test<simple_fc_layer_t>(4,  4,  2,  8, 1);
		fc_test<simple_fc_layer_t>(8,  8,  2,  16,1);
		fc_test<simple_fc_layer_t>(32, 32, 8,  4, 1);
		fc_test<simple_fc_layer_t>(64, 64, 16, 8, 1);
	}
	TEST_F(SimplificationTests, level_2_fc) {
		fc_test<simple_fc_layer_t>(4,  6,  6,  6,  1);
		fc_test<simple_fc_layer_t>(4,  8,  2,  2,  1);
		fc_test<simple_fc_layer_t>(12, 12, 3,  3,  1);
		fc_test<simple_fc_layer_t>(24, 48, 24, 12, 1);
		fc_test<simple_fc_layer_t>(16, 96, 2,  12, 1);
	}

	TEST_F(SimplificationTests, level_3_fc) {
		fc_test<simple_fc_layer_t>(3,  7,  13, 7,  1);
		fc_test<simple_fc_layer_t>(5,  9,  17, 11, 1);
		fc_test<simple_fc_layer_t>(31, 29, 5,  13, 1);
		fc_test<simple_fc_layer_t>(89, 31, 7,  19, 1);
		fc_test<simple_fc_layer_t>(3,  17, 31, 23, 1);
	}

	TEST_F(SimplificationTests, level_4_fc) {
		for (int i = 0; i < 20; i++) {
			srand(i);
			int x = RAND_LARGE(32);
			int y = RAND_LARGE(48);
			int z = RAND_LARGE(48);
			int out = RAND_LARGE(16);
			
			fc_test<simple_fc_layer_t>(x,y,z,out,1);
		}
		
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
