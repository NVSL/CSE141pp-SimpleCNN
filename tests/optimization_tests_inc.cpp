
namespace PREFIX(Tests) {
#define RAND_R(x,y) ((x)+ (rand() % ((y)-(x))))
#define RAND_LARGE(x) RAND_R(x/2, x)

#define REPS 5
	
	class PREFIX(OptimizationTests) :  public ::testing::Test {

	};

	void run_layer(layer_t & l) {
		tensor_t<float> in(l.in.size);
		randomize(in);
		tensor_t<float> next_grads(l.out.size);
		randomize(next_grads);
		l.activate(in);
		l.calc_grads(next_grads);
		l.fix_weights();
	}
	
	void rand_conv(int scale, int reps, int seed) {
		for (int i = 0; i < reps; i++) {
				
			srand(seed++);
			tdsize size(RAND_LARGE(scale),
				    RAND_LARGE(scale),
				    RAND_LARGE(scale/2));
			int ksize =  RAND_LARGE(scale/10);
			int stride = RAND_LARGE(ksize+1);
			int kcount = RAND_LARGE(scale/10);
		
			// Run the optimized version
			srand(seed);
			PREFIX(conv_layer_t) o_layer( stride, ksize, kcount, 0.78,size);
			run_layer(o_layer);

			srand(seed);
			conv_layer_t layer( stride, ksize, kcount, 0.78,size);
			run_layer(layer);
			
			// Check for equality.
			EXPECT_EQ(layer, o_layer);
		}
	}
		

	TEST_F(PREFIX(OptimizationTests), rand_convolution_SLOW) {
		rand_conv(128, 4,  3);
	}
	
	TEST_F(PREFIX(OptimizationTests), rand_convolution) {
		rand_conv(32,  64, 1);
		rand_conv(64,  16, 2);
	}

	void rand_dropout(int scale, int reps, int seed) {
		for (int i = 0; i < reps; i++) {
				
			srand(seed++);
			tdsize size(RAND_LARGE(scale),
				    RAND_LARGE(scale),
				    RAND_LARGE(scale/2));
		
			// Run the optimized version
			srand(seed);
			PREFIX(dropout_layer_t) o_layer( size, 0.5);
			run_layer(o_layer);

			srand(seed);
			dropout_layer_t layer(size, 0.5);
			run_layer(layer);
			
			// Check for equality.
			EXPECT_EQ(layer, o_layer);
		}
	}
		
	TEST_F(PREFIX(OptimizationTests), rand_dropout) {
		rand_dropout(32,  64, 1);
		rand_dropout(64,  16, 2);
	}
	
	void rand_fc(int scale, int reps, int seed) {
		for (int i = 0; i < reps; i++) {
			
			srand(seed++);
			tdsize size(RAND_LARGE(scale),
				    RAND_LARGE(scale),
				    RAND_LARGE(scale/2));
			int out_size = RAND_LARGE(scale);
			
			// Run the optimized version
			srand(seed);
			PREFIX(fc_layer_t) o_layer( size, out_size);
			run_layer(o_layer);

			srand(seed);
			fc_layer_t layer(size, out_size);
			run_layer(layer);
			
			// Check for equality.
			EXPECT_EQ(layer, o_layer);
		}
	}
	
	TEST_F(PREFIX(OptimizationTests), rand_fc) {
		rand_fc(32,  64, 1);
		rand_fc(64,  16, 2);
	}
	
	void rand_pool(int scale, int reps, int seed) {
		for (int i = 0; i < reps; i++) {
			
			srand(seed++);
			tdsize size(RAND_LARGE(scale),
				    RAND_LARGE(scale),
				    RAND_LARGE(scale/2));
			int ksize =  RAND_LARGE(scale/10);
			int stride = RAND_LARGE(ksize+1);
			
			// Run the optimized version
			srand(seed);
			PREFIX(pool_layer_t) o_layer(stride, ksize, 0.65, size);
			run_layer(o_layer);

			srand(seed);
			pool_layer_t layer(stride, ksize, 0.65, size);
			run_layer(layer);
			
			// Check for equality.
			EXPECT_EQ(layer, o_layer);
		}
	}
		
	TEST_F(PREFIX(OptimizationTests), rand_pool) {
		rand_pool(32,  64, 1);
		rand_pool(64,  16, 2);
	}

	void rand_relu(int scale, int reps, int seed) {
		for (int i = 0; i < reps; i++) {
				
			srand(seed++);
			tdsize size(RAND_LARGE(scale),
				    RAND_LARGE(scale),
				    RAND_LARGE(scale/2));
		
			// Run the optimized version
			srand(seed);
			PREFIX(relu_layer_t) o_layer( size);
			run_layer(o_layer);

			srand(seed);
			relu_layer_t layer(size);
			run_layer(layer);
			
			// Check for equality.
			EXPECT_EQ(layer, o_layer);
		}
	}
		
	TEST_F(PREFIX(OptimizationTests), rand_relu) {
		rand_relu(32,  64, 1);
		rand_relu(64,  16, 2);
	}

	void rand_softmax(int scale, int reps, int seed) {
		for (int i = 0; i < reps; i++) {
				
			srand(seed++);
			tdsize size(RAND_LARGE(scale),
				    RAND_LARGE(scale),
				    RAND_LARGE(scale/2));
		
			// Run the optimized version
			srand(seed);
			PREFIX(softmax_layer_t) o_layer( size);
			run_layer(o_layer);

			srand(seed);
			softmax_layer_t layer(size);
			run_layer(layer);
			
			// Check for equality.
			EXPECT_EQ(layer, o_layer);
		}
	}
		
	// TEST_F(PREFIX(OptimizationTests), rand_softmax) {
	// 	rand_softmax(32,  64, 1);
	// 	rand_softmax(64,  16, 2);
	// }

	using SimpleCNNTest = CNNTest::SimpleCNNTest;
	TEST_F(SimpleCNNTest, PREFIX(simple_model_opt)) {
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
		PREFIX(conv_layer_t) layer1_o( 1, 5, 8, 0, rand_ds.data_size);
		PREFIX(relu_layer_t) layer2_o( layer1_o.out.size );
		PREFIX(pool_layer_t) layer3_o( 2, 2, 0, layer2_o.out.size );
		PREFIX(fc_layer_t)   layer4_o(layer3_o.out.size, 10);
		
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
