
namespace PREFIX(Tests) {
	
	class PREFIX(OptimizationTests) :  public ::testing::Test {

	};

	conv_layer_t conv_sized(int x, int y, int z, int ksize, int kcount, int stride) {
		
		tdsize size(x,y,z);
		
		tensor_t<float> in(size.x, size.y, size.z);
		tensor_t<float> next_grads(ROUND_UP_IDIV(in.size.x, stride),
					   ROUND_UP_IDIV(in.size.y, stride),
					   kcount);
		
		randomize(in);
		randomize(next_grads);

		// Run the optimized version
		srand(42);
		conv_layer_t o_layer( stride, ksize, kcount, 0,in.size);
		o_layer.activate(in);
		o_layer.calc_grads(next_grads);
		o_layer.fix_weights();

		// Run the reference version
		srand(42);
		::conv_layer_t layer(stride, ksize, kcount, 0,in.size);
		layer.activate(in);
		layer.calc_grads(next_grads);
		layer.fix_weights();

		// Check for equality.
		EXPECT_EQ(layer, o_layer);
		return layer;
	}

	TEST_F(PREFIX(OptimizationTests), convolution) {
		// Check a range of sizes, especially non-round numbers.
		conv_sized(4,4,4, 2, 2, 1);

		conv_sized(4,4,4, 2, 2, 1);
		
		conv_sized(1,1,1,1,1,1);
//		EXPECT_THROW(conv_sized(1,1,1,7,1,1), AssertionFailureException); // kernel too big
		conv_sized(1,1,1,1,7,1);
		//conv_sized(1,1,1,1,1,7);
		//EXPECT_THROW(conv_sized(2,1,1,1,1,7), AssertionFailureException); // stride does not divide size
		//conv_sized(2,1,1, 1, 1, 7);
				
		conv_sized(11, 11, 11, 5, 7, 2);
		conv_sized(11, 13, 37, 5, 3, 1);
		conv_sized(32, 32, 32, 5, 3, 1);

		conv_sized(32, 32, 32, 8, 3, 1);
		conv_sized(31, 33, 37, 8, 3, 1);
		//conv_sized(64, 64, 64, 16, 3, 1);
		//conv_sized(128, 128, 128, 4, 1, 1);

	}

	dropout_layer_t dropout_sized(int x, int y, int z, float activation) {
		tdsize size(x,y,z);
		
		tensor_t<float> in(size.x, size.y, size.z);
		tensor_t<float> next_grads(in.size);
		
		randomize(in);
		randomize(next_grads);

		// Run the optimized version
		srand(42);
		dropout_layer_opt_t o_layer(in.size, activation);
		o_layer.activate(in);
		o_layer.calc_grads(next_grads);
		o_layer.fix_weights();
		
		// Run the reference version
		srand(42);
		dropout_layer_t layer(in.size, activation);
		layer.activate(in);
		layer.calc_grads(next_grads);
		layer.fix_weights();

		// Check for equality.
		EXPECT_EQ(layer, o_layer);
		return layer;
	}

	TEST_F(PREFIX(OptimizationTests), dropout_sizes) {
		// Check a range of sizes, especially non-round numbers.
		dropout_sized(4, 4, 4, 0.5);

		dropout_sized(1, 1, 1, 0.5);
		dropout_sized(10, 10, 10, 0.0);
		dropout_sized(10, 10, 10, 1.0);
		
		dropout_sized(11, 11, 11, 0.5);
		dropout_sized(11, 13, 37, 0.5);
		dropout_sized(32, 32, 32, 0.5);

		dropout_sized(32, 32, 32, 0.5);
		dropout_sized(31, 33, 37, 0.5);
	}

	fc_layer_t fc_sized(int in_x, int in_y, int in_z, int out_size) {
		tdsize in_size(in_x, in_y, in_z);
		
		tensor_t<float> in(in_size);
		tensor_t<float> next_grads(out_size,1,1);
		
		randomize(in);
		randomize(next_grads);

		// Run the optimized version
		srand(42);
		fc_layer_t o_layer(in.size, out_size);
		o_layer.activate(in);
		o_layer.calc_grads(next_grads);
		o_layer.fix_weights();
		
		// Run the reference version
		srand(42);
		fc_layer_opt_t layer(in.size, out_size);
		layer.activate(in);
		layer.calc_grads(next_grads);
		layer.fix_weights();

		// Check for equality.
		EXPECT_EQ(layer, o_layer);
		return layer;
	}

	TEST_F(PREFIX(OptimizationTests), fc_sizes) {
		// Check a range of sizes, especially non-round numbers.
		fc_sized(4, 4, 4, 4);

		fc_sized(1, 1, 1, 1);
		
		fc_sized(10, 10, 10, 1000);
		fc_sized(10, 10, 10, 10); 
		
		fc_sized(11, 11, 11, 13);
		fc_sized(11, 13, 37, 61);
		fc_sized(32, 32, 32, 91);
	}


	pool_layer_t pool_sized(int x, int y, int z, int ksize, int stride) {
		tdsize size(x,y,z);
		
		tensor_t<float> in(size.x, size.y, size.z);
		tensor_t<float> next_grads(ROUND_UP_IDIV(in.size.x, stride),
					   ROUND_UP_IDIV(in.size.y, stride),
					   in.size.z);
		
		randomize(in);
		randomize(next_grads);

		// Run the optimized version
		srand(42);
		pool_layer_opt_t o_layer( stride, ksize, 0, in.size);
		o_layer.activate(in);
		o_layer.calc_grads(next_grads);
		o_layer.fix_weights();
		
		// Run the reference version
		srand(42);
		pool_layer_t layer(stride, ksize, 0, in.size);
		layer.activate(in);
		layer.calc_grads(next_grads);
		layer.fix_weights();

		// Check for equality.
		EXPECT_EQ(layer, o_layer);
		return layer;
	}

	TEST_F(PREFIX(OptimizationTests), pool_sizes) {
		// Check a range of sizes, especially non-round numbers.
		pool_sized(4, 4, 4, 2, 1);

		pool_sized(1, 1, 1, 1, 1);
		//EXPECT_THROW(pool_sized(1,1,1,7,1), AssertionFailureException); // kernel too big
		//pool_sized(1,1,1,1,7);
		//EXPECT_THROW(pool_sized(2,1,1,1,7), AssertionFailureException); // stride does not divide size
		
		pool_sized(11, 11, 11, 5, 2);
		pool_sized(13, 11, 37, 5, 1);
		pool_sized(32, 32, 32, 5, 1);

		pool_sized(32, 32, 32, 8, 1);
		pool_sized(33, 31, 37, 8, 1);
	}

	
	void relu_sized(int x, int y, int z) {
		tdsize size(x,y,z);
		tensor_t<float> in(size.x, size.y, size.z);
		tensor_t<float> next_grads(size.x, size.y, size.z);

		randomize(in);
		randomize(next_grads);

		// Run the optimized version
		relu_layer_opt_t o_layer(in.size);
		o_layer.activate(in);
		o_layer.calc_grads(next_grads);
		o_layer.fix_weights();

		// Run the reference version
		relu_layer_t layer(in.size);
		layer.activate(in);
		layer.calc_grads(next_grads);
		layer.fix_weights();

		// Check for equality.
		EXPECT_EQ(layer, o_layer);
	}

	TEST_F(PREFIX(OptimizationTests), relu_sizes) {
		// Check a range of sizes, especially non-round numbers.
		relu_sized(1,1,1);
		relu_sized(16,1,1);
		relu_sized(1,16,1);
		relu_sized(1,1,16);
		relu_sized(2,4,8);
		relu_sized(10,10,10);
		relu_sized(47, 65, 98);
	}


	void softmax_sized(int x) {
		tensor_t<float> in(x, 1,1);
		tensor_t<float> next_grads(x,1,1);

		randomize(in);
		randomize(next_grads);

		// Run the optimized version
		softmax_layer_opt_t o_layer(in.size);
		o_layer.activate(in);
		o_layer.calc_grads(next_grads);
		o_layer.fix_weights();

		// Run the reference version
		softmax_layer_t layer(in.size);
		layer.activate(in);
		layer.calc_grads(next_grads);
		layer.fix_weights();

		// Check for equality.
		EXPECT_EQ(layer, o_layer);
	}

	TEST_F(PREFIX(OptimizationTests), softmax_sizes) {
		// Check a range of sizes, especially non-round numbers.
		softmax_sized(1);
		softmax_sized(16);
		softmax_sized(47);
	}

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
