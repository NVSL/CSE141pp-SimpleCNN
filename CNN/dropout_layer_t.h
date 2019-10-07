#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct dropout_layer_t
{
	layer_type type = layer_type::dropout_layer;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<bool> hitmap;
	float p_activation;

	dropout_layer_t( tdsize in_size, float p_activation )
		:
		grads_in( in_size.x, in_size.y, in_size.z ),
		in( in_size.x, in_size.y, in_size.z ),
		out( in_size.x, in_size.y, in_size.z ),
		hitmap( in_size.x, in_size.y, in_size.z ),
		p_activation( p_activation )
	{
		throw_assert(p_activation >= 0 && p_activation <= 1.0, "activation level should be betwene 0.0 and 1.0");
	}

	bool operator==(const dropout_layer_t & o) const {
		if (o.p_activation != p_activation) return false;
		if (o.hitmap != hitmap) return false;
		if (o.in != in) return false;
		if (o.grads_in != grads_in) return false;
		if (o.out != out) return false;
		return true;
	}

	bool operator!=(const dropout_layer_t & o) const {
		return !(*this == o);
	}


	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	void __attribute__((noinline)) activate()
	{
		for ( int i = 0; i < in.size.x*in.size.y*in.size.z; i++ )
		{
			bool active = (rand() % RAND_MAX) / float( RAND_MAX ) <= p_activation;
			hitmap.data[i] = active;
			out.data[i] = active ? in.data[i] : 0.0f;
		}
	}


	void fix_weights()
	{
		
	}

	void __attribute__((noinline)) calc_grads( tensor_t<float>& grad_next_layer )
	{
		for ( int i = 0; i < in.size.x*in.size.y*in.size.z; i++ )
			grads_in.data[i] = hitmap.data[i] ? grad_next_layer.data[i] : 0.0f;
	}
};

#ifdef INCLUDE_TESTS
namespace CNNTest{

	TEST_F(CNNTest, dropout_simple) {
		
		tdsize size(10,10,10);
		dropout_layer_t t1(size, 0.5);
		dropout_layer_t t2(size, 0.5);
		tensor_t<float> in(size);
		randomize(in);
		t1.activate(in);
		EXPECT_EQ(t1,t1);
		EXPECT_NE(t1,t2);

	}

	dropout_layer_t dropout_sized(int x, int y, int z, float activation) {
		tdsize size(x,y,z);
		
		tensor_t<float> in(size.x, size.y, size.z);
		tensor_t<float> next_grads(in.size);
		
		randomize(in);
		randomize(next_grads);

		// Run the optimized version
		srand(42);
		dropout_layer_t o_layer(in.size, activation);
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

	TEST_F(CNNTest, dropout_sizes) {
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

}  // namespace
#endif


#pragma pack(pop)
