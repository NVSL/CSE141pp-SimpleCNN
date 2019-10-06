#pragma once
#include "layer_t.h"

class relu_layer_t
{
protected:	
public:
	layer_type type = layer_type::relu;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;

	relu_layer_t(const tdsize & in_size )
		:
		grads_in( in_size.x, in_size.y, in_size.z ),
		in( in_size.x, in_size.y, in_size.z ),
		out( in_size.x, in_size.y, in_size.z )

	{
	}

	bool operator==(const relu_layer_t & o) const {
		return (o.in == in) && (o.grads_in == grads_in) && (o.out == out);
	}

	bool operator!=(const relu_layer_t & o) const {
		return !(*this == o);
	}
	
	void __attribute__((noinline)) activate( const tensor_t<float>& in ) {
		this->in = in;
		activate();
	}

	void __attribute__((noinline)) activate() {
		assert(out.size == in.size);
		for ( int x = 0; x < in.size.x; x++ )
			for ( int y = 0; y < in.size.y; y++ )
				for ( int z = 0; z < in.size.z; z++ )
				{
					float v = in( x, y, z );
					if ( v < 0 ) {
						v = 0;
					}
					out( x, y, z ) = v;
				}
	}

	void fix_weights()
	{

	}

	void __attribute__((noinline)) calc_grads( tensor_t<float>& grad_next_layer )
	{
		assert(grad_next_layer.size == in.size);
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
				{
					grads_in( i, j, z ) = (in( i, j, z ) < 0) ?
						(0) :
						(grad_next_layer( i, j, z ));
				}
	}
};

class relu_layer_opt_t : public relu_layer_t
{
public:
	relu_layer_opt_t(const tdsize & in_size ): relu_layer_t(in_size)
	{
	}

	void __attribute__((noinline)) activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	void __attribute__((noinline)) activate()
	{
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
				{
					float v = in( i, j, z );
					if ( v < 0 )
						v = 0;
					out( i, j, z ) = v;
				}
		
	}
	
	void fix_weights()
	{
		
	}
	
	void __attribute__((noinline)) calc_grads( tensor_t<float>& grad_next_layer )
	{
		for ( int i = 0; i < in.size.x; i++ )
			for ( int j = 0; j < in.size.y; j++ )
				for ( int z = 0; z < in.size.z; z++ )
				{
					grads_in( i, j, z ) = (in( i, j, z ) < 0) ?
						(0) :
						(1 * grad_next_layer( i, j, z ));
				}
	}
	
};


#ifdef INCLUDE_TESTS
namespace CNNTest{

	TEST_F(CNNTest, relu_math) {
		tensor_t<float> data(4,4,4);
		tensor_t<float> expected(data.size);
		data(0,0,0) = 0;
		data(0,1,1) = 1;
		data(2,2,2) = -1;
		data(3,0,3) = 42;

		relu_layer_t layer(data.size);
		layer.activate(data);

		EXPECT_EQ(layer.out(0,0,0), 0);
		EXPECT_EQ(layer.out(0,1,1), 1);
		EXPECT_EQ(layer.out(2,2,2), 0);
		EXPECT_EQ(layer.out(3,0,3), 42);

		tensor_t<float> grad_next_layer(data.size);
		grad_next_layer(0,0,0) = 0.5;
		grad_next_layer(0,1,1) = 0.5;
		grad_next_layer(2,2,2) = 0.5;
		grad_next_layer(3,0,3) = 0.5;

		//std::cout << "grad_next_layer = \n" << grad_next_layer;
		layer.calc_grads(grad_next_layer);
		//std::cout << "layer.in = \n" << layer.in;
		//std::cout << "layer.out = \n" << layer.out;

		// These numbers are all suspicious.  I think the gradient of
		// relu should always be 0 or 1:
		// https://www.quora.com/How-do-we-compute-the-gradient-of-a-ReLU-for-backpropagation
		// the code in this lib multplies 1 * propogated grad, which I
		// think might also be correct.
		EXPECT_EQ(layer.grads_in(0,0,0), 0.5); 
		EXPECT_EQ(layer.grads_in(0,1,1), 0.5);
		EXPECT_EQ(layer.grads_in(2,2,2), 0);
		EXPECT_EQ(layer.grads_in(3,0,3), 0.5);

		//std::cout << "layer.grads_in = \n" << layer.grads_in;
		
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
		EXPECT_EQ(layer.in, o_layer.in);
	}

	TEST_F(CNNTest, relu_sizes) {
		// Check a range of sizes, especially non-round numbers.
		relu_sized(1,1,1);
		relu_sized(16,1,1);
		relu_sized(1,16,1);
		relu_sized(1,1,16);
		relu_sized(2,4,8);
		relu_sized(10,10,10);
		relu_sized(47, 65, 98);
	}
	
}  // namespace
#endif
