#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"

class fc_layer_t: public layer_t
{
public:
	std::vector<float> input;
	tensor_t<float> weights;
	std::vector<gradient_t> gradients;

	fc_layer_t( tdsize in_size, int out_size )
		:
		layer_t(in_size, tdsize(out_size, 1, 1)),
		input(out_size),
		weights( in_size.x*in_size.y*in_size.z, out_size, 1 ),
		gradients(out_size)
		{
//		input = std::vector<float>( out_size );
			//gradients = std::vector<gradient_t>( out_size );

			int maxval = in_size.x * in_size.y * in_size.z;

			for ( int i = 0; i < out_size; i++ )
				for ( int h = 0; h < in_size.x*in_size.y*in_size.z; h++ )
					weights( h, i, 0 ) = 2.19722f / maxval * rand() / float( RAND_MAX );
			// 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
		}

	size_t get_total_memory_size() const {
		return weights.get_total_memory_size() +
			gradients.size() * sizeof(gradient_t) +
			input.size() * sizeof(float) +
			layer_t::get_total_memory_size();
	}

	std::string kind_str() const {
		return "fc";
	}
	std::string param_str() const {
		std::stringstream ss;
		return ss.str();
	}

	bool operator==(const fc_layer_t & o) const {
		if (o.weights != weights) return false;
		if (o.in != in) return false;
		if (o.grads_in != grads_in) return false;
		if (o.out != out) return false;
		return true;
	}

	bool operator!=(const fc_layer_t & o) const {
		return !(*this == o);
	}


	float activator_function( float x ) {
		//return tanhf( x );
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig;
	}

	float activator_derivative( float x ) {
		//float t = tanhf( x );
		//return 1 - t * t;
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig * (1 - sig);
	}

	int map( point_t d ) {
		return d.z * (in.size.x * in.size.y) +
			d.y * (in.size.x) +
			d.x;
	}

	void activate(const tensor_t<float>& in ) {
		copy_input(in);
		for ( int n = 0; n < out.size.x; n++ )
		{
			float inputv = 0;

			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map( { i, j, z } );
						inputv += in( i, j, z ) * weights( m, n, 0 );
					}

			input[n] = inputv;

			out( n, 0, 0 ) = activator_function( inputv );
		}
	}

	void fix_weights() {
		for ( int n = 0; n < out.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map( { i, j, z } );
						float& w = weights( m, n, 0 );
						w = update_weight( w, grad, in( i, j, z ) );
					}

			update_gradient( grad );
		}
	}

	void calc_grads( const tensor_t<float>& grad_next_layer ) {
		memset( grads_in.data, 0, grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) );
		for ( int n = 0; n < out.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			grad.grad = grad_next_layer( n, 0, 0 ) * activator_derivative( input[n] );

			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map( { i, j, z } );
						grads_in( i, j, z ) += grad.grad * weights( m, n, 0 );
					}
		}
	}
};

class fc_layer_opt_t : public fc_layer_t
{
public:
	fc_layer_opt_t( tdsize in_size, int out_size ) : fc_layer_t(in_size, out_size) {}
			
};

#ifdef INCLUDE_TESTS
namespace CNNTest{

	TEST_F(CNNTest, fc_simple) {
		
		tdsize in_size(10,10,10);
		int  out_size = 5;
		fc_layer_t t1(in_size, out_size);
		fc_layer_t t2(in_size, out_size);
		tensor_t<float> in(in_size);
		randomize(in);
		t1.activate(in);
		EXPECT_EQ(t1,t1);
		EXPECT_NE(t1,t2);

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

	TEST_F(CNNTest, fc_sizes) {
		// Check a range of sizes, especially non-round numbers.
		fc_sized(4, 4, 4, 4);

		fc_sized(1, 1, 1, 1);
		
		fc_sized(10, 10, 10, 1000);
		fc_sized(10, 10, 10, 10); 
		
		fc_sized(11, 11, 11, 13);
		fc_sized(11, 13, 37, 61);
		fc_sized(32, 32, 32, 91);
	}

}  // namespace
#endif


	
