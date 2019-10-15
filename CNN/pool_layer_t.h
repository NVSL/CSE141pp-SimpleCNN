#pragma once
#include "layer_t.h"


class pool_layer_t: public layer_t
{
public:
	const uint16_t stride;
	const uint16_t filter_size;

	pool_layer_t( uint16_t stride, uint16_t filter_size, tdsize in_size )
		:
		layer_t(in_size, tdsize((in_size.x - filter_size) / stride + 1,
				       (in_size.y - filter_size) / stride + 1,
				       in_size.z)),
		stride(stride),
		filter_size(filter_size)
	{
		throw_assert( (float( in_size.x - filter_size ) / stride + 1)
			      ==
			      ((in_size.x - filter_size) / stride + 1), "Stride doesn't divide input size");

		throw_assert( (float( in_size.y - filter_size ) / stride + 1)
				==
				((in_size.y - filter_size) / stride + 1) , "Stride doesn't divide input size");
	}

	std::string kind_str() const {
		return "pool";
	}
	std::string param_str() const {
		std::stringstream ss;
		ss << "stride=" << stride << ", filter_size=" << filter_size;
		return ss.str();
	}

	bool operator==(const pool_layer_t & o) const {
		if (o.stride != stride) return false;
		if (o.filter_size != filter_size) return false;
		if (o.in != in) return false;
		if (o.grads_in != grads_in) return false;
		if (o.out != out) return false;
		return true;
	}

	bool operator!=(const pool_layer_t & o) const {
		return !(*this == o);
	}


	point_t map_to_input( point_t out, int z )
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	int normalize_range( float f, int max, bool lim_min )
	{
		if ( f <= 0 )
			return 0;
		max -= 1;
		if ( f >= max )
			return max;

		if ( lim_min ) // left side of inequality
			return ceil( f );
		else
			return floor( f );
	}

	range_t map_to_output( int x, int y )
	{
		float a = x;
		float b = y;
		return
		{
			normalize_range( (a - filter_size + 1) / stride, out.size.x, true ),
			normalize_range( (b - filter_size + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)out.size.z - 1,
		};
	}

	void activate(const tensor_t<float>& in ) {
		copy_input(in);
		for ( int x = 0; x < out.size.x; x++ )
		{
			for ( int y = 0; y < out.size.y; y++ )
			{
				for ( int z = 0; z < out.size.z; z++ )
				{
					point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
					float mval = -FLT_MAX;
					for ( int i = 0; i < filter_size; i++ )
						for ( int j = 0; j < filter_size; j++ )
						{
							float v = in( mapped.x + i, mapped.y + j, z );
							if ( v > mval )
								mval = v;
						}
					out( x, y, z ) = mval;
				}
			}
		}
	}

	void fix_weights()
	{

	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		for ( int x = 0; x < in.size.x; x++ )
		{
			for ( int y = 0; y < in.size.y; y++ )
			{
				range_t rn = map_to_output( x, y );
				for ( int z = 0; z < in.size.z; z++ )
				{
					float sum_error = 0;
					for ( int i = rn.min_x; i <= rn.max_x; i++ )
					{
						for ( int j = rn.min_y; j <= rn.max_y; j++ )
						{
							int is_max = in( x, y, z ) == out( i, j, z ) ? 1 : 0;
							sum_error += is_max * grad_next_layer( i, j, z );
						}
					}
					grads_in( x, y, z ) = sum_error;
				}
			}
		}
	}
};

class pool_layer_opt_t: public pool_layer_t
{
public:
	pool_layer_opt_t( uint16_t stride, uint16_t filter_size, tdsize in_size ) : pool_layer_t(stride, filter_size, in_size) {}
};

#ifdef INCLUDE_TESTS
namespace CNNTest{

	TEST_F(CNNTest, pool_simple) {
		
		tdsize size(10,10,10);
		pool_layer_t t1(1, 4, size);
		pool_layer_t t2(1, 4, size);
		tensor_t<float> in(size);
		randomize(in);
		t1.activate(in);
		EXPECT_EQ(t1,t1);
		EXPECT_NE(t1,t2);

	}

	pool_layer_t pool_sized(int x, int y, int z, int ksize, int stride) {
		tdsize size(x,y,z);
		
		tensor_t<float> in(size.x, size.y, size.z);
		tensor_t<float> next_grads((in.size.x - ksize ) / stride + 1,
					   (in.size.y - ksize ) / stride + 1,
					   in.size.z);
		
		randomize(in);
		randomize(next_grads);

		// Run the optimized version
		srand(42);
		pool_layer_opt_t o_layer( stride, ksize, in.size);
		o_layer.activate(in);
		o_layer.calc_grads(next_grads);
		o_layer.fix_weights();
		
		// Run the reference version
		srand(42);
		pool_layer_t layer(stride, ksize, in.size);
		layer.activate(in);
		layer.calc_grads(next_grads);
		layer.fix_weights();

		// Check for equality.
		EXPECT_EQ(layer, o_layer);
		return layer;
	}

	TEST_F(CNNTest, pool_sizes) {
		// Check a range of sizes, especially non-round numbers.
		pool_sized(4, 4, 4, 2, 1);

		pool_sized(1, 1, 1, 1, 1);
		EXPECT_THROW(pool_sized(1,1,1,7,1), AssertionFailureException); // kernel too big
		pool_sized(1,1,1,1,7);
		EXPECT_THROW(pool_sized(2,1,1,1,7), AssertionFailureException); // stride does not divide size
		
		pool_sized(11, 11, 11, 5, 2);
		pool_sized(11, 13, 37, 5, 1);
		pool_sized(32, 32, 32, 5, 1);

		pool_sized(32, 32, 32, 8, 1);
		pool_sized(31, 33, 37, 8, 1);
	}

}  // namespace
#endif

