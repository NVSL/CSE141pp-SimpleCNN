#pragma once
#include <sstream>
#include "layer_t.h"

class conv_layer_t: public layer_t
{
public:
	std::vector<tensor_t<float>> filters;  // convolution filter kernels
	std::vector<tensor_t<gradient_t> > filter_grads;
	uint16_t stride;
	uint16_t kernel_size;
	
	conv_layer_t( uint16_t stride,
		      uint16_t kernel_size, // Width and height of the kernel.  This much of the lower-right edges of the input will be ignored.
		      uint16_t kernel_count, // Depth of the output.
		      tdsize in_size )
		:
		layer_t(in_size, tdsize((in_size.x - kernel_size) / stride + 1,
					(in_size.y - kernel_size) / stride + 1,
					kernel_count))
	{
		this->stride = stride;
		this->kernel_size = kernel_size;
		
		// Ensure that stride evenly defivides image size.
		throw_assert( (float( in_size.x - kernel_size ) / stride + 1)
				==
			      ((in_size.x - kernel_size) / stride + 1), "Stride does note divide width");
		
		throw_assert( (float( in_size.y - kernel_size ) / stride + 1)
				==
			      ((in_size.y - kernel_size) / stride + 1), "Stride does not divide height");

		for ( int a = 0; a < kernel_count; a++ ) {	
			tensor_t<float> t( kernel_size, kernel_size, in_size.z );

			int maxval = kernel_size * kernel_size * in_size.z;

			for ( int i = 0; i < kernel_size; i++ )
				for ( int j = 0; j < kernel_size; j++ )
					for ( int z = 0; z < in_size.z; z++ )
						t( i, j, z ) = 1.0f / maxval * rand() / float( RAND_MAX );
			filters.push_back( t );
		}
		for ( int i = 0; i < kernel_count; i++ )
		{
			tensor_t<gradient_t> t( kernel_size, kernel_size, in_size.z );
			filter_grads.push_back( t );
		}

	}

	size_t get_total_memory_size() const {
		size_t sum = 0;
		for(auto & i: filters) {
			sum += i.get_total_memory_size();
		}
		for(auto & i: filter_grads) {
			sum += i.get_total_memory_size();
		}
		return sum + layer_t::get_total_memory_size();
	}

	std::string kind_str() const {
		return "convolution";
	}
	std::string param_str() const {
		std::stringstream ss;
		ss << "stride=" << stride << ", kernel_size=" << kernel_size << ", kernel_count=" << filters.size();
		return ss.str();
	}
	
	bool operator==(const conv_layer_t & o) const {
		if (o.stride != stride) return false;
		if (o.kernel_size != kernel_size) return false;
		if (o.in != in) return false;
		if (o.grads_in != grads_in) return false;
		if (o.out != out) return false;
		if (o.filters != filters) return false;
		if (o.filter_grads != filter_grads) return false;
		return true;
	}

	bool operator!=(const conv_layer_t & o) const {
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
			normalize_range( (a - kernel_size + 1) / stride, out.size.x, true ),
			normalize_range( (b - kernel_size + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)filters.size() - 1,
		};
	}

	void activate(const tensor_t<float>& in ) {
		copy_input(in);
		for ( uint filter = 0; filter < filters.size(); filter++ )
		{
			tensor_t<float>& filter_data = filters[filter];
			for ( int x = 0; x < out.size.x; x++ )
			{
				for ( int y = 0; y < out.size.y; y++ )
				{
					point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
					float sum = 0;
					for ( uint i = 0; i < kernel_size; i++ )
						for ( uint j = 0; j < kernel_size; j++ )
							for ( int z = 0; z < in.size.z; z++ )
							{
								float f = filter_data( i, j, z );
								float v = in( mapped.x + i, mapped.y + j, z );
								sum += f*v;
							}
					out( x, y, filter ) = sum;
				}
			}
		}
	}

	void fix_weights()
	{
		for ( uint a = 0; a < filters.size(); a++ )
			for ( int i = 0; i < kernel_size; i++ )
				for ( int j = 0; j < kernel_size; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						float& w = filters[a].get( i, j, z );
						gradient_t& grad = filter_grads[a].get( i, j, z );
						w = update_weight( w, grad );
						update_gradient( grad );
					}
	}

	void calc_grads( tensor_t<float>& grad_next_layer ) {
		throw_assert(grad_next_layer.size == out.size, "mismatch input size for calc_grads");

		for ( uint k = 0; k < filter_grads.size(); k++ )
		{
			for ( int i = 0; i < kernel_size; i++ )
				for ( int j = 0; j < kernel_size; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						filter_grads[k].get( i, j, z ).grad = 0;
		}

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
						int minx = i * stride;
						for ( int j = rn.min_y; j <= rn.max_y; j++ )
						{
							int miny = j * stride;
							for ( int k = rn.min_z; k <= rn.max_z; k++ )
							{
								int w_applied = filters[k].get( x - minx, y - miny, z );
								sum_error += w_applied * grad_next_layer( i, j, k );
								filter_grads[k].get( x - minx, y - miny, z ).grad += in( x, y, z ) * grad_next_layer( i, j, k );
							}
						}
					}
					grads_in( x, y, z ) = sum_error;
				}
			}
		}
	}
};

class conv_layer_opt_t : public conv_layer_t
{
public:
	conv_layer_opt_t( uint16_t stride,
			  uint16_t kernel_size, 
			  uint16_t kernel_count,
			  tdsize in_size) : conv_layer_t(stride, kernel_size, kernel_count, in_size) {}
			
};

std::ostream& operator<<(std::ostream& os, const conv_layer_t & l)
{
#define DUMP_FIELD(x) #x " = " << l. x << "\n";
	os << DUMP_FIELD(in);
	os << DUMP_FIELD(out);
	for(uint i = 0; i < l.filters.size(); i++) {
		os << "filters[" << i << "] = \n" << l.filters[i];
	}
	for(uint i = 0; i < l.filter_grads.size(); i++) {
		os << "filter_grads[" << i << "] = \n" << l.filter_grads[i];
	}
	os << DUMP_FIELD(stride);
	os << DUMP_FIELD(kernel_size);
	
	return os;
}

       
#ifdef INCLUDE_TESTS
namespace CNNTest{

	void conv_expect_eq(const conv_layer_t & a, const conv_layer_t & b) {
		EXPECT_EQ(a.in, b.in);
		EXPECT_EQ(a.out, b.out);
		for(uint i = 0; i < a.filters.size(); i++) {
			EXPECT_EQ(a.filters[i], b.filters[i]);
		}
		for(uint i = 0; i < a.filter_grads.size(); i++) {
			EXPECT_EQ(a.filter_grads[i], b.filter_grads[i]);
		}
		EXPECT_EQ(a.stride, b.stride);
		EXPECT_EQ(a.kernel_size, b.kernel_size);
	}

	TEST_F(CNNTest, conv_simple) {
		
		tdsize size(10,10,10);
		conv_layer_t t1(1, 4, 5, size);
		conv_layer_t t2(1, 4, 5, size);
		tensor_t<float> in(size);
		randomize(in);
		t1.activate(in);
		EXPECT_EQ(t1,t1);
		EXPECT_NE(t1,t2);

	}
	
	TEST_F(CNNTest, conv_util) {
		srand(42);
		conv_layer_t t1(1, 4, 5, tdsize(10,10,10));
		conv_layer_t t2(1, 4, 5, tdsize(11,10,10));
		conv_layer_t t3(2, 4, 5, tdsize(10,10,10));

		conv_layer_t t4(1, 4, 5, tdsize(10,10,10));
		srand(42);
		conv_layer_t t5(1, 4, 5, tdsize(10,10,10));

		srand(42);
		conv_layer_t t6(1, 1, 1, tdsize(1,1,1));
		srand(42);
		conv_layer_t t7(1, 1, 1, tdsize(1,1,1));

		EXPECT_EQ(t6,t7);
		EXPECT_EQ(t1,t1);
		EXPECT_NE(t1,t2);
		EXPECT_NE(t1,t4); // shouldn't be equal because kernels are random.
		EXPECT_EQ(t1,t5); // should be equal because we set the seed.
		EXPECT_NE(t1,t3);
	}

	conv_layer_t conv_sized(int x, int y, int z, int ksize, int kcount, int stride) {
		
		tdsize size(x,y,z);
		
		tensor_t<float> in(size.x, size.y, size.z);
		tensor_t<float> next_grads((in.size.x - ksize ) / stride + 1,
					   (in.size.y - ksize ) / stride + 1,
					   kcount);
		
		randomize(in);
		randomize(next_grads);

		// Run the optimized version
		srand(42);
		conv_layer_opt_t o_layer( stride, ksize, kcount, in.size);
		o_layer.activate(in);
		o_layer.calc_grads(next_grads);
		o_layer.fix_weights();

		// Run the reference version
		srand(42);
		conv_layer_t layer(stride, ksize, kcount, in.size);
		layer.activate(in);
		layer.calc_grads(next_grads);
		layer.fix_weights();

		// Check for equality.
		EXPECT_EQ(layer, o_layer);
		return layer;
	}

	TEST_F(CNNTest, conv_sizes) {
		// Check a range of sizes, especially non-round numbers.
		conv_sized(4,4,4, 2, 2, 1);

		conv_sized(4,4,4, 2, 2, 1);
		
		conv_sized(1,1,1,1,1,1);
		EXPECT_THROW(conv_sized(1,1,1,7,1,1), AssertionFailureException); // kernel too big
		conv_sized(1,1,1,1,7,1);
		conv_sized(1,1,1,1,1,7);
		EXPECT_THROW(conv_sized(2,1,1,1,1,7), AssertionFailureException); // stride does not divide size
		
		conv_sized(11, 11, 11, 5, 7, 2);
		conv_sized(11, 13, 37, 5, 3, 1);
		conv_sized(32, 32, 32, 5, 3, 1);

		conv_sized(32, 32, 32, 8, 3, 1);
		conv_sized(31, 33, 37, 8, 3, 1);
		//conv_sized(64, 64, 64, 16, 3, 1);
		//conv_sized(128, 128, 128, 4, 1, 1);

	}

}  // namespace
#endif

