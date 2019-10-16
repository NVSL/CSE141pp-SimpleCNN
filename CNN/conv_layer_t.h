#pragma once
#include <sstream>
#include "layer_t.h"
#include "range_t.h"

class conv_layer_t: public layer_t
{
public:
	std::vector<tensor_t<float>> filters;  // convolution filter kernels
	std::vector<tensor_t<gradient_t> > filter_grads;
	uint16_t stride;
	uint16_t kernel_size;
	float pad;
	
	conv_layer_t( uint16_t stride,
		      uint16_t kernel_size, // Width and height of the kernel.  This much of the lower-right edges of the input will be ignored.
		      uint16_t kernel_count, // Depth of the output.
		      float pad,
		      tdsize in_size
		)
		:
		layer_t(in_size, tdsize(ROUND_UP_IDIV(in_size.x, stride),
					ROUND_UP_IDIV(in_size.y, stride),
					kernel_count)),
		pad(pad)
		
	{
		this->stride = stride;
		this->kernel_size = kernel_size;
		throw_assert(kernel_size >= stride, "Convolution kernel size (" << kernel_size << ") must be >= than stride (" << stride << ").");
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
		ss << "stride=" << stride << ", kernel_size=" << kernel_size << ", kernel_count=" << filters.size() << ", pad=" << pad;
		return ss.str();
	}

	std::string internal_state() const {
		std::stringstream ss;
		int i= 0;
		for(auto &k: filters) {
			ss << "Kernel " << i++ << "\n";
			ss << k << "\n";
		}
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
		if (o.pad != pad) return false;
		return true;
	}

	bool operator!=(const conv_layer_t & o) const {
		return !(*this == o);
	}

	range_t map_to_output( int x, int y )
	{
		return map_to_output_impl(x,y, kernel_size, stride, filters.size(), out.size);
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
					point_t mapped(x*stride, y*stride, 0);
					float sum = 0;
					for ( int i = 0; i < kernel_size; i++ )
						for ( int j = 0; j < kernel_size; j++ )
							for ( int z = 0; z < in.size.z; z++ )
							{
								float f = filter_data( i, j, z );
								
								float v;
								if (mapped.x + i >= in.size.x ||
								    mapped.y + j >= in.size.y) {
									v = pad;
								} else {
									v = in( mapped.x + i, mapped.y + j, z );
								}
								sum += f*v;
							}
					out( x, y, filter ) = sum;
				}
			}
		}
	}

	void fix_weights() {
		for ( uint a = 0; a < filters.size(); a++ )
			for ( int i = 0; i < kernel_size; i++ )
				for ( int j = 0; j < kernel_size; j++ )
					for ( int z = 0; z < in.size.z; z++ ) {
						float& w = filters[a].get( i, j, z );
						gradient_t& grad = filter_grads[a].get( i, j, z );
						w = update_weight( w, grad );
						update_gradient( grad );
					}
	}

	void calc_grads( tensor_t<float>& grad_next_layer ) {
		throw_assert(grad_next_layer.size == out.size, "mismatch input size for calc_grads");

		for ( uint k = 0; k < filter_grads.size(); k++ ) 
			for ( int i = 0; i < kernel_size; i++ )
				for ( int j = 0; j < kernel_size; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						filter_grads[k].get( i, j, z ).grad = 0;
		
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
			  float pad,
			  tdsize in_size
			  ) : conv_layer_t(stride, kernel_size, kernel_count, pad, in_size) {}
			
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
		conv_layer_t t1(2, 4, 5, 0, size);
		conv_layer_t t2(2, 4, 5, 0, size);
		tensor_t<float> in(size);
		randomize(in);
		t1.activate(in);
		EXPECT_EQ(t1,t1);
		EXPECT_NE(t1,t2);
	}

	TEST_F(CNNTest, conv_slight_overlap) {
 		conv_layer_t t3(4, 5, 1, 0, tdsize(17,17,1));
		EXPECT_EQ(t3.out.size.x, 5);
			
		auto r1 =  t3.map_to_output(0,0);
		EXPECT_EQ(r1.min_x, 0);
		EXPECT_EQ(r1.max_x, 0);
		EXPECT_EQ(r1.max_z, t3.filters.size()-1);
		      
		auto r2 =  t3.map_to_output(1,1);
		EXPECT_EQ(r2.min_x, 0);
		EXPECT_EQ(r2.max_x, 0);
		
		auto r3 =  t3.map_to_output(3,3);
		EXPECT_EQ(r3.min_x, 0);
		EXPECT_EQ(r3.max_x, 0);
		
		auto r4 =  t3.map_to_output(4,4);
		EXPECT_EQ(r4.min_x, 0);
		EXPECT_EQ(r4.max_x, 1);
		
		auto r5 =  t3.map_to_output(8,8);
		EXPECT_EQ(r5.min_x, 1);
		EXPECT_EQ(r5.max_x, 2);
		
		auto r6 =  t3.map_to_output(16,16);
		EXPECT_EQ(r6.min_x, 3);
		EXPECT_EQ(r6.max_x, 4);
		
	}

	TEST_F(CNNTest, conv_big_overlap) {
		conv_layer_t t3(1, 5, 1, 0, tdsize(17,17,1));
		EXPECT_EQ(t3.out.size.x, 17);
			
		auto r1 =  t3.map_to_output(0,0);
		EXPECT_EQ(r1.min_x, 0);
		EXPECT_EQ(r1.max_x, 0);
		
		auto r2 =  t3.map_to_output(1,1);
		EXPECT_EQ(r2.min_x, 0);
		EXPECT_EQ(r2.max_x, 1);
		
		auto r3 =  t3.map_to_output(4,4);
		EXPECT_EQ(r3.min_x, 0);
		EXPECT_EQ(r3.max_x, 4);
		
		auto r4 =  t3.map_to_output(5,5);
		EXPECT_EQ(r4.min_x, 1);
		EXPECT_EQ(r4.max_x, 5);
		
		auto r5 =  t3.map_to_output(6,6);
		EXPECT_EQ(r5.min_x, 2);
		EXPECT_EQ(r5.max_x, 6);
		
		auto r6 =  t3.map_to_output(16,16);
		EXPECT_EQ(r6.min_x, 12);
		EXPECT_EQ(r6.max_x, 16);
		
	}

	TEST_F(CNNTest, conv_mid_overlap) {
		conv_layer_t t3(2, 4, 1, 0, tdsize(17,17,1));
		EXPECT_EQ(t3.out.size.x, 9);
			
		EXPECT_EQ(t3.map_to_output(0, 0).min_x, 0);
		EXPECT_EQ(t3.map_to_output(0, 0).max_x, 0);

		EXPECT_EQ(t3.map_to_output(1, 1).min_x, 0);
		EXPECT_EQ(t3.map_to_output(1, 1).max_x, 0);
		EXPECT_EQ(t3.map_to_output(1, 1).min_y, 0);
		EXPECT_EQ(t3.map_to_output(1, 1).max_y, 0);
		
		EXPECT_EQ(t3.map_to_output(2, 2).min_x, 0);
		EXPECT_EQ(t3.map_to_output(2, 2).max_x, 1);
		
		EXPECT_EQ(t3.map_to_output(3, 3).min_x, 0);
		EXPECT_EQ(t3.map_to_output(3, 3).max_x, 1);
		
		EXPECT_EQ(t3.map_to_output(4, 4).min_x, 1);
		EXPECT_EQ(t3.map_to_output(4, 4).max_x, 2);
		EXPECT_EQ(t3.map_to_output(4, 4).min_y, 1);
		EXPECT_EQ(t3.map_to_output(4, 4).max_y, 2);
		
		EXPECT_EQ(t3.map_to_output(16, 16).min_x, 7);
		EXPECT_EQ(t3.map_to_output(16, 16).max_x, 8);
		
	}

	TEST_F(CNNTest, conv_gap) {
		EXPECT_THROW(conv_layer_t(4, 2, 1, 0, tdsize(17,17,1)), AssertionFailureException); 
	}
	
	TEST_F(CNNTest, conv_util) {
		srand(42);
		conv_layer_t t1(1, 4, 5, 0,tdsize(10,10,10));
		conv_layer_t t2(1, 4, 5, 0,tdsize(11,10,10));
		conv_layer_t t3(2, 4, 5, 0,tdsize(10,10,10));

		conv_layer_t t4(1, 4, 5, 0,tdsize(10,10,10));
		srand(42);
		conv_layer_t t5(1, 4, 5, 0,tdsize(10,10,10));

		srand(42);
		conv_layer_t t6(1, 1, 1, 0,tdsize(1,1,1));
		srand(42);
		conv_layer_t t7(1, 1, 1, 0,tdsize(1,1,1));

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
		tensor_t<float> next_grads(ROUND_UP_IDIV(in.size.x, stride),
					   ROUND_UP_IDIV(in.size.y, stride),
					   kcount);
		
		randomize(in);
		randomize(next_grads);

		// Run the optimized version
		srand(42);
		conv_layer_opt_t o_layer( stride, ksize, kcount, 0,in.size);
		o_layer.activate(in);
		o_layer.calc_grads(next_grads);
		o_layer.fix_weights();

		// Run the reference version
		srand(42);
		conv_layer_t layer(stride, ksize, kcount, 0,in.size);
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

}  // namespace
#endif

