#pragma once
#include "layer_t.hpp"
#include "range_t.hpp"

class pool_layer_t: public layer_t
{
public:
	const uint16_t stride;
	const uint16_t filter_size;
	double pad;
	pool_layer_t( uint16_t stride, uint16_t filter_size, double pad, tdsize in_size )
		:
		layer_t(in_size, tdsize(ROUND_UP_IDIV(in_size.x, stride),
					ROUND_UP_IDIV(in_size.y, stride),
					in_size.z, in_size.b)),
		stride(stride),
		filter_size(filter_size),
		pad(pad)
	{
		throw_assert(filter_size >= stride, "Pool filter size (" << filter_size << ") must be >= stride (" << stride << ").");
	}

	std::string kind_str() const {
		return "pool_layer_t";
	}
	std::string param_str() const {
		std::stringstream ss;
		ss << "stride=" << stride << ", filter_size=" << filter_size  << ", pad=" << pad;
		return ss.str();
	}

	bool operator==(const pool_layer_t & o) const {
		if (o.stride != stride) return false;
		if (o.filter_size != filter_size) return false;
		if (o.in != in) return false;
		if (o.grads_out != grads_out) return false;
		if (o.out != out) return false;
		return true;
	}

	bool operator!=(const pool_layer_t & o) const {
		return !(*this == o);
	}

	range_t map_to_output( int x, int y )
	{
		return map_to_output_impl(x, y, filter_size, stride, out.size.z, out.size);
	}

	void activate(tensor_t<double>& in ) {
		copy_input(in);
		for ( int b = 0; b < out.size.b; b++ ) {
			for ( int x = 0; x < out.size.x; x++ ) {
				for ( int y = 0; y < out.size.y; y++ ) {
					for ( int z = 0; z < out.size.z; z++ ) {
						point_t mapped(x*stride, y*stride, 0);
						double mval = -FLT_MAX;
						for ( int i = 0; i < filter_size; i++ )
							for ( int j = 0; j < filter_size; j++ ) {
								double v;
								if (mapped.x + i >= in.size.x ||
							    	mapped.y + j >= in.size.y) {
									v = pad;
								} else {
									v = in( mapped.x + i, mapped.y + j, z );
								}

								if ( v > mval )
									mval = v;
							}
						out( x, y, z, b ) = mval;
					}
				}
			}
		}
	}

	void fix_weights()
	{

	}

	void calc_grads(const tensor_t<double>& grad_next_layer )
	{
		for ( int b = 0; b < in.size.b; b++ ) {
			for ( int x = 0; x < in.size.x; x++ ) {
				for ( int y = 0; y < in.size.y; y++ ) {
					range_t rn = map_to_output( x, y );
					for ( int z = 0; z < in.size.z; z++ ) {
						double sum_error = 0;
						for ( int i = rn.min_x; i <= rn.max_x; i++ ) {
							for ( int j = rn.min_y; j <= rn.max_y; j++ ) {
								int is_max = in( x, y, z ) == out( i, j, z ) ? 1 : 0;
								sum_error += is_max * grad_next_layer( i, j, z );
							}
						}
						grads_out( x, y, z, b ) = sum_error;
					}
				}
			}
		}
	}
	std::string regression_code() const {
		std::stringstream ss;
		ss << "pool_test<opt_pool_layer_t>("
		   << in.size.x << ", "
		   << in.size.y << ", "
		   << in.size.z << ", "
		   << in.size.b << ", "
		   << stride << ", "
		   << filter_size << ", "
		   << pad << ", i"
		   << ");";
		return ss.str();
	}
};

template<class T> T* run_pool(int x, int y, int z, int b, uint16_t stride, uint16_t kernel_size, double pad,
			      int seed) {
	srand(seed);
	tdsize size(x,y,z,b);
	T * l = new T(stride, kernel_size, pad, size);
	l->test_me();
	return l;
}

template<class T> T* run_pool_activate(int x, int y, int z, int b, uint16_t stride, uint16_t kernel_size, double pad,
				       int seed) {
	srand(seed);
	tdsize size(x,y,z,b);
	T * l = new T(stride, kernel_size, pad, size);
	l->test_activate();
	return l;
}

template<class T> T* run_pool_calc_grads(int x, int y, int z, int b, uint16_t stride, uint16_t kernel_size, double pad,
					 int seed) {
	srand(seed);
	tdsize size(x,y,z,b);
	T * l = new T(stride, kernel_size, pad, size);
	l->test_calc_grads();
	return l;
}

template<class T> T* run_pool_fix_weights(int x, int y, int z, int b, uint16_t stride, uint16_t kernel_size, double pad,
					  int seed) {
	srand(seed);
	tdsize size(x,y,z,b);
	T * l = new T(stride, kernel_size, pad, size);
	l->test_fix_weights();
	return l;
}

template<class T>
void pool_test(int x, int y, int z, int b, uint16_t stride, uint16_t kernel_size, double pad, int seed) {					
	pool_layer_t * reference = run_pool<pool_layer_t>(x,y,z,b, stride, kernel_size, pad,seed);
	pool_layer_t * optimized = run_pool<T>(x,y,z,b, stride, kernel_size, pad, seed);
	EXPECT_LAYERS_EQ(pool_layer_t, reference, optimized) << "Failure: pool_test("
							     << x << ", "
							     << y << ", "
							     << z << ", "
							     << b << ", "
							     << stride << ", "
							     << kernel_size << ", "
							     << pad << ", "
							     << seed << ");\n";
	delete reference;					
	delete optimized;
}


template<class T>
void pool_test_activate(int x, int y, int z, int b, uint16_t stride, uint16_t kernel_size, double pad, int seed) {
	pool_layer_t * reference = run_pool_activate<pool_layer_t>(x,y,z,b, stride, kernel_size, pad, seed);
	pool_layer_t * optimized = run_pool_activate<T>(x,y,z,b, stride, kernel_size, pad, seed);
	EXPECT_TENSORS_EQ(double, reference->out, optimized->out) << "Failure: pool_test_activate("
								  << x << ", "
								  << y<< ", "
								  << z<< ", "
								  << b << ", "
								  << stride << ", "
								  << kernel_size << ", "
								  << pad << ", "
								  << seed << ");\n";
	delete reference;					
	delete optimized;
}

template<class T>

void pool_test_calc_grads(int x, int y, int z, int b, uint16_t stride, uint16_t kernel_size, double pad, int seed) {
	pool_layer_t * reference = run_pool_calc_grads<pool_layer_t>(x,y,z,b, stride, kernel_size, pad, seed);
	pool_layer_t * optimized = run_pool_calc_grads<T>(x,y,z,b, stride, kernel_size, pad, seed);
	EXPECT_TENSORS_EQ(double, reference->grads_out, optimized->grads_out) << "Failure: grads_out in pool_test_calc_grads("
									      << x << ", "
									      << y<< ", "
									      << z<< ", "
									      << b << ", "
									      << stride << ", "
									      << kernel_size << ", "
									      << pad << ", "
									      << seed << ");\n";
	delete reference;					
	delete optimized;
}

template<class T>
void pool_test_fix_weights(int x, int y, int z, int b, uint16_t stride, uint16_t kernel_size, double pad, int seed) {
	pool_layer_t * reference = run_pool_fix_weights<pool_layer_t>(x,y,y,b, stride, kernel_size, pad, seed);
	pool_layer_t * optimized = run_pool_fix_weights<T>(x,y,z,b, stride, kernel_size, pad, seed);
	delete reference;					
	delete optimized;
}

#ifdef INCLUDE_TESTS
namespace CNNTest{

	TEST_F(CNNTest, pool_simple) {
		
		tdsize size(10,10,10);
		pool_layer_t t1(2, 4, 0, size);
		pool_layer_t t2(2, 4, 0, size);
		tensor_t<double> in(size);
		randomize(in);
		t1.activate(in);
		EXPECT_EQ(t1,t1);
		EXPECT_NE(t1,t2);

		pool_layer_t t3(4, 5, 0, tdsize(17,17,1));
		EXPECT_EQ(t3.out.size.x, 5);

		auto r1 =  t3.map_to_output(0,0);
		EXPECT_EQ(r1.min_x, 0);
		EXPECT_EQ(r1.max_x, 0);
		EXPECT_EQ(r1.min_y, 0);
		EXPECT_EQ(r1.max_y, 0);
		EXPECT_EQ(r1.max_z, t3.out.size.z - 1);

		
	}


}  // namespace
#endif

