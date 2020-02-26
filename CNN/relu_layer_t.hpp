#pragma once
#include "layer_t.hpp"

class relu_layer_t : public layer_t
{
public:
	relu_layer_t(const tdsize & in_size )
		:
		layer_t(in_size, in_size)
	{
	}

	std::string kind_str() const {
		return "relu_layer_t";
	}
	std::string param_str() const {
		std::stringstream ss;
		return ss.str();
	}

	bool operator==(const relu_layer_t & o) const {
		return (o.in == in) && (o.grads_out == grads_out) && (o.out == out);
	}

	bool operator!=(const relu_layer_t & o) const {
		return !(*this == o);
	}
	
	void activate(tensor_t<double>& in ) {
		copy_input(in);
		for (int b = 0; b < in.size.b; b++ )
			for ( int x = 0; x < in.size.x; x++ )
				for ( int y = 0; y < in.size.y; y++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						double v = in( x, y, z, b );
						if ( v < 0 ) {
							v = 0;
						}
						out( x, y, z, b ) = v;
					}
	}

	void fix_weights()
	{

	}

	void calc_grads(const tensor_t<double>& grad_next_layer )
	{
		throw_assert(grad_next_layer.size == in.size, "mismatched input");
		for ( int b = 0; b < in.size.b; b++ )
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						grads_out( i, j, z ) = (in( i, j, z ) < 0) ?
							(0) :
							(grad_next_layer( i, j, z ));
					}

	}
	std::string regression_code() const {
		std::stringstream ss;
		ss << "relu_test<opt_relu_layer_t>("
		   << in.size.x << ", "
		   << in.size.y << ", "
		   << in.size.z << ", "
		   << in.size.b
		   << ", i"
		   << ");";
		return ss.str();
	}
};

template<class T> T* run_relu(int x, int y, int z, int b,
			      int seed) {
	srand(seed);
	tdsize size(x,y,z,b);
	T * l = new T(size);
	l->test_me();
	return l;
}

template<class T> T* run_relu_activate(int x, int y, int z, int b,
				       int seed) {
	srand(seed);
	tdsize size(x,y,z,b);
	T * l = new T(size);
	l->test_activate();
	return l;
}

template<class T> T* run_relu_calc_grads(int x, int y, int z, int b,
					 int seed) {
	srand(seed);
	tdsize size(x,y,z,b);
	T * l = new T(size);
	l->test_calc_grads();
	return l;
}

template<class T> T* run_relu_fix_weights(int x, int y, int z, int b,
					  int seed) {
	srand(seed);
	tdsize size(x,y,z,b);
	T * l = new T(size);
	l->test_fix_weights();
	return l;
}

template<class T>
void relu_test(int x, int y, int z, int b, 
	       int seed) {					
	relu_layer_t * reference = run_relu<relu_layer_t>(x,y,z,b,
							  seed);
	relu_layer_t * optimized = run_relu<T>(x,y,z,b, 
					       seed);
	EXPECT_LAYERS_EQ(relu_layer_t, reference, optimized) << "Failure: relu_test("
							     << x << ", "
							     << y << ", "
							     << z << ", "
							     << b << ", "
							     << seed << ");\n";
	delete reference;					
	delete optimized;
}


template<class T>
void relu_test_activate(int x, int y, int z, int b, 
			int seed) {
	relu_layer_t * reference = run_relu_activate<relu_layer_t>(x,y,z,b,
								   seed);
	relu_layer_t * optimized = run_relu_activate<T>(x,y,z,b,
							seed);
	EXPECT_TENSORS_EQ(double, reference->out, optimized->out) << "Failure: relu_test_activate("
								  << x << ", "
								  << y<< ", "
								  << z<< ", "
								  << b << ", "
								  << seed << ");\n";
	delete reference;					
	delete optimized;
}

template<class T>

void relu_test_calc_grads(int x, int y, int z, int b, 
			  int seed) {
	relu_layer_t * reference = run_relu_calc_grads<relu_layer_t>(x,y,z,b,
								     seed);
	relu_layer_t * optimized = run_relu_calc_grads<T>(x,y,z,b, 
							  seed);
	EXPECT_TENSORS_EQ(double, reference->grads_out, optimized->grads_out) << "Failure: grads_out in relu_test_calc_grads("
									      << x << ", "
									      << y<< ", "
									      << z<< ", "
									      << b << ", "
									      << seed << ");\n";
	delete reference;					
	delete optimized;
}

template<class T>
void relu_test_fix_weights(int x, int y, int z, int b, 
			   int seed) {
	relu_layer_t * reference = run_relu_fix_weights<relu_layer_t>(x,y,y,b,
								      seed);
	relu_layer_t * optimized = run_relu_fix_weights<T>(x,y,z,b, 
							   seed);
	delete reference;					
	delete optimized;
}


#ifdef INCLUDE_TESTS
namespace CNNTest{

	TEST_F(CNNTest, relu_simple) {
		tensor_t<double> data(4,4,4);
		tensor_t<double> expected(data.size);
		tensor_t<double> junk(2,2,2);
		relu_layer_t layer(data.size);
		EXPECT_THROW(layer.activate(junk), AssertionFailureException); // mismatched input size.
		randomize(data);
		layer.activate(data);
		EXPECT_EQ(layer,layer);
		relu_layer_t layer2(data.size);
		randomize(data);
		EXPECT_NE(layer,layer2);
	}
	
	TEST_F(CNNTest, relu_math) {
		tensor_t<double> data(4,4,4);
		tensor_t<double> expected(data.size);
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

		tensor_t<double> grad_next_layer(data.size);
		grad_next_layer(0,0,0) = 0.5;
		grad_next_layer(0,1,1) = 0.5;
		grad_next_layer(2,2,2) = 0.5;
		grad_next_layer(3,0,3) = 0.5;

		layer.calc_grads(grad_next_layer);

		EXPECT_EQ(layer.grads_out(0,0,0), 0.5); 
		EXPECT_EQ(layer.grads_out(0,1,1), 0.5);
		EXPECT_EQ(layer.grads_out(2,2,2), 0);
		EXPECT_EQ(layer.grads_out(3,0,3), 0.5);

	}
	
}  // namespace
#endif
