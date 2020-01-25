#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.hpp"

class fc_layer_t: public layer_t
{
public:
	std::vector<float> activator_input; // Output the sum-the-weights stage.. 
	tensor_t<float> weights; // 2d array of weight (tensor with depth == 1)
	std::vector<gradient_t> gradients; // gradients for back prop.

	fc_layer_t( tdsize in_size, int out_size )
		:
		layer_t(in_size, tdsize(out_size, 1, 1)),
		activator_input(out_size),
		weights( in_size.x*in_size.y*in_size.z, out_size, 1 ),
		gradients(out_size)
		{
			int maxval = in_size.x * in_size.y * in_size.z;

			for ( int i = 0; i < out_size; i++ )
				for ( int h = 0; h < in_size.x*in_size.y*in_size.z; h++ )
					weights( h, i, 0 ) = 2.19722f / maxval * rand() / float( RAND_MAX );
			// 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
		}

	size_t get_total_memory_size() const {
		return weights.get_total_memory_size() +
			gradients.size() * sizeof(gradient_t) +
			activator_input.size() * sizeof(float) +
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
		if (o.grads_out != grads_out) return false;
		if (o.out != out) return false;
		return true;
	}

	bool operator!=(const fc_layer_t & o) const {
		return !(*this == o);
	}


	float activator_function( float x ) {
		// THis is the logistic function.  Detail here: https://en.wikipedia.org/wiki/Logistic_function#Derivative
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig;
	}

	float activator_derivative( float x ) {
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig * (1 - sig);
	}

	int map(int x, int y, int z) {
		return z * (in.size.x * in.size.y) +
			y * (in.size.x) +
			x;
	}

	void activate(const tensor_t<float>& in ) {
		copy_input(in);
		// Here's the math we are doing here:
		//
		// We'll use some short hand:
		//
		// x = in
		// w = weights
		// L = activator_function  
		// f(x,w) = x*w
		//
		// This layer is a function, F, that maps `x` to `out`
		//
		// out = F(x,w) = L(f(x,w))
		// 
		// f(x,w) = x*w is vector-matrix product which yields a vector.
		// We apply L to each element to get the output vector.

		for ( int n = 0; n < out.size.x; n++ )
		{
			float inputv = 0;

			// Both `in` and `weights` are tensors instead
			// of vectors in this code.

			// compute the dot product, we "straighten"
			// them using map() - it converts x,y,z into a
			// linear index for the out vector.
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map(i, j, z);
						inputv += in( i, j, z ) * weights( m, n, 0 );
					}

			activator_input[n] = inputv;
			// Apply L
			out( n, 0, 0 ) = activator_function( inputv );
		}
	}

	void calc_grads( const tensor_t<float>& grad_next_layer ) {
		
		memset( grads_out.data, 0, grads_out.size.x * grads_out.size.y * grads_out.size.z * sizeof( float ) );

		// Using the notation from activate():
		//
		// We do two things in the loop below: 1) compute the
		// gradient (grad.grad). 2) propagate the error to the
		// previous error.

		// The gradient:
		
		// We are calculating the derivative of F with respect
		// to w.  The derivative, F', is a vector, so it's a
		// gradient (stored in `gradients`)
		//
		// F(x, w)  = L(f(x,w)) // from above
		// F'(x, w) = L'(f(x,w)) * f'(x,w)
		//
		// To compute index, i, of the F', we calculate the
		// derivative with respect to w[i].
		//
		// Note that
		//
		// f(x,w) = x[0]*w[0] + x[1]*w[1] + ... + x[n]*w[n]
		//
		// So the derivative wrt w[i] is 
		//
		// df(x,w)/dw[i] = x[i]
		//
		// L'(x) = activator_derivative(x) // look at the code, if you're curious.

		// The inner loop is responsible for back-propagating
		// the error.  Intuitively, we are assigning 'blame'
		// for the error in this layer's output to the
		// elements of the input tensor.
		//
		// The amount of blame we assign to each input for the
		// error in a particular output is proportional to
		// that input's weight for that output.  If the weight
		// for an input is large, it had a large impact on the
		// output, so it more responsible for the resulting
		// error.

		// The errors attributed to each input is the sum of
		// the error it contributed across all the outputs.
		for ( int n = 0; n < out.size.x; n++ ) {
			gradient_t& grad = gradients[n];

			// In `activator()` we saved the value of
			// f(x,w) as `activator_input`, so we are
			// reusing it here to compute L'(f(x,w))
			grad.grad = grad_next_layer( n, 0, 0 ) * activator_derivative( activator_input[n] );

			// We are calculating how much each input
			// contributed to the error.  That
			// contribution is proportional to the
			// weights.
			
			// As in activate() we are "straighting" a
			// tensor `grads_out` to do the produdct with
			// one row of `weights`
			
			// This loops is really just doing a vector-matrix
			// product between the gradient and the weight matrix.

			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ ) {
						int m = map(i, j, z);
						grads_out( i, j, z ) += grad.grad * weights( m, n, 0 );
					}
		}
	}
	
	void fix_weights() {
		// Here, we are updating the weights.  The amount we
		// change the input primarily depends on the gradient
		// and the input value.  We use gradient decent, which
		// means we follow the gradient downward to minimize
		// error.
		//
		// Recall that during back propagation, the input the
		// layer is the error and the derivatives are with
		// respect to the weights.  This means that the
		// gradient points in the direction we should move the
		// weights to reduce the error.
		//
		// We calculated the gradientt in calc_grads(), and
		// proportional to the error (i.e., grad_next_layer)
		// and the derivative of the activator function.  This
		// means that larger errors or steeper slopes causes
		// bigger changes in the weights.
		//
		// The basic update rule is
		//
		// w_new = w - gradient * input
		//
		// This update rule is too aggressive, however, so we
		// add a learning rate, u:
		//
		// w_new = w - gradient * input * u
		//
		// There is a also problem that can arise when the
		// gradient get small: progress toward the minimum can
		// slow.  So we also have a "momentum" term, M:
		//
		// t = gradient + old_gradient*momentum
		// w_new = w - u * input * t
		//
		// Finally, to smooth out the changes in gradient, we
		// add a 'decay' term governed by a decay, D:
		//
		// t = gradient + old_gradient*momentum
		// w_new = w - (u * input * t + D * w)
		//
		// All this complication lives in update_weight()
		// 
		// Since the above needs old_gradient, the gradient
		// tensor has the old and new gradient values in it.
		// update_gradient() updates the old gradient with the
		// new value.
		for ( int n = 0; n < out.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map(i, j, z);
						float& w = weights( m, n, 0 );
						//w = w - grad * input * rate/
						w = update_weight( w, grad, in( i, j, z ) );
					}
			update_gradient( grad );
		}
	}


	virtual ~fc_layer_t(){}
	
	virtual std::string analyze_inequality_with(layer_t* other) {
		auto _other = dynamic_cast<fc_layer_t*>(other);
		throw_assert(_other, "You called 'analyze_inequality_with' without a mismatched layer type")
		std::stringstream out;
		if (this->activator_input.size() != _other->activator_input.size()) {
			out << "Activator_Input sizes don't match: " << DUMP(this->activator_input.size()) << " != " << DUMP(_other->activator_input.size()) << "\n";
		}
		if (this->weights.size != _other->weights.size) {
			out << "Weights sizes don't match: " << DUMP(this->weights.size) << " != " << DUMP(_other->weights.size) << "\n";
		}
		if (this->gradients.size() != _other->gradients.size()) {
			out << "Gradients sizes don't match: " << DUMP(this->gradients.size()) << " != " << DUMP(_other->gradients.size()) << "\n";
		}
		
		out << this->layer_t::analyze_inequality_with(_other);
	
		out << "Diff of ->activator_input: " << diff(this->activator_input, _other->activator_input) << "\n";
		out << "Diff of ->weights: " << diff(this->weights, _other->weights) << "\n";
		out << "Diff of ->gradients: " << diff(this->gradients, _other->gradients) << "\n";
		return out.str();
	}
	
};

class simple_fc_layer_t: public fc_layer_t
{
public:
	simple_fc_layer_t( tdsize in_size, int out_size ) : fc_layer_t(in_size, out_size) {
	}
	void activate(const tensor_t<float>& in ) {
		copy_input(in);
		// Here's the math we are doing here:
		//
		// We'll use some short hand:
		//
		// x = in
		// w = weights
		// L = activator_function  
		// f(x,w) = x*w
		//
		// This layer is a function, F, that maps `x` to `out`
		//
		// out = F(x,w) = L(f(x,w))
		// 
		// f(x,w) = x*w is vector-matrix product which yields a vector.
		// We apply L to each element to get the output vector.
		for ( int n = 0; n < out.size.x; n++ ) {
			activator_input[n] = 0;

			// Both `in` and `weights` are tensors instead
			// of vectors in this code.

			// compute the dot product, we "straighten"
			// them using map() - it converts x,y,z into a
			// linear index for the out vector.
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ ) {
						int m = map(i, j, z);
						activator_input[n] += in( i, j, z ) * weights( m, n, 0 );
					}
			
			
			out( n, 0, 0 ) = activator_function( activator_input[n]);
		}
	}
	
	void calc_grads( const tensor_t<float>& grad_next_layer ) {
		
		memset( grads_out.data, 0, grads_out.size.x * grads_out.size.y * grads_out.size.z * sizeof( float ) );

		// Using the notation from activate():
		//
		// We do two things in the loop below: 1) compute the
		// gradient (grad.grad). 2) propagate the error to the
		// previous error.

		// The gradient:
		
		// We are calculating the derivative of F with respect
		// to w.  The derivative, F', is a vector, so it's a
		// gradient (stored in `gradients`)
		//
		// F(x, w)  = L(f(x,w)) // from above
		// F'(x, w) = L'(f(x,w)) * f'(x,w)
		//
		// To compute index, i, of the F', we calculate the
		// derivative with respect to w[i].
		//
		// Note that
		//
		// f(x,w) = x[0]*w[0] + x[1]*w[1] + ... + x[n]*w[n]
		//
		// So the derivative wrt w[i] is 
		//
		// df(x,w)/dw[i] = x[i]
		//
		// L'(x) = activator_derivative(x) // look at the code, if you're curious.

		// The inner loop is responsible for back-propagating
		// the error.  Intuitively, we are assigning 'blame'
		// for the error in this layer's output to the
		// elements of the input tensor.
		//
		// The amount of blame we assign to each input for the
		// error in a particular output is proportional to
		// that input's weight for that output.  If the weight
		// for an input is large, it had a large impact on the
		// output, so it more responsible for the resulting
		// error.

		// The errors attributed to each input is the sum of
		// the error it contributed across all the outputs.
		for ( int n = 0; n < out.size.x; n++ ){
			gradient_t& grad = gradients[n];

			// In `activator()` we saved the value of
			// f(x,w) as `activator_input`, so we are
			// reusing it here to compute L'(f(x,w))
			grad.grad = grad_next_layer( n, 0, 0 ) * activator_derivative( activator_input[n] );
		}
		
		// We are calculating how much each input
		// contributed to the error.  That
		// contribution is proportional to the
		// weights.
		
		// As in activate() we are "straighting" a
		// tensor `grads_out` to do the produdct with
		// one row of `weights`
		//
		// This loops is really just doing a vector-matrix
		// product between the gradient and the weight matrix.
		for ( int n = 0; n < out.size.x; n++ ) {
			gradient_t& grad = gradients[n];
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ ) {
						int m = map(i, j, z);
						grads_out( i, j, z ) += grad.grad * weights( m, n, 0 );
					}
		}
	}
};
template<class T> T* run_fc(int x, int y, int z,
			    int out_size,
			    int seed) {
	srand(seed);
	tdsize size(x,y,z);
	T * l = new T( size, out_size);
	l->test_me();
	return l;
}


template<class T>
void fc_test(int x, int y, int z, int out, int seed) {					
	fc_layer_t * reference = run_fc<fc_layer_t>(x,y,z,out,seed); 
	fc_layer_t * optimized = run_fc<T>(x,y,z,out,seed); 
	EXPECT_LAYERS_EQ(fc_layer_t, reference, optimized) << "Failure: fc_test("<< x << ", " << y<< ", " << z<< ", " << out << ", " << seed << ");\n";
	delete reference;					
	delete optimized;
}


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

}  // namespace
#endif


	
