#pragma once
#include "layer_t.hpp"

class softmax_layer_t : public layer_t
{
public:
	softmax_layer_t(const tdsize & in_size )
		:
		layer_t(in_size, in_size)
	{
	}

	std::string kind_str() const {
		return "softmax";
	}
	std::string param_str() const {
		std::stringstream ss;
		return ss.str();
	}

	bool operator==(const softmax_layer_t & o) const {
		return (o.in == in) && (o.grads_out == grads_out) && (o.out == out);
	}

	bool operator!=(const softmax_layer_t & o) const {
		return !(*this == o);
	}
	
	void activate(tensor_t<double>& in ) {
		copy_input(in);
		double s = 0;
		TENSOR_FOR(in, x,y,z,b) {
			s += exp(in(x,y,z,b));
		}
		TENSOR_FOR(in, x,y,z,b) {
			out(x,y,z,b) = exp(in(x,y,z,b))/s;
		}
	}

	void fix_weights()
	{

	}

	void calc_grads(const tensor_t<double>& grad_next_layer )
	{
		throw_assert(grad_next_layer.size == in.size, "mismatched input");
		TENSOR_FOR(in, ix,iy,iz,ib) {
			grads_out(ix,iy,iz,ib) = 0;
			TENSOR_FOR(in, jx,jy,jz,jb) {
				double k = ix==jx && iy == jy && iz == jz && ib == jb ? 1.0 : 0.0;
				grads_out(ix,iy,iz,ib) += out(ix,iy,iz,ib)*(k - out(jx,jy,jz,jb))*grad_next_layer(ix,iy,iz,ib);
			}
		}
	}
};


#ifdef INCLUDE_TESTS
namespace CNNTest{

	TEST_F(CNNTest, softmax_simple) {
		tensor_t<double> data(4,1,1);

		randomize(data);

		softmax_layer_t layer(data.size);
		layer.activate(data);
		EXPECT_LE(data.max(), 1.0);
		EXPECT_GE(data.min(), 0.0);
		double s = 0;
		TENSOR_FOR(layer.out, x,y,z,b) s += layer.out(x,y,z,b);
		EXPECT_FLOAT_EQ(s, 1.0);

		
		tensor_t<double> next_grads(4,1,1);
		layer.calc_grads(next_grads);
	}

	
}  // namespace
#endif

