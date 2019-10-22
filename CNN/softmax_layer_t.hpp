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
		return (o.in == in) && (o.grads_in == grads_in) && (o.out == out);
	}

	bool operator!=(const softmax_layer_t & o) const {
		return !(*this == o);
	}
	
	void activate(const tensor_t<float>& in ) {
		copy_input(in);
		float s = 0;
		TENSOR_FOR(in, x,y,z) {
			s += exp(in(x,y,z));
		}
		TENSOR_FOR(in, x,y,z) {
			out(x,y,z) = exp(in(x,y,z))/s;
		}
	}

	void fix_weights()
	{

	}

	void calc_grads(const tensor_t<float>& grad_next_layer )
	{
		throw_assert(grad_next_layer.size == in.size, "mismatched input");
		TENSOR_FOR(in, ix,iy,iz) {
			grads_in(ix,iy,iz) = 0;
			TENSOR_FOR(in, jx,jy,jz) {
				float k = ix==jx && iy == jy && iz == jz ? 1.0 : 0.0;
				grads_in(ix,iy,iz) += out(ix,iy,iz)*(k - out(jx,jy,jz))*grad_next_layer(ix,iy,iz);
			}
		}
	}
};

class opt_softmax_layer_t : public softmax_layer_t
{
public:
	opt_softmax_layer_t(const tdsize & in_size ): softmax_layer_t(in_size){}
};


#ifdef INCLUDE_TESTS
namespace CNNTest{

	TEST_F(CNNTest, softmax_simple) {
		tensor_t<float> data(4,1,1);

		randomize(data);

		softmax_layer_t layer(data.size);
		layer.activate(data);
		EXPECT_LE(data.max(), 1.0);
		EXPECT_GE(data.min(), 0.0);
		float s = 0;
		TENSOR_FOR(layer.out, x,y,z) s += layer.out(x,y,z);
		EXPECT_FLOAT_EQ(s, 1.0);

		
		tensor_t<float> next_grads(4,1,1);
		layer.calc_grads(next_grads);
	}

	
}  // namespace
#endif
