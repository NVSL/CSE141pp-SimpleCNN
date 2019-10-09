#pragma once
#include "tensor_t.h"

enum class layer_type
{
	none = 0,
	conv,
	fc,
	relu,
	pool,
	dropout_layer
};


class layer_t
{
public:
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<float> grads_in;

	void copy_input(const tensor_t<float>& in ) {
		throw_assert(this->in.size == in.size, "Passed incorrectly-sized inputs to layer");
		this->in = in;
	}

	virtual void activate(const tensor_t<float>& in) = 0;
	
	virtual void fix_weights() = 0;
	virtual void calc_grads(tensor_t<float>& grad_next_layer ) = 0;

	layer_t(const tdsize & in_size, const tdsize & out_size) :  in(in_size), out(out_size), grads_in(in_size) {}

	virtual ~layer_t(){}
};
