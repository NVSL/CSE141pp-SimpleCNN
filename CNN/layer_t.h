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
	virtual size_t get_total_memory_size() const {
		return in.get_total_memory_size() + out.get_total_memory_size() + grads_in.get_total_memory_size();
	}
	virtual std::string param_str() const {return "<missing>";}
	virtual std::string kind_str() const {return "<missing>";}
	std::string spec_str() const {
		std::stringstream ss;
		ss << kind_str() << "(" << param_str() << ")";
		return ss.str();
	}
	virtual std::string internal_state() const {
		return "";
	}

	virtual void configure(const tdsize & in_size) {
		in = tensor_t<float>(in_size);
		grads_in = tensor_t<float>(in_size);
	}

	layer_t(const tdsize & in_size, const tdsize & out_size) :  in(in_size), out(out_size), grads_in(in_size) {}
	
	virtual ~layer_t(){}
};
