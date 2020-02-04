#pragma once
#include "tensor_t.hpp"

enum class layer_type
{
	none = 0,
	conv,
	fc,
	relu,
	pool, 
	dropout_layer
};

#define DUMP(x) #x " = " << x
#define RAND_R(x,y) ((x)+ (rand() % ((y)-(x))))
#define RAND_LARGE(x) RAND_R(x/2, x)


class layer_t
{
public:
	// All Layers have these inputs/outputs.
	tensor_t<double> in;
	tensor_t<double> out;
	tensor_t<double> grads_out;

	// These are key methods a layer must implement.
	virtual void activate(const tensor_t<double>& in) = 0;
	virtual void fix_weights() = 0;
	virtual void calc_grads(const tensor_t<double>& grad_next_layer ) = 0;

	// Everything else is utility functions.
	void change_batch_size(int new_batch_size) {
		//only valid after training
		tdsize new_in_size = in.size;
		tdsize new_out_size = out.size;
		new_in_size.b = new_batch_size;
		new_out_size.b = new_batch_size;
		std::cout << "this->in.size: " << in.size << std::endl;
		in.resize(new_in_size);
		std::cout << "Resized this->in to " << in.size << std::endl;
		out.resize(new_out_size);
	}

	void copy_input(const tensor_t<double>& in ) {
		throw_assert(this->in.size == in.size, "Passed incorrectly-sized inputs to layer " << this->in.size << " == " << in.size);
		this->in = in;
	}

	virtual size_t get_total_memory_size() const {
		return in.get_total_memory_size() + out.get_total_memory_size() + grads_out.get_total_memory_size();
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
		in = tensor_t<double>(in_size);
		grads_out = tensor_t<double>(in_size);
	}

	layer_t(const tdsize & in_size, const tdsize & out_size) :  in(in_size), out(out_size), grads_out(in_size) {}
	
	virtual ~layer_t(){}

	virtual std::string analyze_inequality_with(layer_t* other) {
		std::stringstream out;
		if (this->in.size != other->in.size) {
			out << "Input sizes don't match: " << DUMP(this->in.size) << " != " << DUMP(other->in.size) << "\n";
		}
		if (this->out.size != other->out.size) {
			out << "Output sizes don't match: " << DUMP(this->out.size) << " != " << DUMP(other->out.size) << "\n";
		}
		
		if (this->grads_out.size != other->grads_out.size) {
			out << "Grads sizes don't match: " << DUMP(this->grads_out.size) << " != " << DUMP(other->grads_out.size) << "\n";
		}
		out << "Diff of ->in: " << diff(this->in, other->in) << "\n";
		out << "Diff of ->out: " << diff(this->out, other->out) << "\n";
		out << "Diff of ->grads_out: " << diff(this->grads_out, other->grads_out) << "\n";
		return out.str();
	}


	void test_me() {
		tensor_t<double> in(this->in.size);
		randomize(in);
		tensor_t<double> next_grads(this->out.size);
		randomize(next_grads);
		activate(in);
		calc_grads(next_grads);
		fix_weights();
	}

	void test_activate() {
		tensor_t<double> _in(this->in.size);
		randomize(_in);
		activate(_in);
	}

	void test_calc_grads() {
		tensor_t<double> _out(this->out.size);
		randomize(_out);
		calc_grads(_out);
	}

	void test_fix_weights() {
		randomize(this->grads_out);
		fix_weights();
	}

};


// Customized assertion formatter for googletest
template<class T>
::testing::AssertionResult AssertLayersEqual(const char* m_expr,
					     const char* n_expr,
					     T * m,
					     T * n) {
	if (*m == *n) return ::testing::AssertionSuccess();

	return ::testing::AssertionFailure() << "Here's what's different. '#' denotes a position where your result is incorrect.\n" << m->analyze_inequality_with(n);
}

#define ASSERT_LAYERS_EQ(T, a,b) ASSERT_PRED_FORMAT2(AssertLayersEqual<T>, a,b)
#define EXPECT_LAYERS_EQ(T, a,b) EXPECT_PRED_FORMAT2(AssertLayersEqual<T>, a,b)
	
static inline void run_layer(layer_t & l) {
	tensor_t<double> in(l.in.size);
	randomize(in);
	tensor_t<double> next_grads(l.out.size);
	randomize(next_grads);
	l.activate(in);
	l.calc_grads(next_grads);
	l.fix_weights();
}
