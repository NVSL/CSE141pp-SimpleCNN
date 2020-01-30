#pragma once
#include"canela.hpp"

class opt_conv_layer_t : public conv_layer_t
{
public:
	opt_conv_layer_t( uint16_t stride,
			  uint16_t kernel_size, 
			  uint16_t kernel_count,
			  double pad,
			  tdsize in_size
		) : conv_layer_t(stride, kernel_size, kernel_count, pad, in_size) {}

};

class opt_dropout_layer_t : public dropout_layer_t
{
public:
	opt_dropout_layer_t( tdsize in_size, float p_activation ) : dropout_layer_t(in_size, p_activation) {}
};

class opt_fc_layer_t : public fc_layer_t
{
public:
	opt_fc_layer_t( tdsize in_size, int out_size ) : fc_layer_t(in_size, out_size) {}
			
};

class opt_pool_layer_t: public pool_layer_t
{
public:
	opt_pool_layer_t( uint16_t stride, uint16_t filter_size, double pad, tdsize in_size ) : pool_layer_t(stride, filter_size, pad, in_size) {}
};

class opt_relu_layer_t : public relu_layer_t
{
public:
	opt_relu_layer_t(const tdsize & in_size ): relu_layer_t(in_size){}
};


class opt_softmax_layer_t : public softmax_layer_t
{
public:
	opt_softmax_layer_t(const tdsize & in_size ): softmax_layer_t(in_size){}
};
