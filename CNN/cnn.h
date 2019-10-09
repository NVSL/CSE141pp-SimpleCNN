#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

#include "tensor_t.h"
#include "optimization_method.h"
#include "fc_layer.h"
#include "pool_layer_t.h"
#include "relu_layer_t.h"
#include "conv_layer_t.h"
#include "dropout_layer_t.h"

#include "model_t.h"

float train( std::vector<layer_t*>& layers, tensor_t<float>& data, tensor_t<float>& expected )
{
	for ( uint i = 0; i < layers.size(); i++ )
	{
		if ( i == 0 ) {
			layers[i]->activate(data);
		} else {
			layers[i]->activate(layers[i - 1]->out);
		}
	}

	tensor_t<float> grads = layers.back()->out - expected;

	for (int i = (int)layers.size() - 1; i >= 0; i-- )
	{
		if ( i == (int)layers.size() - 1 ) {
			layers[i]->calc_grads( grads );
		} else {
			layers[i]->calc_grads(layers[i + 1]->grads_in );
		}
	}

	for ( uint i = 0; i < layers.size(); i++ )
	{
		layers[i]->fix_weights();
	}

	float err = 0;
	for ( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++ )
	{
		float f = expected.data[i];
		if ( f > 0.5 )
			err += abs(grads.data[i]);
	}
	return err * 100;
}


void forward( std::vector<layer_t*>& layers, tensor_t<float>& data )
{
	for ( uint i = 0; i < layers.size(); i++ )
	{
		if ( i == 0 )
			layers[i]->activate(data );
		else
			layers[i]->activate(layers[i - 1]->out );
	}
}
