#pragma once
#include "tensor_t.h"
#include "layer_t.h"
#include <vector>

class model_t
{
public:
	std::vector<layer_t*> layers;
	
	void add_layer(layer_t & l) {
		layers.push_back(&l);
	}

	float train(const tensor_t<float>& data, const tensor_t<float>& expected) {
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


        tensor_t<float> & apply(const tensor_t<float>& data ) const {
		for ( uint i = 0; i < layers.size(); i++ )
		{
			if ( i == 0 )
				layers[i]->activate(data );
			else
				layers[i]->activate(layers[i - 1]->out );
		}
		return layers.back()->out;
	}

};


#ifdef INCLUDE_TESTS
namespace CNNTest{

	TEST_F(CNNTest, model_simple) {
		
		tdsize size(10,10,10);
	}
}
#endif
