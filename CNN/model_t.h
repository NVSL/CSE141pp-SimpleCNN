#pragma once
#include "tensor_t.h"
#include "layer_t.h"
#include "dataset_t.h"
#include <vector>
#include <sstream>

class model_t
{
public:
	std::vector<layer_t*> layers;
	
	void add_layer(layer_t & l) {
		layers.push_back(&l);
	}

	float train(const test_case_t & tc, bool debug=false) {
		return train(tc.data, tc.label, debug);
	}

	float train(const tensor_t<float>& data, const tensor_t<float>& expected, bool debug=false) {
		for ( uint i = 0; i < layers.size(); i++ )
		{
			const tensor_t<float> * d;
			if ( i == 0 ) {
				d = &data;
			} else {
				d = &layers[i - 1]->out;
			}
			if (debug) {
				std::cout << layers[i]->spec_str() << "\n" << "Input: " << *d << "\n";
				std::cout << "Weights: " << layers[i]->internal_state() <<"\n";
			}
			layers[i]->activate(*d);
			if (debug) {
				std::cout << "Output: " << layers[i]->out << "\n";
			}
		}
		
		tensor_t<float> grads = layers.back()->out - expected;

		if (debug) {
			std::cout << "Expected: " << expected <<"\n";
			std::cout << "Error   : " << grads <<"\n";
		}
		for (int i = (int)layers.size() - 1; i >= 0; i-- )
		{
			tensor_t<float> * g;
			
			if ( i == (int)layers.size() - 1 ) {
				g = & grads;
			} else {
				g = &layers[i + 1]->grads_in;
			}
			layers[i]->calc_grads( *g );
			if (debug) {
				std::cout << layers[i]->spec_str() << "\n" << "Gradients: " << *g << "\n";
			}
		}

		for ( uint i = 0; i < layers.size(); i++ )
		{
			layers[i]->fix_weights();
			if (debug) {
				std::cout << layers[i]->spec_str() << "\n" << "Weights: " << layers[i]->internal_state() << "\n";
			}
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

	size_t get_total_memory_size() const {
		size_t sum = 0;

		for(auto &r: layers) {
			sum += r->get_total_memory_size();
		}
		return sum;
	}

	std::string geometry() const {
		std::stringstream ss;
		int i = 0;
		ss << "IN    " << layers[0]->in.size << "\n";
		for(auto &r: layers) {
			auto s = r->get_total_memory_size();
			ss << "L" << i << "  ->" << r->out.size << " " << (s+0.0)/(1024.0) << " kB (" << (s+0.0)/get_total_memory_size()*100.0 << "%) : " << r->spec_str() << "\n"; 
			i++;
		}
	        ss << "Total " << i << ": " << get_total_memory_size()/1024.0 << " kB\n";
		return ss.str();
	}
};

#ifdef INCLUDE_TESTS
#include "gtest/gtest.h"


namespace CNNTest {

	TEST_F(CNNTest, model_output) {
		model_t model;
		
		conv_layer_t  layer1( 1, 5, 8, 0, tdsize(28,28,1) );	
		relu_layer_t  layer2( layer1.out.size );
		pool_layer_t layer3( 2, 2, 0, layer2.out.size );	
		fc_layer_t  layer4(layer3.out.size, 10);
		
		model.add_layer(layer1 );
		model.add_layer(layer2 );
		model.add_layer(layer3 );
		model.add_layer(layer4 );
		model.geometry();
	}
}

#endif
