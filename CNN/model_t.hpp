#pragma once
#include "tensor_t.hpp"
#include "layer_t.hpp"
#include "dataset_t.hpp"
#include <vector>
#include <sstream>

class model_t
{
public:
	std::vector<layer_t*> layers;
	
	void add_layer(layer_t & l) {
		layers.push_back(&l);
	}

	// Run one instance forward through the model.
	void forward_one(const tensor_t<double> & data, bool debug) {
		
		for ( uint i = 0; i < layers.size(); i++ )
		{
			const tensor_t<double> * d;
			if ( i == 0 ) { // First layer gets the input instance
				d = &data;
			} else { // the rest get the output of the previous layer.
				d = &layers[i - 1]->out;
			}
			if (debug) {
				std::cout << layers[i]->spec_str() << "\n" << "Input: " << *d << "\n";
				std::cout << "Weights: " << layers[i]->internal_state() <<"\n";
			}
			layers[i]->activate(*d); // Apply the layer to *d (a tensor).  This sets layer[i]->out
			
			if (debug) {
				std::cout << "Output: " << layers[i]->out << "\n";
			}
		}
	}

	void backward(const tensor_t<double> & error, bool debug) {
		// Back propagation is in two phases.

		// First we compute gradients for each layer starting
		// at the output.
		for (int i = (int)layers.size() - 1; i >= 0; i-- )
		{
			const tensor_t<double> * g;
			
			if ( i == (int)layers.size() - 1 ) {
				g = & error;
			} else {
				g = &layers[i + 1]->grads_out;
			}
			layers[i]->calc_grads( *g );
			if (debug) {
				std::cout << layers[i]->spec_str() << "\n" << "Gradients: " << *g << "\n";
			}
		}

		// then we adjust the weights (i.e., parameters) in
		// each layer starting from the input layer.
		for ( uint i = 0; i < layers.size(); i++ )
		{
			layers[i]->fix_weights();
			if (debug) {
				std::cout << layers[i]->spec_str() << "\n" << "Weights: " << layers[i]->internal_state() << "\n";
			}
		}
	}

	int train_batch(const dataset_t & ds, dataset_t::iterator & start, int count, bool debug=false) {
		throw_assert(false, "THis code doesn't terminate correctly.");
		tensor_t<double> error(layers.back()->out.size);
		int i = 0;
		while(start != ds.end() && i < count) {
			forward_one(start->data, debug);
			error = error + layers.back()->out - start->label;
			i++;
			start++;
		}
		
		if (debug) {
			std::cout << "Error   : " << error <<"\n";
		}

		backward(error, debug);
		return i;
	}
	
	double train(const test_case_t & tc, bool debug=false) {
		return train(tc.data, tc.label, debug);
	}

	double train(const tensor_t<double>& data, const tensor_t<double>& expected, bool debug=false) {

		// Run one instance farward.
		forward_one(data, debug);

		// Compute the error.
		tensor_t<double> error = layers.back()->out - expected;

		if (debug) {
			std::cout << "Expected: " << expected <<"\n";
			std::cout << "Error   : " << error <<"\n";
		}

		// Run the error back through the network to adjust the parameters.
		backward(error, debug);

		// Sum up the error tensor.  I think this code might not be needed anymore.
		double err = 0;
		for ( int i = 0; i < error.size.x * error.size.y * error.size.z; i++ )
		{
			double f = expected.data[i];
			if ( f > 0.5 )
				err += abs(error.data[i]);
		}
		return err * 100;
	}


        tensor_t<double> & apply(const tensor_t<double>& data ) const {
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
