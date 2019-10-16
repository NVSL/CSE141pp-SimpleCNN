#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "CNN/cnn.h"
#include "CNN/dataset_t.h"
#include "util/mnist.h"

using namespace std;

int main()
{

	dataset_t imagenet = dataset_t::read("../datasets/imagenet/imagenet.dataset");

	model_t model;

	conv_layer_t layer1( 4, 11, 96, 0, tdsize(224,224,3));
	pool_layer_t layer2( 2, 3, 0, layer1.out.size );	
	relu_layer_t layer3( layer2.out.size );
	
	conv_layer_t layer4( 1, 5, 256, 2, layer3.out.size );
	pool_layer_t layer5( 2, 3, 0, layer4.out.size );	
	relu_layer_t layer6( layer5.out.size );
	
	conv_layer_t layer7( 1, 3, 384, 1, layer6.out.size );
	relu_layer_t layer8( layer7.out.size );

	conv_layer_t layer9( 1, 3, 384, 1, layer8.out.size );
	relu_layer_t layer10( layer9.out.size );
	
	pool_layer_t layer11( 2, 3, 0, layer10.out.size );	
	relu_layer_t layer12( layer11.out.size );

	fc_layer_t layer13( layer12.out.size, 4096 );
	dropout_layer_t layer14(layer13.out.size, 0.5);
	fc_layer_t layer15( layer14.out.size, 4096 );
	dropout_layer_t layer16(layer15.out.size, 0.5);
	fc_layer_t layer17( layer16.out.size, 1000 );
	//softmax_layer_t layer18(layer17.out.size);
	
	model.add_layer(layer1 );
	model.add_layer(layer2 );
	model.add_layer(layer3 );
	model.add_layer(layer4 );
	model.add_layer(layer5 );
	model.add_layer(layer6 );
	model.add_layer(layer7 );
	model.add_layer(layer8 );
	model.add_layer(layer9 );
	model.add_layer(layer10 );
	model.add_layer(layer11 );
	model.add_layer(layer12 );
	model.add_layer(layer13 );
	model.add_layer(layer14 );
	model.add_layer(layer15 );
	model.add_layer(layer16 );
	model.add_layer(layer17 );
	//model.add_layer(layer18 );
	
	std::cout << model.geometry() << "\n";

	auto i = imagenet.begin();
	model.train_batch(imagenet, i, 1);
	
	return 0;
}
