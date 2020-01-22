#include <iostream>
#include "CNN/canela.hpp"

using namespace std;

model_t * build_two_layer(const dataset_t & ds)  {
	model_t * model = new model_t();
	fc_layer_t *layer1 = new fc_layer_t(ds.data_size, 100);
	fc_layer_t *layer2 = new fc_layer_t(layer1->out.size, ds.label_size.x);
	model->add_layer(*layer1 );
	model->add_layer(*layer2 );
	return model;
}

model_t * build_perceptron(const dataset_t & ds)  {
	model_t * model = new model_t();
	fc_layer_t *layer2 = new fc_layer_t(ds.data_size, ds.label_size.x);
	model->add_layer(*layer2 );
	return model;
}

model_t * build_conv(const dataset_t & ds)  {
	model_t * model = new model_t();

	layer_t *layer1 = new conv_layer_t(1, 5, 5, 1, ds.data_size);
	layer_t * layer2 = new relu_layer_t( layer1->out.size );
	layer_t *layer3 = new fc_layer_t(layer2->out.size, ds.label_size.x);
	model->add_layer(*layer1 );
	model->add_layer(*layer2 );
	model->add_layer(*layer3 );
	return model;
}

model_t * build_deep(const dataset_t & ds) {
	model_t * model = new model_t();

	conv_layer_t *layer1 = new conv_layer_t ( 1, 5, 8, 0, ds.data_size);
	relu_layer_t *layer2 = new relu_layer_t ( layer1->out.size );
	pool_layer_t *layer3 = new pool_layer_t ( 2, 2, 0, layer2->out.size );				// 24 24 8 -> 12 12 8
			                 
	conv_layer_t *layer4 = new conv_layer_t ( 1, 3, 10, 0, layer3->out.size );			// 12 12 6 -> 10 10 10
	relu_layer_t *layer5 = new relu_layer_t ( layer4->out.size );
	pool_layer_t *layer6 = new pool_layer_t ( 2, 2, 0, layer5->out.size );				// 10 10 10 -> 5 5 10
			                 
	fc_layer_t   *layer7 = new fc_layer_t   ( layer6->out.size, ds.label_size.x );					// 4 * 4 * 16 -> 10

	model->add_layer(*layer1 );
	model->add_layer(*layer2 );
	model->add_layer(*layer3 );
	model->add_layer(*layer4 );
	model->add_layer(*layer5 );
	model->add_layer(*layer6 );
	model->add_layer(*layer7 );

	return model;
}

model_t * build_mid(const dataset_t & ds) {
	model_t * model = new model_t();

	conv_layer_t *layer1 = new conv_layer_t ( 1, 5, 8, 0, ds.data_size);
	relu_layer_t *layer2 = new relu_layer_t ( layer1->out.size );
	pool_layer_t *layer3 = new pool_layer_t ( 2, 2, 0, layer2->out.size );				// 24 24 8 -> 12 12 8
			                 
#if(0)
	conv_layer_t *layer4 = new conv_layer_t ( 1, 3, 10, 0, layer3->out.size );			// 12 12 6 -> 10 10 10
	relu_layer_t *layer5 = new relu_layer_t ( layer4->out.size );
	pool_layer_t *layer6 = new pool_layer_t ( 2, 2, 0, layer5->out.size );				// 10 10 10 -> 5 5 10
#endif		                 
	fc_layer_t   *layer7 = new fc_layer_t   ( layer3->out.size, ds.label_size.x );					// 4 * 4 * 16 -> 10

	model->add_layer(*layer1 );
	model->add_layer(*layer2 );
	model->add_layer(*layer3 );
	//model->add_layer(*layer4 );
	//model->add_layer(*layer5 );
	//model->add_layer(*layer6 );
	model->add_layer(*layer7 );

	return model;
}

model_t * build_cache_1(const dataset_t & ds)  { 
	model_t * model = new model_t();
	layer_t *layer1 = new conv_layer_t(1, 5, 5, 1, ds.data_size);
	layer_t *layer2 = new relu_layer_t( layer1->out.size );
	layer_t *layer3 = new dropout_layer_t(layer2->out.size, 0.5);
	layer_t *layer4 = new fc_layer_t(layer3->out.size, ds.label_size.x);
	model->add_layer(*layer1 );
	model->add_layer(*layer2 );
	model->add_layer(*layer3 );
	model->add_layer(*layer4 );
	return model;
}

float simple(const std::string &model_name, const std::string & ds, int scale_factor) 
{
	
	dataset_t *train = new dataset_t;
	dataset_t *test = new dataset_t;

	if (ds == "mnist") {
		*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/mnist-train.dataset", 200 * scale_factor);
		*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/mnist-test.dataset", 200 * scale_factor);
	} else if (ds == "emnist") {
		*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/emnist-byclass-train.dataset", 200 * scale_factor);
		*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/emnist-byclass-test.dataset", 200 * scale_factor);
	} else if (ds == "cifar10") {
		*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar10_data_batch_1.dataset", 100 * scale_factor);
		*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar10_test_batch.dataset", 100 * scale_factor);
	} else if (ds == "cifar100") {
		*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar100_train.dataset", 100 * scale_factor);
		*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar100_test.dataset", 100 * scale_factor);
	} else if (ds == "imagenet") {
		*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/imagenet.dataset", 1 * scale_factor);
		*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/imagenet.dataset", 1 * scale_factor);
	} else {
		throw_assert(false, "Illegal dataset: " << ds << "\n");
	}
	       

	model_t * model = nullptr;
	
	if (model_name == "perceptron") {
		model = build_perceptron(*train);
	} else if (model_name == "two_layer") { 
		model = build_two_layer(*train);
	} else if (model_name == "conv") {
		model = build_conv(*train);
	} else if (model_name == "mid") {
		model = build_mid(*train);
	} else if (model_name == "cache-1") {
		model = build_cache_1(*train);
	} else if (model_name == "deep") {
		model = build_deep(*train);
	} else {
		throw_assert(false, "Illegal model name: " << model_name << "\n");
	}


	std::cout << model->geometry() << "\n";
	std::cout << "Training data size: " << (train->get_total_memory_size()+0.0)/(1024*1024)  << " MB" << std::endl;
	std::cout << "Training cases    : " << train->size() << std::endl;
	std::cout << "Testing data size : " << (test->get_total_memory_size()+0.0)/(1024*1024)  << " MB" << std::endl;
	std::cout << "Testing cases     : " << test->size() << std::endl;

	if (!scale_factor) {
		return 0;
	}
		       
	for ( test_case_t& t : *train ) {
		model->train(t);
	}
	

	int correct  = 0, incorrect = 0;
	for ( test_case_t& t : *test ) {
		tensor_t<float>& out = model->apply(t.data);
		
		tdsize guess = out.argmax();
	        tdsize answer = t.label.argmax();
		if (guess == answer) {
			correct++;
		} else {
			incorrect++;
		}
	}

	float total_error = (correct+0.0)/(correct+ incorrect +0.0);
	std::cout << "Accuracy: " << total_error << ": " << correct << "/" << correct + incorrect << "\n";
	return total_error;
}


#ifndef EXCLUDE_MAIN
int main(int argc, char*argv[]) {
	simple(argv[1], argv[2], atoi(argv[3]));
	return 0;
}

#endif
