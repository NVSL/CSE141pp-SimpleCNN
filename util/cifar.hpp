#pragma once
#include <cstdint>
#include <fstream>
#include <vector>
#include "CNN/types.hpp"
#include "CNN/dataset_t.hpp"
#include "byteswap.hpp"

uint8_t* read_file( const std::string & f, size_t & length )
{
	std::ifstream file( f.c_str(), std::ios::binary | std::ios::ate );
	std::streamsize size = file.tellg();
	file.seekg( 0, std::ios::beg );

	if ( size == -1 ) {
		std::cerr << "Couldn't open " << f << "\n";
		exit(1);
	}

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size );
	length = size;
	return buffer;
}

dataset_t load_cifar(const std::string & file, bool cifar100=false, unsigned int max_frames = std::numeric_limits<unsigned int>::max())
{
	dataset_t cases;
	size_t len;
	uint8_t* fdata = read_file(file, len);
	uint8_t* l = fdata;
#define CIFAR10_FRAME_SIZE 3073
#define CIFAR100_FRAME_SIZE 3074
	if (cifar100) {
		throw_assert((len % CIFAR100_FRAME_SIZE) == 0, "Incorrect format for cifar100");
	} else {
		throw_assert((len % CIFAR10_FRAME_SIZE) == 0, "Incorrect format for cifar10");
	}
				
	uint case_count = cifar100 ? len/CIFAR100_FRAME_SIZE: len/CIFAR10_FRAME_SIZE; // bytes per label + image

	case_count = std::min(case_count, max_frames);

	
	for ( uint i = 0; i < case_count; i++ )
	{
		tensor_t<double> data( 32, 32, 3 );
		tensor_t<double> label_data( cifar100 ? 100: 10 , 1, 1 );

		if (cifar100) {
			l++; // eat the coarse category
		}
		label_data(*l++, 0,0) = 1;

		for(int z = 0; z < 3; z++) {
			for(int y = 0; y < 32; y++) {
				for(int x = 0; x < 32; x++) {
					data(x,y,z) = (*l++ + 0.0)/255.0;
				}
			}
		}
		cases.add(data, label_data);
	}
	delete[] fdata;

	return cases;
}

#ifdef INCLUDE_TESTS

#include "png_util.hpp"
namespace CNNTest {

	TEST_F(CNNTest, cifar_io) {
		auto r = load_cifar("../datasets/cifar/cifar-10-batches-bin/test_batch.bin", false);
		write_tensor_to_png("output/cifar10-concorde.png", r.test_cases[3].data);
		auto s = load_cifar("../datasets/cifar/cifar-100-binary/train.bin", true);
		write_tensor_to_png("output/cifar100-concorde.png", s.test_cases[6].data);
	}
}

#endif
