#pragma once
#include <cstdint>
#include <fstream>
#include <vector>
#include "CNN/types.h"
#include "CNN/dataset_t.h"
#include "byteswap.h"

uint8_t* read_file( const std::string & f )
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
	return buffer;
}

dataset_t load_mnist(const std::string & images, const std::string & labels)
{
	dataset_t cases;

	uint8_t* train_image = read_file( images );
	uint8_t* train_labels = read_file( labels);

	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	for ( uint i = 0; i < case_count; i++ )
	{
		tensor_t<float> data( 28, 28, 1 );
		tensor_t<float> label_data( 10, 1, 1 );

		uint8_t* img = train_image + 16 + i * (28 * 28);
		uint8_t* label = train_labels + 8 + i;

		for ( int x = 0; x < 28; x++ )
			for ( int y = 0; y < 28; y++ )
				data( x, y, 0 ) = img[x + y * 28] / 255.f;

		for ( int b = 0; b < 10; b++ )
			label_data( b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

		cases.add(data, label_data);
	}
	delete[] train_image;
	delete[] train_labels;

	return cases;
}

#ifdef INCLUDE_TESTS
#include "gtest/gtest.h"


namespace CNNTest {

	TEST_F(CNNTest, mnist_io) {
		auto r = load_mnist("../datasets/mnist/t10k-images-idx3-ubyte",
				    "../datasets/mnist/t10k-labels-idx1-ubyte");
	}
}

#endif
