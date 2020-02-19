#include <iostream>
#include "CNN/dataset_t.hpp"
#include "util/jpeg_util.hpp"
#include "util/tensor_util.hpp"
int main(int argc, char *argv[])
{
	dataset_t imagenet;

	if (argc < 3) {
		std::cerr << "Usage: build_imagenet.exe <output image size> <output file name>\n";
		exit(1);
	}
	
	int size = atoi(argv[1]);
	
	std::ifstream in("image-list.txt");

	std::string line;
	while (std::getline(in, line)) {
		auto r = load_tensor_from_jpeg(line.c_str());
		r = pad_or_crop(r, {size,size,3}, true);
		tensor_t<double> label(1000,1,1);
		label(1,0,0) = 1.0;
		imagenet.add(r, label);
	}
	
	std::ofstream out (argv[2],std::ofstream::binary);
	imagenet.write(out);
	out.close();

	return 0;
}
