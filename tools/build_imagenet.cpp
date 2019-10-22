#include <iostream>
#include "CNN/dataset_t.hpp"
#include "util/jpeg_util.hpp"
#include "util/tensor_util.hpp"
int main()
{
	dataset_t imagenet;

	std::ifstream in("image-list.txt");

	std::string line;
	while (std::getline(in, line)) {
		auto r = load_tensor_from_jpeg(line.c_str());
		r = pad_or_crop(r, {224,224,3}, true);
		tensor_t<float> label(1000,1,1);
		label(1,0,0) = 1.0;
		imagenet.add(r, label);
	}
	
	std::ofstream out ("imagenet.dataset",std::ofstream::binary);
	imagenet.write(out);
	out.close();

	return 0;
}
