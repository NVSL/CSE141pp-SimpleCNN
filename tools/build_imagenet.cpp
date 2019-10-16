#include <iostream>
#include "CNN/dataset_t.h"
#include "util/jpeg_util.h"

int main()
{
	dataset_t imagenet;

	auto r = load_tensor_from_jpeg("P1050082.jpg");
	tensor_t<float> label(1000,1,1);
	label(1,0,0) = 1.0;

	imagenet.add(r, label);
	
	std::ofstream out ("imagenet.dataset",std::ofstream::binary);
	imagenet.write(out);
	out.close();

	return 0;
}
