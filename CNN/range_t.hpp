#pragma once
#include "types.hpp"

struct range_t
{
	int min_x, min_y, min_z;
	int max_x, max_y, max_z;
};

int clamp(int x, int max)
{
	if (x < 0) {
		return 0;
	}
	if (x > max) {
		return max;
	}
	return x;
}

#define ROUND_UP_IDIV(x,y) (((x) + (y) - 1)/(y))
	
range_t map_to_output_impl( int x, int y, int kernel_size, int stride, int depth, const tdsize & size )
{
	return {
			clamp( ROUND_UP_IDIV(x - kernel_size + 1, stride), size.x - 1),
			clamp( ROUND_UP_IDIV(y - kernel_size + 1, stride), size.y - 1),
			0,
			clamp( x / stride, size.x - 1),
			clamp( y / stride, size.y - 1),
			depth - 1,
		};
}

