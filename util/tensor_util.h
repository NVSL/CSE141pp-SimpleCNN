#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include"CNN/tensor_t.h"
#include <algorithm>


#define MAX(x,y) ((x)>(y)? (x) : (y))
#define MIN(x,y) ((x)<(y)? (x) : (y))
#define CLAMP(x, lower, upper) MAX(MIN((x), (upper)), (lower))

tensor_t<float> point2D(float x, float y){
	tensor_t<float> p(3, 1, 1);
	p(0, 0, 0) = x;
	p(1, 0, 0) = y;
	p(2, 0, 0) = 1.0;
	return p;
}


tensor_t<float> ident2D(){
	tensor_t<float> r(3,3,1);
	r(0,0,0) = 1;
	r(1,1,0) = 1;
	r(2,2,0) = 1;
	return r;
}

tensor_t<float> rotate2D(float degrees){
	float radians = degrees/180.0*M_PI;
	
	tensor_t<float> r(3,3,1);
	r(0,0,0) = cos(radians);
	r(1,0,0) = -sin(radians);
	r(0,1,0) = sin(radians);
	r(1,1,0) = cos(radians);
	r(2,2,0) = 1;
	return r;
}

tensor_t<float> translate2D(float x, float y){
	tensor_t<float> r(3,3,1);
	r(0,0,0) = 1;
	r(1,1,0) = 1;
	r(2,2,0) = 1;

	r(0,2,0) = x;
	r(1,2,0) = y;
	return r;
}

tensor_t<float> scale2D(float x, float y) {
	tensor_t<float> r(3,3,1);
	r(0,0,0) = x;
	r(1,1,0) = y;
	r(2,2,0) = 1;
	return r;
}

tensor_t<float> shear2D(float x, float y) {
	auto r = ident2D();
	r(0,1,0) = x;
	r(1,0,0) = y;
	return r;
}

tensor_t<float> inv_affine2D_nn(const tensor_t<float> & in, const tensor_t<float> & trans, const tdsize & dst_size) {
	throw_assert(in.size.z == dst_size.z, "affine2D only works with matched z depths in.size = " << in.size
		     << "; dst_size = " << dst_size);
	throw_assert(trans.size.x == 3 &&
		     trans.size.y == 3 &&
		     trans.size.z == 1, "affine2D transform must be 3,3,1.  trans.size = " << trans.size);

	tensor_t<float> dst(dst_size);

	auto scale = scale2D((in.size.x+0.0)/(dst.size.x+0.0),
			     (in.size.y+0.0)/(dst.size.y+0.0));

	
	TENSOR_FOR(dst, dst_x, dst_y, z) {
		auto p = point2D(dst_x, dst_y);
		auto i = scale.matmul(trans).matmul(p);
		//std::cerr << i << "\n";
		dst(dst_x, dst_y, z) = in(CLAMP(round(i(0,0,0)), 0.0, in.size.x-1.0),
					  CLAMP(round(i(1,0,0)), 0.0, in.size.y-1.0),
					  z);
	}
	//std::cerr << in.size << "\n";
	//std::cerr << dst.size << "\n";
	//std::cerr << scale << "\n";
	
	return dst;
}

tensor_t<float> scale_nn(const tensor_t<float> & in, const tdsize & dst_size) {
	tensor_t<float> dst(dst_size);

	TENSOR_FOR(dst, dst_x, dst_y, dst_z) {
		float x_scale = (dst_size.x+0.0)/(in.size.x+0.0);
		float y_scale = (dst_size.y+0.0)/(in.size.y+0.0);
		float z_scale = (dst_size.z+0.0)/(in.size.z+0.0);

		dst(dst_x, dst_y, dst_z) = in(CLAMP(round((dst_x+0.0)/x_scale), 0.0, in.size.x-1.0),
					      CLAMP(round((dst_y+0.0)/y_scale), 0.0, in.size.y-1.0),
					      CLAMP(round((dst_z+0.0)/z_scale), 0.0, in.size.z-1.0));
	}
	return dst;
}

#ifdef INCLUDE_TESTS
#include "gtest/gtest.h"
#include "util/png_util.h"

namespace CNNTest {

	TEST_F(CNNTest, tensor_scale_nn) {
		tensor_t<float> t1(3,4,5);
		randomize(t1);
		auto r1 = scale_nn(t1, t1.size);
		EXPECT_EQ(t1, r1);
		auto r2 = scale_nn(t1, tdsize(t1.size.x * 2, t1.size.y * 2, t1.size.z * 2));
		auto r3 = scale_nn(t1, t1.size);
		EXPECT_EQ(t1, r3);
	}

		
	TEST_F(CNNTest, tensor_transforms) {
		auto p = point2D(1,0);

		EXPECT_EQ(translate2D(1,1).matmul(p), point2D(2,1));
		EXPECT_EQ(scale2D(2,2).matmul(p), point2D(2,0));
		EXPECT_NEAR(rotate2D(90).matmul(p)(0,0,0), 0,  0.00001);
		EXPECT_NEAR(rotate2D(90).matmul(p)(1,0,0), -1, 0.00001);
		EXPECT_EQ(p, ident2D().matmul(p));
	}
	
	TEST_F(CNNTest, scale_png) {
		tensor_t<float> in = load_tensor_from_png("../tests/images/NVSL.png");
		auto scaled = scale_nn(in, {40,40,in.size.z});
		write_tensor_to_png("NVSL-scale1.png", scaled);

		auto ident = inv_affine2D_nn(in, ident2D(), in.size);
		write_tensor_to_png("NVSL-ident.png", ident);
		
		auto rotate = inv_affine2D_nn(in, rotate2D(-30), in.size);
		write_tensor_to_png("NVSL-rotate.png", rotate);
		
		auto translate = inv_affine2D_nn(in, translate2D(20,20), in.size);
		write_tensor_to_png("NVSL-translate.png", translate);
		
		auto scale = inv_affine2D_nn(in, scale2D(.75, 0.75), in.size);
		write_tensor_to_png("NVSL-scale.png", scale);
		
		auto shear = inv_affine2D_nn(in, shear2D(0.1, 0), in.size);
		write_tensor_to_png("NVSL-shear.png", shear);
		
	}
}

#endif
