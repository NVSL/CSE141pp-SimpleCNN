#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include"CNN/tensor_t.hpp"
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

tensor_t<float> perspective2D(float d) {
	auto r = ident2D();
	r(1,2,0) = 1/d;
	return r;
}

// This applies the inverse of `trans` to `in`.  It works by generating a set
// of sample points (one for each point in the output), appling `trans` to
// them, and then sampling `in` at the resulting points.
tensor_t<float> inv_2Dtransform_nn(const tensor_t<float> & in, const tensor_t<float> & trans, const tdsize & dst_size) {
	throw_assert(in.size.z == dst_size.z, "2Dtransform only works with matched z depths in.size = " << in.size
		     << "; dst_size = " << dst_size);
	throw_assert(trans.size.x == 3 &&
		     trans.size.y == 3 &&
		     trans.size.z == 1, "2Dtransform transform must be 3,3,1.  trans.size = " << trans.size);

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

// take `in` and make it's size match `target_size` by cropping or padding it.  If
// 'letterbox' is true, pad with 0s to match size, otherwise crop.
tensor_t<float> pad_or_crop(const tensor_t<float> & in, const tdsize & target_size, bool letterbox) {
	tdsize scaled_size;
	float scale;
	if (letterbox) {
		scale = MIN((target_size.x + 0.0)/(in.size.x + 0.0),
			    (target_size.y + 0.0)/(in.size.y + 0.0));
	} else {
		scale = MAX((target_size.x + 0.0)/(in.size.x + 0.0),
			    (target_size.y + 0.0)/(in.size.y + 0.0));
	}
	auto scaled = inv_2Dtransform_nn(in, ident2D(), tdsize(in.size.x*scale, in.size.y*scale, in.size.z));

	if (letterbox) {
		tensor_t<float> out(target_size);
		out.paste({
				(target_size.x-scaled.size.x)/2,
				(target_size.y-scaled.size.y)/2,
				0
			}, scaled);
		return out;
	} else {
		return scaled.copy(
			{
				(scaled.size.x-target_size.x)/2,
				(scaled.size.y-target_size.y)/2,
				0
			},target_size);
	}
}

#ifdef INCLUDE_TESTS
#include "util/png_util.hpp"
#include "util/jpeg_util.hpp"

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
	
	TEST_F(CNNTest, transform_image) {
		tensor_t<float> in = load_tensor_from_png("../tests/images/NVSL.png");
		auto scaled = scale_nn(in, {40,40,in.size.z});
		write_tensor_to_png(DEBUG_OUTPUT "NVSL-scale1.png", scaled);

		auto ident = inv_2Dtransform_nn(in, ident2D(), in.size);
		write_tensor_to_png(DEBUG_OUTPUT "NVSL-ident.png", ident);
		
		auto rotate = inv_2Dtransform_nn(in, rotate2D(-30), in.size);
		write_tensor_to_png(DEBUG_OUTPUT "NVSL-rotate.png", rotate);
		
		auto translate = inv_2Dtransform_nn(in, translate2D(20,20), in.size);
		write_tensor_to_png(DEBUG_OUTPUT "NVSL-translate.png", translate);
		
		auto scale = inv_2Dtransform_nn(in, scale2D(.75, 0.75), in.size);
		write_tensor_to_png(DEBUG_OUTPUT "NVSL-scale.png", scale);
		
		auto shear = inv_2Dtransform_nn(in, shear2D(0.1, 0), in.size);
		write_tensor_to_png(DEBUG_OUTPUT "NVSL-shear.png", shear);

		auto perspective = inv_2Dtransform_nn(in, perspective2D(0.5), in.size);
		write_tensor_to_png(DEBUG_OUTPUT "NVSL-perspective.png", perspective);

	}
	
	TEST_F(CNNTest, adjust_image) {
		auto portrait_img = load_tensor_from_jpeg("images/bear.jpg");
		auto landscape_img = load_tensor_from_jpeg("images/bear_rot.jpg");

		tdsize landscape(200,300,3);
		tdsize portrait(300,200,3);
		
		write_tensor_to_png(DEBUG_OUTPUT "ppt.png", pad_or_crop(portrait_img, portrait, true));
		write_tensor_to_png(DEBUG_OUTPUT "lpt.png", pad_or_crop(landscape_img, portrait, true));
		write_tensor_to_png(DEBUG_OUTPUT "plt.png", pad_or_crop(portrait_img, landscape, true));
		write_tensor_to_png(DEBUG_OUTPUT "llt.png", pad_or_crop(landscape_img, landscape, true));
		
		write_tensor_to_png(DEBUG_OUTPUT "ppf.png", pad_or_crop(portrait_img, portrait, false));
		write_tensor_to_png(DEBUG_OUTPUT "lpf.png", pad_or_crop(landscape_img, portrait, false));
		write_tensor_to_png(DEBUG_OUTPUT "plf.png", pad_or_crop(portrait_img, landscape, false));
		write_tensor_to_png(DEBUG_OUTPUT "llf.png", pad_or_crop(landscape_img, landscape, false));

	}
}

#endif
