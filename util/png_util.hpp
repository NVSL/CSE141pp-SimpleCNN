#pragma once
/*
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 */

#include <iostream>
#include <png.h>


tensor_t<double> load_tensor_from_png(const char* file_name)
{
        char header[8];    
	int width, height;
	int color_type;
	png_infop info_ptr;
	//int number_of_passes;
	png_bytep * row_pointers;

        /* open file and test for it being a png */
        FILE *fp = fopen(file_name, "rb");
	throw_assert(fp, "[read_png_file] File " << file_name << " could not be opened for reading");
	
        int r = fread(header, 1, 8, fp);
	throw_assert(r == 8, "short read reading " << file_name << "\n");
        throw_assert(png_sig_cmp((png_bytep)header, 0, 8) == 0, "[read_png_file] File " << file_name << " is not recognized as a PNG file");

        /* initialize stuff */
        png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        throw_assert(png_ptr, "[read_png_file] png_create_read_struct failed");

        info_ptr = png_create_info_struct(png_ptr);
        throw_assert(info_ptr, "[read_png_file] png_create_info_struct failed");

        throw_assert(!setjmp(png_jmpbuf(png_ptr)), "[read_png_file] Error during init_io");

        png_init_io(png_ptr, fp);
        png_set_sig_bytes(png_ptr, 8);

        png_read_info(png_ptr, info_ptr);

        width = png_get_image_width(png_ptr, info_ptr);
        height = png_get_image_height(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);
        png_read_update_info(png_ptr, info_ptr);

        /* read file */
	
        throw_assert(!setjmp(png_jmpbuf(png_ptr)),"[read_png_file] Error during read_image");

        row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
        for (int y=0; y<height; y++)
                row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

        png_read_image(png_ptr, row_pointers);

	int depth;
	switch(color_type) {
	case PNG_COLOR_TYPE_RGB:
		depth = 3;
		break;
	case PNG_COLOR_TYPE_RGB_ALPHA:
		depth = 4;
		break;
	case PNG_COLOR_TYPE_GRAY_ALPHA:
		depth = 2;
		break;
	case PNG_COLOR_TYPE_GRAY:
		depth = 1;
		break;
	default:
		throw_assert(false, "Unknown color type in PNG");
	}
	tensor_t<double> out(width, height, depth);

	for (int y=0; y<height; y++) {
                png_byte* row = row_pointers[y];
                for (int x=0; x<width; x++) {
                        png_byte* ptr = &(row[x*depth]);
			for(int i = 0;i < depth; i++) {
				out(x,y,i) = (ptr[i]+0.0)/255.0;
			}
		}
	}
        for (int y=0; y<height; y++)
                free(row_pointers[y]);
        free(row_pointers);

        fclose(fp);
	return out;
}



void write_tensor_to_png(const char* file_name, tensor_t<double> t)
{
	png_bytep * row_pointers;
        /* create file */
        FILE *fp = fopen(file_name, "wb");
        throw_assert(fp, "[write_png_file] File " << file_name << " could not be opened for writing");


        /* initialize stuff */
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        throw_assert(png_ptr, "[write_png_file] png_create_write_struct failed");

	png_infop info_ptr = png_create_info_struct(png_ptr);
        throw_assert(info_ptr, "[write_png_file] png_create_info_struct failed");

        throw_assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during init_io");

        png_init_io(png_ptr, fp);


        /* write header */
        throw_assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during writing header");

	int color_type;
	switch(t.size.z) {
	case 4:
		color_type = PNG_COLOR_TYPE_RGB_ALPHA;
		break;
	case 3:
		color_type = PNG_COLOR_TYPE_RGB;
		break;
	case 2:
		color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
		break;
	case 1:
		color_type = PNG_COLOR_TYPE_GRAY;
		break;
	default:
		throw_assert(false, "Unexpected tensor depth in png output: " << t.size.z);
		
	}
	
        png_set_IHDR(png_ptr, info_ptr, t.size.x, t.size.y,
                     8, color_type, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        png_write_info(png_ptr, info_ptr);


        /* write bytes */
	throw_assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during writing bytes");
	
	row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * t.size.y);
        for (int y=0; y<t.size.y; y++)
                row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

	for (int y=0; y< t.size.y; y++) {
                png_byte* row = row_pointers[y];
                for (int x=0; x<t.size.x; x++) {
                        png_byte* ptr = &(row[x*t.size.z]);
			for(int i= 0; i < t.size.z; i++) {
				ptr[i] = t(x,y,i) * 255.0;
			}
		}
	}

        png_write_image(png_ptr, row_pointers);

        /* end write */
        throw_assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during end of write");

        png_write_end(png_ptr, NULL);

        /* cleanup heap allocation */
        for (int y=0; y<t.size.y; y++)
                free(row_pointers[y]);
        free(row_pointers);

        fclose(fp);
}


#ifdef INCLUDE_TESTS

namespace CNNTest {

	TEST_F(CNNTest, png_readwrite)
	{
		auto r = load_tensor_from_png("images/NVSL.png");
		write_tensor_to_png(DEBUG_OUTPUT "copied.png", r);
		auto reload = load_tensor_from_png(DEBUG_OUTPUT "copied.png");
		auto d = r - reload;
		TENSOR_FOR(d, x,y,z,b)
			d(x,y,z,b) = fabs(d(x,y,z,b));
		EXPECT_LT(d.max(), 0.01);

		// Gray scale
		auto gray = reload.copy({0,0,1}, {reload.size.x, reload.size.y, 1});
		EXPECT_EQ(gray.size.z, 1);
		write_tensor_to_png(DEBUG_OUTPUT "gray.png", gray);

		// Gray with alpha channel
		auto agray = reload.copy({0,0,2}, {reload.size.x, reload.size.y, 2});
		EXPECT_EQ(agray.size.z, 2);
		write_tensor_to_png(DEBUG_OUTPUT "agray.png", agray);
	}
}

#endif
