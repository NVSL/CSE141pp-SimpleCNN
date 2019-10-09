/*
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 */

#include <iostream>
#include <png.h>

#include "tensor_t.h"


tensor_t<float> load_tensor_from_png(char* file_name)
{
        char header[8];    
	int width, height;
	png_infop info_ptr;
	//int number_of_passes;
	png_bytep * row_pointers;

        /* open file and test for it being a png */
        FILE *fp = fopen(file_name, "rb");
	throw_assert(fp, "[read_png_file] File " << file_name << " could not be opened for reading");
	
        fread(header, 1, 8, fp);
        throw_assert(png_sig_cmp((png_const_bytep)header, 0, 8) == 0, "[read_png_file] File " << file_name << " is not recognized as a PNG file");

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
        png_read_update_info(png_ptr, info_ptr);

        /* read file */
	
        throw_assert(!setjmp(png_jmpbuf(png_ptr)),"[read_png_file] Error during read_image");

        row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
        for (int y=0; y<height; y++)
                row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

        png_read_image(png_ptr, row_pointers);

	tensor_t<float> out(width, height, 3);

	for (int y=0; y<height; y++) {
                png_byte* row = row_pointers[y];
                for (int x=0; x<width; x++) {
                        png_byte* ptr = &(row[x*4]);
			out(x,y,0) = (ptr[0]+0.0)/255.0;
			out(x,y,1) = (ptr[1]+0.0)/255.0;
			out(x,y,2) = (ptr[2]+0.0)/255.0;
		}
	}
        for (int y=0; y<height; y++)
                free(row_pointers[y]);
        free(row_pointers);

        fclose(fp);
	return out;
}



void write_tensor_to_png(char* file_name, tensor_t<float> t)
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

        png_set_IHDR(png_ptr, info_ptr, t.size.x, t.size.y,
                     8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
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
                        png_byte* ptr = &(row[x*3]);
			ptr[0] = t(x,y,0) * 255.0;
			ptr[1] = t(x,y,1) * 255.0;
			ptr[2] = t(x,y,2) * 255.0;
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




int main(int argc, char **argv)
{
        auto r = load_tensor_from_png(argv[1]);
	std::cout << r << "\n";
        write_tensor_to_png(argv[2], r);

        return 0;
}
