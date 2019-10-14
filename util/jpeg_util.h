#include "CNN/tensor_t.h"
#include <stdio.h>


#include "jpeglib.h"

// Adapted from https://gist.github.com/PhirePhly/3080633
tensor_t<float>
load_tensor_from_jpeg(const char * filename)
{
	/* This struct contains the JPEG decompression parameters and pointers to
	 * working space (which is allocated as needed by the JPEG library).
	 */
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
		
	/* We use our private extension JPEG error handler.
	 * Note that this struct must live as long as the main JPEG parameter
	 * struct, to avoid dangling-pointer problems.
	 */
	FILE * infile = NULL;		/* source file */
	JSAMPARRAY buffer;		/* Output row buffer */
	int row_stride;		/* physical row width in output buffer */
  
	/* In this example we want to open the input file before doing anything else,
	 * so that the setjmp() error recovery below can assume the file is open.
	 * VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
	 * requires it in order to read binary files.
	 */

	if ((infile = fopen(filename, "r")) == NULL) {
		throw_assert(false, "Can't open " << filename << "\n");
	}

	/* Step 1: allocate and initialize JPEG decompression object */
	cinfo.err = jpeg_std_error(&jerr);	
	jpeg_create_decompress(&cinfo);

	/* Step 2: specify data source (eg, a file) */

	jpeg_stdio_src(&cinfo, infile);

	/* Step 3: read file parameters with jpeg_read_header() */

	(void) jpeg_read_header(&cinfo, TRUE);

	(void) jpeg_start_decompress(&cinfo);

	/* We may need to do some setup of our own at this point before reading
	 * the data.  After jpeg_start_decompress() we have the correct scaled
	 * output image dimensions available, as well as the output colormap
	 * if we asked for color quantization.
	 * In this example, we need to make an output work buffer of the right size.
	 */ 
	/* JSAMPLEs per row in output buffer */
	row_stride = cinfo.output_width * cinfo.output_components;
	/* Make a one-row-high sample array that will go away when done with image */
	buffer = (*cinfo.mem->alloc_sarray)
		((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

	/* Step 6: while (scan lines remain to be read) */
	/*           jpeg_read_scanlines(...); */

	tensor_t<float> out(row_stride/3, cinfo.output_height, 3);
	for (int y=0; y<out.size.y; y++) {

		(void) jpeg_read_scanlines(&cinfo, buffer, 1);

		for (int x=0; x< out.size.x; x++) {
			for(int i = 0;i < 3; i++) {
				out(x,y,i) = (buffer[0][x*3 + i]+0.0)/255.0;
			}
		}
	}

	/* Step 7: Finish decompression */

	(void) jpeg_finish_decompress(&cinfo);

	/* Step 8: Release JPEG decompression object */

	/* This is an important step since it will release a good deal of memory. */
	jpeg_destroy_decompress(&cinfo);

	fclose(infile);

	return out;
}

#ifdef INCLUDE_TESTS
#include "gtest/gtest.h"

namespace CNNTest {

	TEST_F(CNNTest, jpeg_read)
	{
		auto r = load_tensor_from_jpeg("images/bear.jpg");
		write_tensor_to_png("bear.png", r);
		auto reload = load_tensor_from_png("bear.png");
		auto d = r - reload;
		TENSOR_FOR(d, x,y,z)
			d(x,y,z) = fabs(d(x,y,z));
		EXPECT_LT(d.max(), 0.01);
	}
}

#endif
