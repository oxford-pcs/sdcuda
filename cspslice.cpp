#include "cspslice.h"

#include <stdio.h>
#include <vector>
#include <numeric>

#include <cuda_runtime.h>
#include <cufft.h>

#include "ccube.h"
#include "regions.h"

spslice::spslice(cube* datacube, Complex* p_data, rectangle region, double wavelength) {
	spslice::datacube = datacube;
	spslice::p_data = p_data;
	spslice::wavelength = wavelength;
	spslice::region = region;
	spslice::n_elements = spslice::region.x_size * spslice::region.y_size;
}

int spslice::crop(long start_x, long start_y, long new_size_x, long new_size_y) {
	for (int row = 0; row < new_size_y; row++) {
		spslice::datacube->memcpyhh(&p_data[(row*new_size_x)], &p_data[((row + start_y)*spslice::region.x_size) + start_x], new_size_x*sizeof(Complex));
	}
	rectangle new_region = rectangle(start_x, start_y, new_size_x, new_size_y);
	spslice::region = new_region;
	spslice::n_elements = spslice::region.x_size * spslice::region.y_size;
	return 0;
}