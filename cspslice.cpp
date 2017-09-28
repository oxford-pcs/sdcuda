#include "cspslice.h"

#include <stdio.h>
#include <vector>
#include <numeric>

#include <cuda_runtime.h>
#include <cufft.h>

#include "ccube.h"
#include "regions.h"

hspslice::hspslice(hcube* datacube, Complex* p_data, rectangle region, double wavelength) {
	hspslice::datacube = datacube;
	hspslice::p_data = p_data;
	hspslice::wavelength = wavelength;
	hspslice::region = region;
	hspslice::n_elements = hspslice::region.x_size * hspslice::region.y_size;
}

int hspslice::crop(long start_x, long start_y, long new_size_x, long new_size_y) {
	for (int row = 0; row < new_size_y; row++) {
		hspslice::datacube->memcpyhh(&p_data[(row*new_size_x)], &p_data[((row + start_y)*hspslice::region.x_size) + start_x], new_size_x*sizeof(Complex));
	}
	rectangle new_region = rectangle(start_x, start_y, new_size_x, new_size_y);
	hspslice::region = new_region;
	hspslice::n_elements = hspslice::region.x_size * hspslice::region.y_size;
	return 0;
}


dspslice::dspslice(dcube* datacube, Complex* p_data, rectangle region, double wavelength) {
	dspslice::datacube = datacube;
	dspslice::p_data = p_data;
	dspslice::wavelength = wavelength;
	dspslice::region = region;
	dspslice::n_elements = dspslice::region.x_size * dspslice::region.y_size;
}

int dspslice::crop(long start_x, long start_y, long new_size_x, long new_size_y) {
	for (int row = 0; row < new_size_y; row++) {
		dspslice::datacube->memcpydd(&p_data[(row*new_size_x)], &p_data[((row + start_y)*dspslice::region.x_size) + start_x], new_size_x*sizeof(Complex));
	}
	rectangle new_region = rectangle(start_x, start_y, new_size_x, new_size_y);
	dspslice::region = new_region;
	dspslice::n_elements = dspslice::region.x_size * dspslice::region.y_size;
	return 0;
}


